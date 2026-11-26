import math

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Polygon, box

from src.common.config import load_config, path_setting
from src.data.rasterize import coverage_analytic, coverage_supersample, fill_center
from src.data.roi import RoiWindow, square_roi, utm_to_wgs84, wgs84_to_utm


def test_projection_roundtrip():
    lon, lat = 113.4925, 23.1735
    x, y = wgs84_to_utm(lon, lat)
    assert 700000 < x < 800000 and 2_500_000 < y < 2_600_000
    lo, la = utm_to_wgs84(x, y)
    assert abs(lo - lon) < 1e-9 and abs(la - lat) < 1e-9


def test_roi_window_arbitrary_shape():
    roi = Polygon([(750000.3, 2560000.7), (750100.9, 2560020.2), (750060.1, 2560140.5)])
    w = RoiWindow.from_roi(roi)
    assert w.origin == (math.floor(750000.3 - 16), math.floor(2560000.7 - 16))
    assert w.shape[0] % 32 == 0 and w.shape[1] % 32 == 0
    assert w.resolution == 0.20
    xmin, ymin, xmax, ymax = w.bounds
    assert xmax >= roi.bounds[2] + 16 and ymax >= roi.bounds[3] + 16
    px = w.world_to_pixel(np.array([[w.origin[0], w.origin[1]]]))
    assert np.allclose(px, [[0, w.shape[0]]])           # 左下角 → (col 0, row H)


def test_analytic_coverage_exact_area():
    rng = np.random.default_rng(1)
    for _ in range(20):
        w, L = rng.uniform(0.1, 1.2), rng.uniform(2, 30)     # 0.1–1.2 px 宽的窄条
        p = affinity.rotate(box(-L / 2, -w / 2, L / 2, w / 2), rng.uniform(0, 180), origin=(0, 0))
        p = affinity.translate(p, 32.3, 31.7)
        cov = coverage_analytic([[np.asarray(p.exterior.coords)]], 64, 64)
        assert cov.sum() == pytest.approx(p.area, rel=1e-6)   # float32 输出
        assert cov.min() >= 0 and cov.max() <= 1


def test_analytic_more_accurate_than_ss4():
    rng = np.random.default_rng(2)
    polys = []
    for _ in range(25):
        p = affinity.rotate(box(-8, -0.3, 8, 0.3), rng.uniform(0, 180), origin=(0, 0))
        polys.append([np.asarray(affinity.translate(p, rng.uniform(10, 54), rng.uniform(10, 54)).exterior.coords)])
    ref = coverage_supersample(polys, 64, 64, ss=32)
    ea = np.abs(coverage_analytic(polys, 64, 64) - ref).mean()
    e4 = np.abs(coverage_supersample(polys, 64, 64, ss=4) - ref).mean()
    assert ea < e4


def test_hole_and_out_of_bounds():
    ring = [np.array([[-10, -10], [40, -10], [40, 40], [-10, 40], [-10, -10]], float),
            np.array([[10, 10], [10, 20], [20, 20], [20, 10], [10, 10]], float)]
    cov = coverage_analytic([ring], 32, 32)
    assert cov[15, 15] == 0 and cov[5, 5] == 1 and cov[25, 30] == 1
    assert fill_center([ring], 32, 32)[15, 15] == 0


def test_channel_contract():
    cfg = load_config("data")
    assert len(cfg["channels"]) == 36
    names = [c["name"] for c in cfg["channels"]]
    assert len(set(names)) == 36
    for t in ("turn_L", "turn_S", "turn_R", "turn_U", "turn_FL", "turn_FR", "z_order_mean", "z_order_span"):
        assert t in names


@pytest.fixture(scope="module")
def dev_feat():
    import pyogrio
    from src.data.roi import read_roi_features
    gp = path_setting("gpkg_dev")
    sig = pyogrio.read_dataframe(gp, layer="信号灯点位", columns=["分类代码"])
    x, y = wgs84_to_utm(sig.geometry.x.values, sig.geometry.y.values)
    return read_roi_features(gp, RoiWindow.from_roi(square_roi(float(x[4]), float(y[4]), 80)))


@pytest.mark.gpkg
def test_rasterize_real_roi(dev_feat):
    from src.data.rasterize import rasterize
    s = rasterize(dev_feat)
    T = {n: s.tensor[k] for k, n in enumerate(s.names)}
    assert s.tensor.shape[0] == 36 and s.tensor.dtype == np.float32
    assert T["road_surface"].mean() > 0.05 and T["roi_mask"].mean() > 0.3
    assert T["white_solid"].max() > 0.5 and (T["udf_marking"] < 1).mean() > 0.01
    am = T["arrow_mask"] > 0
    assert am.any() and np.allclose(np.hypot(T["arrow_cos"][am], T["arrow_sin"][am]), 1, atol=1e-5)
    ax = np.hypot(T["marking_axis_cos2"], T["marking_axis_sin2"])
    assert ax.max() <= 1 + 1e-5
    assert np.all(T["z_order_mean"] == 0)
    # 窄线未被硬二值化：覆盖率存在大量 (0, 1) 中间值
    w = T["white_dashed"] + T["white_solid"]
    assert ((w > 0.05) & (w < 0.95)).sum() > 0.3 * (w > 0.05).sum()


@pytest.mark.gpkg
def test_arrow_heading_convention(dev_feat):
    """箭头朝向约定：头部最宽断面位于朝向前端。"""
    from shapely.geometry import LineString
    road = dev_feat.road
    a = road[road["分类代码"] == "arrow_straight"]
    assert len(a)
    front = []
    for geom, hd in zip(a.geometry.values, a["heading"].values):
        c = geom.centroid
        r = affinity.rotate(geom, -math.degrees(hd), origin=c, use_radians=False)
        x0, y0, x1, y1 = r.bounds
        xs = np.linspace(x0, x1, 41)[1:-1]
        wid = [r.intersection(LineString([(x, y0 - 1), (x, y1 + 1)])).length for x in xs]
        front.append((xs[int(np.argmax(wid))] - x0) / (x1 - x0) > 0.5)
    assert np.mean(front) > 0.9


@pytest.mark.gpkg
def test_tokens(dev_feat):
    from src.data.tokenize import TOKEN_CLASSES, tokenize
    tk = tokenize(dev_feat)
    assert 0 < tk.n <= 512
    cls = [TOKEN_CLASSES[c] for c in tk.cls[tk.mask]]
    assert "arrow" in cls and "road_boundary" in cls
    arrows = tk.mask & (tk.cls == 0)
    assert np.allclose(np.linalg.norm(tk.heading[arrows], axis=1), 1, atol=1e-6)
    assert tk.turns[arrows].sum(1).min() >= 1
    marks = tk.mask & np.isin(tk.cls, [1, 2, 6])
    assert 0.08 < np.median(tk.size[marks, 0]) < 0.35       # 实测漆宽量级
