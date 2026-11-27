"""Layer 1 在线数据管线（docs/DESIGN.md §3.1、§5 Layer 1-2/3）。

每个样本在 DataLoader 进程内在线完成：
1. 多尺寸随机窗口（512–896 px，0.20 m/px，非方形）+ 任意形状 / 长宽比 roi_mask（凸多边形 / 旋转矩形 /
   椭圆 / 完整净区）；以 ``pair_prob`` 概率产出相邻两窗共享核心边界的跨块端口对比样本；
2. 百度风格域随机化：漆面磨损缺损、箭头缺失、逐要素小幅抖动、长标线随机瓦片切缝；
3. 与真实数据同一套 36 通道选择性抗锯齿光栅化与矢量 Token 化；
4. 真值：ROI 裁切子图 → 定长张量 + 成对连接轨迹 + F_2 尺度稠密真值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
import torch
from shapely.affinity import rotate as sh_rotate
from shapely.affinity import translate as sh_translate
from shapely.geometry import Point, Polygon, box

from src.common.config import load_config
from src.data.rasterize import rasterize
from src.data.roi import RoiFeatures, RoiWindow
from src.data.synth.codec import SampleRecord
from src.data.synth.library import SynthLibrary
from src.data.targets import CropError, crop_graph_to_roi, dense_targets, graph_targets, gt_region, node_areas
from src.data.tokenize import tokenize

MARKING_PREFIX = ("white_", "yellow_", "bus_lane", "variable_lane", "reversible_lane", "channelization",
                  "greenway")


@dataclass
class WindowSpec:
    origin: tuple[float, float]     # 场景局部坐标
    shape: tuple[int, int]          # (H, W) px
    roi: Polygon                    # 场景局部坐标
    pair: int = -1                  # 跨块端口对比样本对编号（-1 = 单窗）


# ====================================================================== 窗口与 ROI
def _focus_points(rec: SampleRecord) -> np.ndarray:
    cores = [n.pos[:2] for n in rec.graph.nodes if n.node_type in ("junction_core", "midblock_node")]
    if cores:
        return np.asarray(cores)
    return np.asarray([l.points[len(l.points) // 2, :2] for l in rec.graph.lanes])


def random_roi(inner: Polygon, rng, cfg: dict) -> Polygon:
    if rng.random() < cfg["roi_full_prob"]:
        return inner
    xmin, ymin, xmax, ymax = inner.bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    for _ in range(20):
        kind = rng.integers(3)
        if kind == 0:                                   # 旋转矩形（任意长宽比）
            w = rng.uniform(0.4, 1.3) * (xmax - xmin)
            h = rng.uniform(0.4, 1.3) * (ymax - ymin)
            p = sh_rotate(box(-w / 2, -h / 2, w / 2, h / 2), rng.uniform(0, 180), origin=(0, 0))
        elif kind == 1:                                 # 椭圆
            a = rng.uniform(0.35, 0.75) * (xmax - xmin)
            b = rng.uniform(0.35, 0.75) * (ymax - ymin)
            t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            p = sh_rotate(Polygon(np.c_[a * np.cos(t), b * np.sin(t)]), rng.uniform(0, 180), origin=(0, 0))
        else:                                           # 随机凸包
            pts = rng.uniform([-0.6, -0.6], [0.6, 0.6], size=(int(rng.integers(5, 10)), 2))
            pts *= [(xmax - xmin), (ymax - ymin)]
            p = shapely.convex_hull(shapely.multipoints(pts))
        p = sh_translate(p, cx + rng.normal(0, 0.1 * (xmax - xmin)), cy + rng.normal(0, 0.1 * (ymax - ymin)))
        p = p.intersection(inner)
        if p.geom_type == "Polygon" and p.area >= cfg["roi_min_area_frac"] * inner.area:
            return shapely.set_precision(p, 0.01)
    return inner


def _pair_split(rec: SampleRecord, rng, areas, co: float):
    """跨块对比样本的共享边界：取车道中段点所在坐标，且边界线段不穿过任何节点区域
    （与单窗“被选区触及的节点保持完整”的真值约定一致，避免两种裁切约定在同一输入上冲突）。"""
    lanes = rec.graph.lanes
    for _ in range(12):
        l = lanes[int(rng.integers(len(lanes)))]
        c = l.points[int(rng.integers(4, len(l.points) - 4)), :2]
        axis = int(rng.integers(2))
        m = float(np.round(c[axis], 1))
        o_lo = float(c[1 - axis] + rng.uniform(-0.4, 0.4) * co - co / 2)
        seg = shapely.LineString([(m, o_lo), (m, o_lo + co)] if axis == 0 else [(o_lo, m), (o_lo + co, m)])
        if not any(a.intersects(seg) for a in areas):
            return axis, m, o_lo
    return None


def sample_windows(rec: SampleRecord, rng, cfg: dict, res: float, buf: float, pair_ok: bool = True,
                   areas=None) -> list[WindowSpec]:
    choices = cfg["px_choices"]
    foci = _focus_points(rec)
    c = foci[int(rng.integers(len(foci)))] + rng.normal(0, cfg["center_jitter_m"] / 2, 2)
    split = None
    if pair_ok and rng.random() < cfg["pair_prob"]:
        pa, pb = int(rng.choice(choices)), int(rng.choice(choices))
        po = int(rng.choice(choices))
        ca, cb, co = pa * res - 2 * buf, pb * res - 2 * buf, po * res - 2 * buf
        split = _pair_split(rec, rng, areas or [], co)
    if split is not None:
        axis, m, o_lo = split
        out = []
        for k, (lo, hi, px) in enumerate(((m - ca, m, pa), (m, m + cb, pb))):
            if axis == 0:
                core = box(lo, o_lo, hi, o_lo + co)
                shape = (po, px)
            else:
                core = box(o_lo, lo, o_lo + co, hi)
                shape = (px, po)
            x0, y0 = core.bounds[0] - buf, core.bounds[1] - buf
            out.append(WindowSpec((x0, y0), shape, core, pair=1))
        return out
    H, W = int(rng.choice(choices)), int(rng.choice(choices))
    x0 = math.floor(c[0] - W * res / 2)
    y0 = math.floor(c[1] - H * res / 2)
    inner = box(x0 + buf, y0 + buf, x0 + W * res - buf, y0 + H * res - buf)
    return [WindowSpec((float(x0), float(y0)), (H, W), random_roi(inner, rng, cfg))]


# ====================================================================== 域随机化
def randomize_features(fs_rows: dict, rng, cfg: dict) -> dict:
    dr = cfg
    road = []
    for code, g, hd, w, l in fs_rows["road"]:
        is_mark = code.startswith(MARKING_PREFIX)
        is_arrow = code.startswith("arrow_")
        if is_mark and rng.random() < dr["wear_drop_prob"]:
            continue
        if is_arrow and rng.random() < dr["arrow_missing_prob"]:
            continue
        if code != "road_surface" and dr["jitter_m"] > 0:
            dx, dy = rng.normal(0, dr["jitter_m"], 2)
            ang = rng.normal(0, dr["jitter_rot_deg"])
            g = sh_translate(sh_rotate(g, ang, origin="centroid"), dx, dy)
            if is_arrow and np.isfinite(hd):
                hd = hd + math.radians(ang)
        if code in ("white_solid_stroke", "yellow_solid_stroke") and rng.random() < dr["seam_prob"]:
            for piece in _seam(g, rng, dr):
                road.append((code, piece, hd, w, l))
            continue
        road.append((code, g, hd, w, l))
    stripes = [g for g in fs_rows["stripes"] if rng.random() >= dr["wear_drop_prob"]]
    return {"road": road, "stripes": stripes, "separation": fs_rows["separation"],
            "other_area": fs_rows["other_area"], "bus_stop": fs_rows["bus_stop"]}


def _seam(g, rng, dr) -> list:
    xmin, ymin, xmax, ymax = g.bounds
    t = float(rng.uniform(*dr["seam_tile_m"]))
    if max(xmax - xmin, ymax - ymin) < t:
        return [g]
    gap = float(rng.uniform(*dr["seam_gap_m"]))
    ox, oy = float(rng.uniform(0, t)), float(rng.uniform(0, t))
    cells = [box(x + gap / 2, y + gap / 2, x + t - gap / 2, y + t - gap / 2)
             for x in np.arange(xmin - ox, xmax, t) for y in np.arange(ymin - oy, ymax, t)]
    out = []
    for p in shapely.intersection(g, np.asarray(cells, dtype=object)):
        if p is not None and not p.is_empty and p.area > 1e-3:
            out.append(p)
    return out or [g]


# ====================================================================== 要素 → RoiFeatures
class FeatureIndex:
    """按场景缓存要素外包框，快速筛选窗口内要素。"""

    def __init__(self, rec: SampleRecord):
        fs = rec.features
        self.areas = list(node_areas(rec.graph).values())
        self.rows = {"road": fs.road, "stripes": fs.stripes, "separation": fs.separation,
                     "other_area": fs.other_area, "bus_stop": fs.bus_stop}
        self.bounds = {}
        for k, rows in self.rows.items():
            geoms = [r[1] if isinstance(r, tuple) else r for r in rows]
            self.bounds[k] = shapely.bounds(np.asarray(geoms, dtype=object)) if geoms else np.zeros((0, 4))

    def select(self, xmin, ymin, xmax, ymax) -> dict:
        out = {}
        for k, rows in self.rows.items():
            b = self.bounds[k]
            if len(b) == 0:
                out[k] = []
                continue
            m = (b[:, 2] >= xmin) & (b[:, 0] <= xmax) & (b[:, 3] >= ymin) & (b[:, 1] <= ymax)
            out[k] = [rows[i] for i in np.nonzero(m)[0]]
        return out


def to_roi_features(rows: dict, win: RoiWindow) -> RoiFeatures:
    road = rows["road"]
    rdf = gpd.GeoDataFrame({
        "分类代码": [r[0] for r in road],
        "heading": np.array([r[2] for r in road], float),
        "显示宽度米": np.array([r[3] for r in road], float),
        "显示长度米": np.array([r[4] for r in road], float),
    }, geometry=[r[1] for r in road])
    rdf["旋转角度"] = np.where(np.isfinite(rdf["heading"]), (90.0 - np.degrees(rdf["heading"])) % 360, np.nan)
    st = gpd.GeoDataFrame({"分类代码": ["crosswalk_white_stripe"] * len(rows["stripes"])}, geometry=rows["stripes"])
    sp = gpd.GeoDataFrame({"分类代码": ["separation_structure_path"] * len(rows["separation"])},
                          geometry=rows["separation"])
    oa = gpd.GeoDataFrame({"分类代码": [c for c, _ in rows["other_area"]]}, geometry=[g for _, g in rows["other_area"]])
    bs = gpd.GeoDataFrame({"分类代码": ["bus_stop_poi"] * len(rows["bus_stop"])}, geometry=rows["bus_stop"])
    return RoiFeatures(win, rdf, st, sp, oa, bs, meta={"synthetic": True})


def tokens_dict(tok) -> dict:
    return {"tok_cls": tok.cls, "tok_pts": tok.pts.astype(np.float32), "tok_tan": tok.tan.astype(np.float32),
            "tok_size": tok.size.astype(np.float32), "tok_heading": tok.heading.astype(np.float32),
            "tok_turns": tok.turns.astype(np.float32), "tok_mask": tok.mask}


def build_input(rows: dict, spec: WindowSpec, cfg_data: dict) -> tuple[RoiWindow, dict]:
    win = RoiWindow(spec.roi, spec.origin, spec.shape, cfg_data["resolution_m"], cfg_data["context_buffer_m"])
    feat = to_roi_features(rows, win)
    bev = rasterize(feat).tensor
    tok = tokenize(feat)
    return win, {"bev": bev.astype(np.float16), "hw": np.array(spec.shape, np.int64), **tokens_dict(tok)}


# ====================================================================== Dataset
class SynthWindowDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, root=None, train: bool = True, seed: int = 0, length: int | None = None,
                 randomize: bool = True, pair: bool = True, fixed_px: int | None = None):
        self.lib = SynthLibrary(split, root)
        self.train = train
        self.seed = seed
        self.length = length or len(self.lib)
        self.randomize = randomize
        self.pair = pair
        self.fixed_px = fixed_px
        self.cfg = load_config("synth")
        self.cfg_data = load_config("data")
        self.epoch = 0
        self._cache: tuple[int, SampleRecord, FeatureIndex] | None = None

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return self.length

    def record(self, idx: int) -> tuple[SampleRecord, FeatureIndex]:
        if self._cache is None or self._cache[0] != idx:
            rec = self.lib[idx]
            self._cache = (idx, rec, FeatureIndex(rec))
        return self._cache[1], self._cache[2]

    def __getitem__(self, i: int) -> list[dict]:
        idx = i % len(self.lib)
        rng = np.random.default_rng([self.seed, self.epoch if self.train else 0, i])
        rec, fidx = self.record(idx)
        wcfg = dict(self.cfg["window"])
        if self.fixed_px:
            wcfg["px_choices"] = [self.fixed_px]
        res, buf = self.cfg_data["resolution_m"], self.cfg_data["context_buffer_m"]
        last = None
        for _ in range(6):
            specs = sample_windows(rec, rng, wcfg, res, buf, pair_ok=self.pair and self.train, areas=fidx.areas)
            try:
                return [self._make(rec, fidx, s, rng, idx) for s in specs]
            except CropError as e:
                last = e
        raise RuntimeError(f"样本 {idx} 无法构造合法窗口：{last}")

    def _make(self, rec: SampleRecord, fidx: FeatureIndex, spec: WindowSpec, rng, idx: int) -> dict:
        res = self.cfg_data["resolution_m"]
        H, W = spec.shape
        x0, y0 = spec.origin
        rows = fidx.select(x0, y0, x0 + W * res, y0 + H * res)
        if self.randomize:
            rows = randomize_features(rows, rng, self.cfg["domain_random"])
        win, sample = build_input(rows, spec, self.cfg_data)
        wbox = box(x0 + 0.5, y0 + 0.5, x0 + W * res - 0.5, y0 + H * res - 0.5)
        region = gt_region(rec.graph, spec.roi, wbox)
        sub = crop_graph_to_roi(rec.graph, region, shift=spec.origin, min_piece=self.cfg["window"]["min_piece_m"])
        sample.update(graph_targets(sub))
        sample["dense"] = dense_targets(sub, H, W, res).astype(np.float16)
        roi_local = shapely.affinity.translate(spec.roi, -x0, -y0)
        arrows = []
        for a in rec.meta.get("arrows", []):
            p = np.asarray(a["pos"]) - (x0, y0)
            if roi_local.contains(Point(p)):
                arrows.append({"pos": p, "heading": a["heading"], "turns": a["turns"], "lane": a.get("lane")})
        negs = []
        for ring in rec.meta.get("neg_regions", []):
            q = Polygon(np.asarray(ring) - (x0, y0)).buffer(0).intersection(roi_local)
            if not q.is_empty and q.area > 1.0:
                negs.append(q)
        sample["meta"] = {"idx": idx, "seed": rec.meta.get("seed"), "kind": rec.meta.get("kind"),
                          "origin": spec.origin, "scene_origin": rec.graph.origin, "roi": roi_local,
                          "arrows": arrows, "neg_regions": negs, "pair": spec.pair, "graph": sub}
        return sample


# ====================================================================== collate
def collate(batch: list[list[dict]], max_pairs: int | None = None) -> dict:
    """将 DataLoader 样本列表合并为批次张量。

    当指定 ``max_pairs`` 时，单个 micro-batch 内最多保留 ``max_pairs`` 组跨块对比样本对（超出者仅保留
    首个窗口并置 ``pair_id=-1``），从而把单步最大窗口张量数严格限定为 ``len(batch) + max_pairs``，
    与 ``scripts/calibrate.py`` 的显存预算完全对齐，杜绝随机多对同时命中导致的 OOM。
    """
    items: list[dict] = []
    pair_id: list[int] = []
    pairs_used = 0
    for gi, group in enumerate(batch):
        if len(group) == 2 and (max_pairs is None or pairs_used < max_pairs):
            items.extend(group)
            pair_id.extend([gi, gi])
            pairs_used += 1
        else:
            items.append(group[0])
            pair_id.append(-1)
    Hm = max(int(s["hw"][0]) for s in items)
    Wm = max(int(s["hw"][1]) for s in items)
    out: dict = {}
    bev = np.zeros((len(items), items[0]["bev"].shape[0], Hm, Wm), np.float16)
    dense = np.zeros((len(items), items[0]["dense"].shape[0], Hm // 2, Wm // 2), np.float16)
    for k, s in enumerate(items):
        h, w = s["bev"].shape[1:]
        bev[k, :, :h, :w] = s["bev"]
        dense[k, :, : h // 2, : w // 2] = s["dense"]
    out["bev"] = torch.from_numpy(bev)
    out["dense"] = torch.from_numpy(dense)
    for key in items[0]:
        if key in ("bev", "dense", "meta"):
            continue
        out[key] = torch.from_numpy(np.stack([np.asarray(s[key]) for s in items]))
    out["pair_id"] = torch.tensor(pair_id, dtype=torch.long)
    out["meta"] = [s["meta"] for s in items]
    return out
