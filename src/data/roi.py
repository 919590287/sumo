"""任意 ROI 的米制投影与 GPKG 只读提取（docs/DESIGN.md §3.1）。

- 全市统一 EPSG:32649（UTM 49N）米制坐标；窗口左下角取整全局偏移 (X_0, Y_0)；
- 选区外自动扩展 16 m 上下文缓冲；GPKG 只读（pyogrio bbox + 空间索引）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from src.common.config import load_config

_T_FWD = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
_T_INV = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)


def wgs84_to_utm(lon, lat):
    return _T_FWD.transform(lon, lat)


def utm_to_wgs84(x, y):
    return _T_INV.transform(x, y)


@dataclass
class RoiWindow:
    """一个推理/训练窗口：任意形状 ROI（UTM）+ 16 m 缓冲 + 固定 0.20 m/px 栅格。"""
    roi: BaseGeometry                     # 全局 UTM 选区多边形
    origin: tuple[float, float]           # 窗口左下角取整全局偏移
    shape: tuple[int, int]                # (H, W) 像素
    resolution: float = 0.20
    buffer_m: float = 16.0

    @classmethod
    def from_roi(cls, roi: BaseGeometry, pad_multiple: int = 32, cfg: dict | None = None) -> "RoiWindow":
        cfg = cfg or load_config("data")
        res, buf = cfg["resolution_m"], cfg["context_buffer_m"]
        xmin, ymin, xmax, ymax = roi.bounds
        x0, y0 = math.floor(xmin - buf), math.floor(ymin - buf)
        w = math.ceil((xmax + buf - x0) / res)
        h = math.ceil((ymax + buf - y0) / res)
        w = math.ceil(w / pad_multiple) * pad_multiple
        h = math.ceil(h / pad_multiple) * pad_multiple
        return cls(roi, (float(x0), float(y0)), (h, w), res, buf)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.shape
        return (self.origin[0], self.origin[1], self.origin[0] + w * self.resolution,
                self.origin[1] + h * self.resolution)

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """全局 UTM → 连续像素坐标 (col, row)，像素 (r, c) 覆盖 [c, c+1) × [r, r+1)，行向下 = 北在上。"""
        xy = np.asarray(xy, dtype=np.float64)
        h, _ = self.shape
        col = (xy[..., 0] - self.origin[0]) / self.resolution
        row = h - (xy[..., 1] - self.origin[1]) / self.resolution
        return np.stack([col, row], axis=-1)

    def local_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """窗口局部米制坐标（全局减 origin）→ 像素坐标。"""
        xy = np.asarray(xy, dtype=np.float64)
        h, _ = self.shape
        return np.stack([xy[..., 0] / self.resolution, h - xy[..., 1] / self.resolution], axis=-1)


def square_roi(cx: float, cy: float, half: float) -> Polygon:
    return box(cx - half, cy - half, cx + half, cy + half)


@dataclass
class RoiFeatures:
    """窗口（含缓冲）内全部相关要素，已投影至 UTM 全局坐标。"""
    window: RoiWindow
    road: gpd.GeoDataFrame
    stripes: gpd.GeoDataFrame
    separation: gpd.GeoDataFrame
    other_area: gpd.GeoDataFrame
    bus_stop: gpd.GeoDataFrame
    meta: dict = field(default_factory=dict)


def _read(path: Path, layer: str, bbox_wgs, columns) -> gpd.GeoDataFrame:
    try:
        df = pyogrio.read_dataframe(path, layer=layer, bbox=bbox_wgs, columns=columns)
    except Exception as e:  # 图层缺失（小样本 GPKG 可能不含某些图层）
        raise RuntimeError(f"读取 {layer} 失败：{e}") from e
    return df.to_crs(load_config("data")["crs_metric"]) if len(df) else df.set_crs(
        load_config("data")["crs_wgs84"], allow_override=True).to_crs(load_config("data")["crs_metric"])


def read_roi_features(gpkg: Path, window: RoiWindow) -> RoiFeatures:
    """按窗口外包框（含缓冲）只读提取要素，并计算箭头 UTM 朝向。"""
    cfg = load_config("data")
    L = cfg["layers"]
    xmin, ymin, xmax, ymax = window.bounds
    lons, lats = utm_to_wgs84(np.array([xmin, xmax, xmin, xmax]), np.array([ymin, ymin, ymax, ymax]))
    bbox_wgs = (min(lons), min(lats), max(lons), max(lats))
    win = box(xmin, ymin, xmax, ymax)
    road = _read(gpkg, L["road"], bbox_wgs, ["分类代码", "旋转角度", "显示宽度米", "显示长度米"])
    road = road[road.intersects(win)].reset_index(drop=True)
    road["heading"] = np.nan
    arrows = road["分类代码"].isin(list(cfg["arrow_turns"].keys())) & road["旋转角度"].notna()
    if arrows.any():
        road.loc[arrows, "heading"] = _arrow_heading_utm(road[arrows])
    stripes = _read(gpkg, L["stripes"], bbox_wgs, ["分类代码"])
    sep = _read(gpkg, L["separation"], bbox_wgs, ["分类代码"])
    other = _read(gpkg, L["other_area"], bbox_wgs, ["分类代码"])
    bus = _read(gpkg, L["bus_stop"], bbox_wgs, ["分类代码"])
    clip = [d[d.intersects(win)].reset_index(drop=True) for d in (stripes, sep, other, bus)]
    return RoiFeatures(window, road, *clip, meta={"gpkg": str(gpkg), "bbox_wgs84": bbox_wgs})


def _arrow_heading_utm(arrows: gpd.GeoDataFrame) -> np.ndarray:
    """旋转角度（自正北顺时针）→ UTM 网格坐标系下的朝向角（弧度，x 轴逆时针），消除子午线收敛角。"""
    c = arrows.geometry.centroid
    lon, lat = utm_to_wgs84(c.x.values, c.y.values)
    bearing = np.radians(arrows["旋转角度"].values.astype(float))
    d = 1e-5
    dlon = d * np.sin(bearing) / np.cos(np.radians(lat))
    dlat = d * np.cos(bearing)
    x0, y0 = wgs84_to_utm(lon, lat)
    x1, y1 = wgs84_to_utm(lon + dlon, lat + dlat)
    return np.arctan2(np.asarray(y1) - y0, np.asarray(x1) - x0)
