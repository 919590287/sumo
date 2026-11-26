"""G → plain XML → netconvert（交付网 + 测试台副本）→ 往返核验 的一站式编译管线。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.config import load_config
from src.graph.lane_graph import LaneGraph
from src.sumo_io.netconvert import CompileResult, run_netconvert
from src.sumo_io.roundtrip import RoundTripReport, verify_roundtrip
from src.sumo_io.serializer import PlainXml, SumoSerializer, junction_core_ids


@dataclass
class CompiledGraph:
    plain: PlainXml
    paths: dict[str, Path]
    site: CompileResult
    roundtrip: RoundTripReport | None
    eval_sim: CompileResult | None

    @property
    def ok(self) -> bool:
        return bool(self.site.ok and self.roundtrip is not None and self.roundtrip.ok
                     and (self.eval_sim is None or self.eval_sim.ok))

    @property
    def zero_warning(self) -> bool:
        return not self.site.warnings and (self.eval_sim is None or not self.eval_sim.warnings)


def compile_graph(g: LaneGraph, out_dir: Path, stem: str = "site", with_eval_sim: bool = True,
                  tol: float | None = None) -> CompiledGraph:
    out_dir = Path(out_dir)
    plain = SumoSerializer().serialize(g)
    paths = plain.write(out_dir, stem)
    site = run_netconvert(paths, out_dir / f"{stem}.net.xml")
    rt = verify_roundtrip(g, plain.ids, site.net_path, tol) if site.ok else None
    ev = None
    if with_eval_sim:
        tls = junction_core_ids(g, plain.ids)
        ev = run_netconvert(paths, out_dir / "eval_sim.net.xml", tls_set=tls or None,
                            extra_args=list(load_config("sumo_export").get("eval_tls_args", [])) if tls else None)
    return CompiledGraph(plain, paths, site, rt, ev)
