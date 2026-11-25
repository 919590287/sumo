# 百度路网转 SUMO

本项目从百度道路要素重建路口车道，并生成可供 SUMO 仿真的路网。
安装依赖后，将 `LANEGEN_GPKG` 指向本地道路要素 GeoPackage，运行 `python -m lanegen.run 名称 经度 纬度` 即可生成车道模型。
运行 `python -m fitgen.run 输出目录 名称=经度,纬度` 可生成拟合路网并进行仿真。
