"""下载/校验 ACDC (Adverse Conditions Dataset with Correspondences) 原始数据。

职责:
    - 从 ETH 官方渠道拉取 ACDC 压缩包；支持断点续传与 sha256 校验
    - 按 fog / night / rain / snow 四个子集分别落盘到 data/raw/acdc/<subset>/
    - 幂等：已存在且校验通过的目录直接跳过
    - 不在这里做任何解析或转换，保持 raw 层只读、可复现

依赖: car_smart_assist.utils.io, car_smart_assist.utils.logging
被谁调用: 人工执行 `python scripts/download_acdc.py --config configs/data/acdc.yaml`
"""
