"""配置加载与继承合并。

职责:
    - 读取 configs/ 下的 yaml，支持 base 继承（子配置覆盖父配置的指定字段）
    - 支持命令行覆盖（--set train.lr=1e-4）用于快速实验
    - 合并后做一次 schema 校验再返回，杜绝「跑起来才发现少个字段」
    - 解析相对路径为绝对路径，并做存在性检查

依赖: car_smart_assist.config.schema
被谁调用: scripts/*, car_smart_assist.engine
"""
