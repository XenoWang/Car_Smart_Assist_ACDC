"""统一评测入口，同时覆盖两个 Stage。

职责:
    - 读 checkpoint，在 val/test 上跑 car_smart_assist.engine.Evaluator
    - 指标：分割 mIoU、路况分类 F1、接管边界 precision/recall（漏报代价 >> 误报）、
            距离 MAE/RMSE、指令文本的 BLEU/ROUGE 与人工抽检表
    - 产出 artifacts/reports/eval_<timestamp>.json + 可读的 markdown 摘要
    - 支持 --stage perception|advisory|all

依赖: car_smart_assist.engine, car_smart_assist.evaluation
被谁调用: 人工执行 / CI
"""
