"""ACDC 帧 + 外部距离标签的联合数据集。

职责:
    - 以 ACDC 图像为视觉输入，挂接 build_distance_labels.py 产出的距离标签
    - 返回 image + boxes + distances + 路况标签；没有距离标注的帧要正确标记为无效而不是填 0
    - 处理样本不均衡：有距离标注的帧通常远少于纯分割帧

依赖: data.datasets.base, data.label_mapping
被谁调用: datamodule（只有距离任务的 head 会消费这部分监督）
"""
