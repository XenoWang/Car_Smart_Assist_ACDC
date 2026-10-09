# 常规目标与道路障碍物识别

当前已实现 ACDC 八类交通目标与 Lost & Found 道路杂物候选的联合 YOLO 检测。
新增类别来自标注小障碍实例；不能把“没有类别框”解释成“道路无障碍”，也不能认定覆盖所有未知物体。
输出通过 `object_detection_classes` 与 `road_obstacle_detection_available` 明确能力边界。
已有能见度自编码器评估的是图像可用性，不能直接当作未知道路障碍检测器。

当前使用同一模型输出交通目标与杂物候选，后续补道路／路径上下文：

1. 常规类别检测：车辆、行人等，继续使用现有 YOLO。
2. 杂物候选检测：已增加 road_obstacle 类并联合训练；道路／可行驶区域的预测分割仍待实现。
3. 后续风险融合：结合行驶路径、距离、时序稳定性与置信度评估。路边陌生物体不一定危险，
   停在车道上的常规车辆也可能阻塞道路。当前没有可靠路径／距离监督，不编造这些量。

杂物框检测已实现，路径关系分支仍待实现；当前没有将候选框直接等同于自车路径阻塞。
所需标注至少包含 RGB、道路区域、障碍物区域及忽略区域，最好有序列和深度。
训练还需要包含没有障碍的道路负样本，降低雨水反光、积雪、阴影、车道线造成的误报。

候选数据：

- [Lost & Found](https://arxiv.org/abs/1609.04653)：道路小障碍物、道路区域标注，
  适合掉落货物分支训练；已从 [埃斯林根大学公开镜像](https://huggingface.co/datasets/iis-esslingen/LostAndFoundDataset)
  下载至 `data/external/lost_and_found/`，保留原始压缩包和官方目录结构。
- [RoadObstacle21 / SegmentMeIfYouCan](https://segmentmeifyoucan.com/datasets)：
  用于独立道路障碍评估；大部分测试真值不公开，不适合作为唯一训练来源。

Lost & Found 已获取，RoadObstacle21 尚未下载。Lost & Found 实际核对 2,239 组图片／标注，
官方 train/test 为 1,036/1,203；11,195 个 PNG 完整性检查通过，无缺失配对或尺寸不一致。
来源版本、文件 SHA256 见数据目录的 `download_manifest.json`，检查报告见
`artifacts/reports/detection/lost_and_found_data/integrity.json`。该数据已接入联合检测训练与 pipeline。
日常清洗通过统一 `scripts/clean_data.py --config configs/data/cleaning_lost_and_found.yaml` 执行，
损坏过滤逻辑集中在 `data/preprocessing.py`，清单为 `data/processed/manifests/lost_and_found.json`。
清洗仅过滤损坏图片；文件内容、其他标注／压缩包／说明文件均保留，原始文件不删除。
ACDC 已有分割真值可补充道路区域监督，检测真值仅覆盖既定八类；
其不确定区域不是掉落货物标签，KITTI 的 DontCare 也不是异常障碍类别。

取得数据后的验证要求：按序列／拍摄地点分组，另外留出训练中没有出现的障碍物类型；
道路与障碍分支分别报告区域召回、误报及小物体漏检，恶劣天气另行分组评估。
阈值在 calibration 上确定，最终 test 不参与选模或阈值调整。
只有经过该验证后才考虑接入接管辅助，不能仅凭类别“异常”直接推断碰撞风险。

## 已实施的联合策略与验证

- ACDC 原四分保持不变；Lost & Found 官方 train 按地点分为 861/88/87，官方 test 1,203 张留出。
- 新增 ID 8 的 road_obstacle 类。只从 labelTrainIds=2 与实例 ID 交集提取框，所有有效小框保留；
  背景／忽略类型不转换成杂物。
- 按来源屏蔽未标注的背景类别损失，Lost & Found 仅监督已知 ROI；明确正样本按类别互斥监督。
  不将杂物定义为所有阻塞车辆，普通车辆阻塞仍需要路径／距离／时序判断。
- 旧八类分类输出在扩展时复制，从已训练模型初始化；ACDC 回放与杂物数据一起训练。
  关闭会破坏 ROI 对齐的空间增强，保留颜色与轻度模糊；按部署单类别 NMS 做验证和评估。
- 测试 ROI 的障碍 mAP50 71.48%、F1 69.12%，仅代表已标注范围。原八类 mAP50 26.91%，
  比旧基线 33.02% 下降；车头误报和极小障碍漏检仍存在，不能宣称模型已可靠。
- 推理不使用真值 ROI，未实现可行驶区域分割、类型独立留出验证及恶劣天气杂物评估。
  详细结果和预测对照图见 `artifacts/reports/detection/yolo11n_acdc_laf_v2/`。
