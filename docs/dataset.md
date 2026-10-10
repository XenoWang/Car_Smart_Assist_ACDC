# 数据说明

## 目录与文件

| 数据集 | 目录 | 使用文件 |
|---|---|---|
| ACDC | `data/raw/acdc/` | RGB 图像、参考图、原生标注 |
| KITTI | `data/external/kitti/` | image_2、label_2、calib |
| Pixel Accurate | `data/external/pixel_accurate_depth_benchmark/pixel_accurate_depth_benchmark/` | `rgb_left_8bit.zip` |
| Lost & Found | `data/external/lost_and_found/` | leftImg8bit、gtCoarse |

## ACDC

从 ACDC 官方入口申请数据，下载后按包内目录解压到 `data/raw/acdc/`。
图像主包为 `rgb_anon_trainvaltest.zip`；语义与检测标注分别为 `gt_trainval.zip`、`gt_detection_trainval.zip`。
ACDC 的使用范围和许可按数据提供方条款执行。

天气 RGB 路径：

```text
rgb_anon/{fog,rain,snow,night}/{train,val,test}/{sequence}/*_rgb_anon.png
rgb_anon/{fog,rain,snow,night}/{train_ref,val_ref,test_ref}/{sequence}/*_rgb_ref_anon.png
```

主条件目录用于天气标签；参考图用于能见度重建训练。
GOPRxxxx 与 GP01xxxx 等章节属于同一录制组，划分时整体处理。
语义标签使用 Cityscapes 19 类；检测框按配置中的八类映射导出。
ACDC 不提供本项目需要的积水深度、雪厚、米制可见距离、真实接管行为和单目目标距离真值。

## KITTI

使用目标检测基准中的左目图片、目标标签和相机标定：

```text
training/image_2/
training/label_2/
training/calib/
```

标签格式和后续距离约定见 [标签规范](label_spec.md)。距离训练和标签构造入口目前还没实现。

## Pixel Accurate

保留原始 ZIP，天气入口直接读取 `rgb_left_8bit.zip` 中的成员。
识别文件名：

```text
scene{编号}_{day|night}_{clear|fog数值|rain数值}_{帧编号}.png
```

fog／rain 与 night 分别形成标签；文件名中的数值保留为来源元数据，模型目前不输出物理雨量或雾距离。
同一个 scene 整体划分。增强配置使用 scene 1–2 训练、scene 4 验证、scene 3 校准。

## Lost & Found

RGB 与标签目录：

```text
leftImg8bit/{train,test}/{scene}/
gtCoarse/{train,test}/{scene}/
```

保留 labelIds、labelTrainIds、instanceIds 与原说明文件。
检测数据准备从实例标注和障碍标签的交集生成框，按拍摄地点分组；官方 test 不参与训练。
未标注区域不当作完整的背景负例，RGB 与标签几何尺寸保持一致。

## 清洗与处理产物

清洗统一运行 `scripts/clean_data.py`，配置位于 `configs/data/cleaning*.yaml`。
过滤清单在 `data/processed/manifests/`，检查报告在 `artifacts/reports/cleaning/`。
invalid 只表示确认损坏的项；suspect 保留。原始图像、标签和 ZIP 不修改。
ZIP 损坏项以 `archive::member` 标识。

天气缓存位于 `data/processed/weather/`。检测准备入口为 `scripts/prepare_detection.py`，输出位置由对应数据 YAML 指定。
标签与输出字段见 [标签规范](label_spec.md)，运行命令见 [README](../README.md)。
