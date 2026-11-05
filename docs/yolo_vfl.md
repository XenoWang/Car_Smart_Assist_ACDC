# 联合检测的可选VFL分类损失

默认配置继续使用BCE。VFL仅在显式实验配置中启用，不增加推理头，也不修改默认权重、输入尺寸、阈值/NMS或既有输入输出提速路径。实现完成不等于已经获得检测精度提升。

## 本项目采用的机制

p=sigmoid(logit)，q为现有TAL分配器生成的质量软目标：

- q>0：权重为q。
- q=0：权重为alpha×p^gamma。
- 将权重乘以逐元素BCEWithLogits(logit,q)，再沿用来源/ROI可监督掩码、前景互斥与来源损失权重。

负例权重不detach，保留对logit的梯度。返回未约减的[B,anchors,classes]张量，由原判据继续sum／target_scores_sum、乘分类gain。CIoU、DFL、分配器、正例归一化和三项公开损失均保留。半精度输入使用关闭autocast的FP32计算；FP64输入不降精度。

这里是Varifocal Loss权重对现有TAL软目标的适配；q并非原论文直接以IoU构造的完整IACS监督。本实现不是完整VFNet，也不包含星形特征或额外框细化分支。[原论文](https://arxiv.org/abs/2008.13367)和[作者实现](https://github.com/hyz-xmaster/VarifocalNet/blob/master/mmdet/models/losses/varifocal_loss.py)提供机制依据，论文增益不能直接作为本项目结果。

## 配置与恢复

配置位于yolo_detection顶层，而不是train下：

    classification_loss:
      mode: vfl
      alpha: 0.75
      gamma: 2.0

顶层配置缺省、null或空映射保持BCE；显式BCE使用mode: bce。底层损失构造只接受None或带显式mode的映射，不把空容器静默当作关闭。VFL的alpha/gamma必须是真实、有限、非负数，不接受bool或字符串；未知键和未知mode拒绝。BCE不接受额外alpha/gamma。0.75/2.0仅为起始配方，不是已经验证的最优值。

非默认配方写入model_selection_contract.training_classification_loss，并由既有运行元数据、验证总结和format5训练状态传播。该字段是恢复身份，不改变选模评分公式。切换BCE/VFL或改变VFL参数须新开训练目录；同目录恢复会拒绝身份变化。默认BCE不新增合同字段，旧BCE实例缺新字段时继续原行为。不得把缺失VFL配方的旧状态当成VFL完整恢复。

## 独立对照配置

- configs/model/yolo_vfl_baseline.yaml：BCE控制组。
- configs/model/yolo_vfl_trial.yaml：VFL候选组。

两组固定CUDA:0，从相同初始化出发，数据/种子/960输入/batch4/学习率/增强/后处理一致，准备80轮且关闭早停的同预算筛选；只有损失和输出身份不同。80轮配置不代表已经批准或启动正式训练。两组enabled_for_inference=false，不改变默认部署路由。

现行来源等权选模fitness保持，九类AP作为候选比较主分，同时报告交通八类、道路障碍物、夜间和尺寸桶召回/误报。选模目标权重取舍与VFL是不同变量，本轮不一起更改。所有候选在独立calibration上校准后才比较操作点，不能沿用不匹配阈值，不能用反复查看的开发test宣称全新独立泛化。

应先通过手算损失/梯度、掩码、旧实例、混合精度、真实微型网络反向和恢复身份回归，再单独批准正式训练。低质量正例也会被质量权重减弱，是否帮助小目标/弱类必须以真实配对结果判断。
