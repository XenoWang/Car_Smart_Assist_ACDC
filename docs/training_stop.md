# 统一训练 stop / continue / fresh

单独模型与串行训练管线采用同一含义：

| 动作 | 行为 |
|---|---|
| `--stop` | 只提出停止请求。当前 epoch 完成训练、验证和保存后退出75；不启动下一轮或后续模型。 |
| `--continue` | 清除本次停止请求，从配对 last 恢复训练状态并继续原累计目标；预算已完成不会追加轮数。 |
| `--fresh` | 保留旧产物后，从配置的初始化权重重新训练，不继承旧进度与优化器状态。 |

前台按 `Ctrl+C` 和后台创建 `STOP` 文件也会提出同样的轮末停止请求。没有 quit 动作。

`last` 对应最后完整轮，保存模型和优化器等恢复状态；`best` 按原验证指标选优，非改善轮停止时不会以 last 冒充 best。检查点缺失或身份不配对时，continue 会拒绝启动。SWAD 手停时保留 last 与 single_best，未导出的平均模型不伪写为 best。

## 训练管线

从项目根使用 `.venv`：

```powershell
.venv\Scripts\python.exe scripts\train_pipeline.py --fresh
.venv\Scripts\python.exe scripts\train_pipeline.py --stop
.venv\Scripts\python.exe scripts\train_pipeline.py --continue
```

命令中的 fresh 和 continue 会实际启动训练；本说明不表示代理已执行过它们。管线默认顺序为天气必要 preflight、天气注意力、能见度 AE、YOLO。参数 `--weather-config`、`--visibility-config`、`--detection-config` 选择现有配方，轮数和其他训练参数沿用配方。

默认输出为 `artifacts/training/model_pipeline/current/`，其中 state.json 记录活动模型、步骤终态、last/best 路径、身份与日志。`--run-dir` 可以选择另一专用 `artifacts/training/` 子目录；三个控制命令须使用同一 run-dir。fresh 将旧 current 整体归档到 history；continue 跳过已完成模型，只续未完成步骤。训练配置输出重定向到该独立目录，不替换默认部署模型，也不自动发布。

stop 命令返回仅表示请求已经写入，实际训练退出并保存后 state 才变为 PAUSED。父进程阻塞等待子进程终态，不轮询训练分数。配置、源码或已记录检查点改变时拒绝 continue，保留产物供核查。

原 full_800_20261022 已由用户暂停，此入口不会接管或恢复旧任务，也不改原监督脚本/冻结 spec。

## 单独模型

对应训练命令使用同一套互斥参数，例如：

```powershell
.venv\Scripts\python.exe scripts\train_visibility.py --config configs/model/visibility.yaml --stop
.venv\Scripts\python.exe scripts\train_visibility.py --config configs/model/visibility.yaml --continue
.venv\Scripts\python.exe scripts\train_visibility.py --config configs/model/visibility.yaml --fresh
```

天气注意力、YOLO、历史 CNN 与历史金字塔/消融入口也传播这些控制动作。多任务感知与 advisory 训练脚本仍为占位，没有训练循环。

`--stop-file` 指定共享标记；未指定时 STOP 位于各入口检查点/实验目录。continue/fresh 只清除此控制器自己的停止标记。管线子进程不会清除父进程在启动期间刚提出的共享停止请求。

普通自动恢复命令保留原来的兼容行为。需要明确“只继续原目标，不追加预算”时使用 `--continue`。天气训练必须采用独立实验输出配置；受保护默认部署目录仍拒绝训练写入。

已由主代理亲自审查接口、源码和用例并修正。本轮定向42项、全量850项全部通过，0失败/错误/跳过；全量含8项真实CPU/CUDA检查点集成。测试期间76项冻结源码、测试和受保护产物SHA保持一致。证据见 `artifacts/test_runs/training_stop_20261023/round2/`。

实际训练控制用例使用CPU小型/合成数据，注意力评分为受控指标；未正式训练、未恢复原800轮任务，不据此宣称精度提升或任意YOLO梯度积累的逐步等价。旧CNN缺原目标、旧套件缺组进度时显式continue拒绝推测恢复。

当前工作区位于 `pipeline_stop_test`，本轮改动尚未提交/合并。按全局规则由用户提交，再执行已授权的合并。
