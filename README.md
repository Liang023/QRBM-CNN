# QRBM-CNN 信用卡交易欺诈检测

本项目使用模拟退火辅助的受限玻尔兹曼机（RBM）提取交易特征，与原始特征融合后输入一维CNN，结合Focal Loss和验证集阈值选择完成欺诈检测。

代码仓库：https://github.com/Liang023/QRBM-CNN

## 1. 文件说明

| 文件或目录 | 用途 |
|---|---|
| fin-fraud.py | 主实验脚本：数据处理、RBM预训练、基准模型、CNN训练、阈值选择与结果保存 |
| fraud_results_ulb_fixed/ | 仓库已保存的模型参数、训练日志及结果表；也是代码默认输出目录 |
| README.md | 本运行说明，放在主脚本同级目录 |
| visualize_results.py | 配套提供的绘图脚本，需将下载文件放在主脚本同级目录 |
| creditcard.csv | 运行时需准备的原始数据文件，当前GitHub仓库未包含 |

## 2. 环境准备

需要Python环境，以及numpy、pandas、scikit-learn、PyTorch。绘图还需matplotlib。Python和PyTorch版本应与所用Kaiwu SDK及其PyTorch插件兼容；仓库未提供锁定的依赖版本。

安装通用依赖：

```bash
python -m pip install numpy pandas scikit-learn torch matplotlib
```

Kaiwu SDK与提供kaiwu.torch_plugin的组件请按官方安装说明配置：

- [Kaiwu SDK文档](https://kaiwu-sdk-docs.qboson.com/)
- [Kaiwu PyTorch Plugin文档](https://kaiwu-pytorch-plugin.readthedocs.io/)

运行前检查本项目使用的两个接口是否可导入：

```bash
python -c "from kaiwu.torch_plugin import RestrictedBoltzmannMachine; from kaiwu.classical import SimulatedAnnealingOptimizer; print('Kaiwu imports OK')"
```

脚本按torch.cuda.is_available()选择CUDA或CPU。导入检查通过表示接口可用，并不等同于已完成模型训练验证。

## 3. 数据准备

准备ULB Credit Card Fraud Detection数据集中的creditcard.csv，放在fin-fraud.py同级目录。对应实验数据包含284,807行交易记录，其中492行为欺诈交易。

必需字段：Time、V1至V28、Amount、Class。Class=0表示正常交易，Class=1表示欺诈交易。

程序在fin-fraud.py顶部读取：

```python
DATA_PATH = r"creditcard.csv"
```

若数据保存在其他位置，将其修改为实际路径，例如：

```python
DATA_PATH = r"D:\datasets\creditcard.csv"
```

不需要手工拆分数据。脚本按60%/20%/20%进行训练、验证、测试分层划分；标准化与归一化参数只在训练集拟合。RBM预训练使用全部训练集欺诈样本和20,000条抽样正常交易，CNN使用完整训练集。

## 4. 运行实验

在项目根目录打开终端，然后运行：

```bash
python fin-fraud.py
```

也可保存控制台日志：

```bash
python -u fin-fraud.py > run.log 2>&1
```

顺序为：读取与划分数据、数据变换、RBM预训练、特征提取、传统分类模型、普通CNN、QRBM+CNN、Focal模型、完整模型结果汇总。

参数通过编辑脚本顶部变量修改，不通过命令行传入。默认输出目录为相对于当前工作目录的fraud_results_ulb_fixed；同名结果会被覆盖。如需保留仓库已有结果，可先修改：

```python
RESULT_DIR = "./fraud_results_run_new"
```

运行主脚本会重新训练模型；程序不会自动加载已有.pth文件跳过训练。

## 5. 主要参数

| 变量 | 默认值 | 含义 |
|---|---:|---|
| SEED | 42 | 随机种子 |
| RBM_HIDDEN_DIM | 24 | RBM隐藏维度 |
| RBM_EPOCHS | 15 | RBM训练轮数 |
| RBM_LR | 0.03 | RBM学习率 |
| RBM_BATCH_SIZE | 256 | RBM批量大小 |
| RBM_NORMAL_SAMPLE_SIZE | 20000 | RBM正常样本抽样数量 |
| BATCH_SIZE | 1024 | CNN批量大小 |
| CNN_EPOCHS | 60 | CNN最大训练轮数 |
| CNN_LR | 0.001 | CNN初始学习率 |
| WEIGHT_DECAY | 0.001 | CNN权重衰减 |
| EARLY_STOP_PATIENCE | 15 | 早停等待轮数 |
| BASELINE_POS_WEIGHT | 5 | 加权交叉熵正类权重 |
| FOCAL_ALPHA | 0.75 | Focal类别权重参数 |
| FOCAL_GAMMA | 2 | Focal聚焦参数 |

## 6. 输出文件

以下路径均相对于RESULT_DIR。

| 输出 | 内容 |
|---|---|
| model_comparison.csv | 各模型测试指标、阈值及tn/fp/fn/tp |
| experiment_config.json | 数据规模与训练配置 |
| qrbm_model.pth | RBM参数、层维度和训练历史 |
| qrbm_history.csv | RBM逐轮训练目标记录 |
| Plain_CNN.pth | 普通CNN保存的参数 |
| QRBM_plus_CNN.pth | 特征融合CNN保存的参数 |
| QRBM_plus_CNN_plus_Focal.pth | Focal模型保存的参数，完整模型复用此模型 |
| Plain_CNN_history.csv | 普通CNN训练损失及验证指标 |
| QRBM_plus_CNN_history.csv | 特征融合CNN训练日志 |
| QRBM_plus_CNN_plus_Focal_history.csv | Focal模型训练日志 |
| Plain_CNN_threshold.csv | 普通CNN验证集候选阈值及指标 |
| QRBM_plus_CNN_threshold.csv | 特征融合CNN验证集候选阈值及指标 |

完整模型在内存中计算full_threshold_df，但原脚本未将其保存为CSV。原脚本也未保存逐样本测试预测分数、标准化器或传统分类器模型文件。

## 7. 结果可视化

将配套的visualize_results.py放入项目根目录后运行：

```bash
python visualize_results.py
```

自定义结果目录时使用：

```bash
python visualize_results.py --result-dir fraud_results_run_new --output-dir figures
```

绘图仅读取CSV，不需要Kaiwu、模型权重或原始交易数据。默认输出到fraud_results_ulb_fixed/figures，包含PNG和SVG格式的五类图：模型测试指标对比、完整模型混淆矩阵、验证集AP变化、CNN训练损失、RBM预训练目标变化。检测到*_threshold.csv时额外生成验证集阈值敏感性图。

## 8. 指标与已有结果

precision为精确率，recall为召回率，f1为两者的调和平均，roc_auc为ROC曲线下面积。pr_auc在本代码中使用average_precision_score计算，图表统一称为平均精确率AP。

现有完整模型结果：精确率0.888889、召回率0.808081、F1为0.846561、ROC-AUC为0.977176、AP为0.766853；阈值约0.406872；TN=56853、FP=10、FN=19、TP=80。

结果来自已保存实验。不同运行环境、依赖或随机采样可能影响重跑结果。

## 9. 结果解释注意事项

- 当前采样器为SimulatedAnnealingOptimizer。
- 主脚本中Focal行与完整模型行使用相同模型，最终测试也都搜索了验证阈值，因此结果相同；不能将两行用于固定阈值与调优阈值的独立消融比较。
- ZeroR与Random Guess的常数概率经当前阈值规则处理后均将所有样本判为欺诈，配套主对比图不使用这两行。
- 测试集ROC/PR曲线需要逐样本真实标签与预测分数，不能从AP或ROC-AUC汇总数值反推。

## 10. 常见问题

1. 找不到creditcard.csv：检查当前工作目录与DATA_PATH。推荐使用绝对数据路径。
2. 缺少Class或V列：核对CSV字段，不要使用只有汇总统计的表格代替原始交易数据。
3. 找不到kaiwu.torch_plugin：检查当前Python解释器，以及SDK和插件是否安装于同一环境；参照官方说明确认接口版本。
4. CUDA相关报错：可先将device配置改为torch.device("cpu")进行定位。
5. 重跑覆盖结果：先更改RESULT_DIR，再启动主脚本。
6. 只需查看已有结果：直接打开model_comparison.csv，或运行visualize_results.py。

## 11. 提交打包

源代码包可包含fin-fraud.py、README.md、配套visualize_results.py和fraud_results_ulb_fixed目录。数据按单独准备的方式说明；如提交时一并提供creditcard.csv，应放在主脚本同级目录，并同步更新材料清单。Kaiwu作为环境依赖配置，不应在材料清单中写成已随包提供的源码目录。
