# AI-lab01

当代人工智能2026实验1：新闻文本10分类。此仓库提供实验代码、配置、教师原始 CSV、真实实验结果及最终预测文件，**不包含实验报告、本地 Python 环境和训练模型文件**。

## 运行环境与教师数据

原实验环境为 Python 3.12.14；主要依赖版本固定在 `requirements.txt`。建议使用 Python 3.12。仓库已包含教师 `data-Project1` 中的两个原始文件：

```text
data/raw/train_data.csv          # 列：text,target；标签 0–9
data/raw/test_data_unlabeled.csv # 列：text；保持原始行顺序
```

两个 CSV 与教师原件逐字节一致，无需另行复制；不得替换为其他数据后将结果视为本实验结果。原实验有标签数据 7368 条，外部无标签测试数据 2457 条。无需提供教师预测示例即可运行；预测格式已固定为无表头、无索引、单列整数标签。

## 一键复现

在仓库根目录打开 PowerShell，执行：

```powershell
.\setup_environment.ps1 -PythonExe "C:\实际路径\python.exe"
.\run_all.ps1
```

已有同版本依赖环境时无需重新安装：

```powershell
.\run_all.ps1 -PythonExe "C:\已有环境\Scripts\python.exe"
```

要完整运行两次并核对确定性划分、全部验证指标、内部测试指标与预测文件哈希：

```powershell
.\run_all.ps1 -CheckReproducibility
```

运行会覆盖本地结果与预测文件；运行时间随机器变化，不作为确定性比较项。脚本发现数据缺失或任何子步骤失败会立即报错。运行生成的划分、模型及额外审计材料保存在本仓库目录内，但不随本次提交上传。

## 实验设计

- 分层划分 train/validation/internal test = 70%/15%/15%，实际为 5156/1106/1106 条；随机种子为 42 和 43。
- 三组 TF-IDF × 三种模型 × 每种模型三组参数，共 27 组验证实验；全部配置见 `configs/experiment_config.json`。
- Multinomial Naive Bayes 搜索 alpha = 0.1、0.5、1；Logistic Regression 与 Linear SVM 搜索 C = 0.1、1、10。
- 验证阶段 TF-IDF 仅在训练集 fit；只依据验证指标选择方案。锁定后在 train+validation 重训并评估内部测试集，最后在全部有标签数据上重训并预测外部测试集。
- 两组创新对照：教师 unigram/5000 维基线与优化 TF-IDF 对照；优化方案与仅增加文本清洗的方案对照。自定义清洗器保留一致的小写化和去重音处理，避免混杂变量。

已保存结果选择 `tuned_word_tfidf + LinearSVC(C=1)`；Validation Macro-F1 = 0.952037，Internal Test Macro-F1 = 0.932786。准确结果和逐组参数以 CSV/JSON 为准，复现比较不含运行时间。

## 代码与提交材料

| 路径 | 用途 |
| --- | --- |
| `data/raw/` | 教师原始训练与无标签测试 CSV；`data/splits/` 为运行时生成 |
| `src/lab1_common.py` | 文本清洗、TF-IDF、分类器、指标与输入检查 |
| `src/run_experiments.py` | 固定划分、27 组实验、模型选择、评估、绘图与最终预测 |
| `src/verify_project.py` | 数据划分、选择依据、格式、模型预测与输入哈希检查 |
| `src/reproducibility_check.py` | 两次完整运行的确定性结果比较 |
| `tests/test_project.py` | 清洗、归一化、配置、划分及预测格式测试 |
| `results/experiment_results.csv` | 全部验证实验的参数、指标与运行时间 |
| `results/model_selection.json` | 仅使用验证集的最终模型选择依据 |
| `results/innovation_ablation.csv` | 同模型、同参数的配对特征消融结果 |
| `results/classification_report.csv`、`results/confusion_matrix.csv` | 内部测试集分类指标及混淆矩阵 |
| `results/reproducibility_report.json`、`results/verification.log` | 已完成的复现比较与完整性检查记录 |
| `results/plots/` | 超参数验证曲线与内部测试混淆矩阵图片 |
| `submission/predictions.csv` | 2457 行最终预测，顺序与教师无标签测试集一致 |

提交预测文件 SHA-256：`592e58ca38303f7441492581752b06aaafbfbeaf2fe59fd948ff0c6b7a807067`。
