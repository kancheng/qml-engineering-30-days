# Day25 Wine Benchmark Report

由 `plot_results.py` 依保存的JSON產生。第二個真實資料集使用UCI Wine的cultivar2／3；119筆、13個原始特徵，非完整三分類，也非Wine Quality。

## Protocol與資料

- Dataset seeds2030／2031，各train70／validation23／test26；初始化seeds42／43。
- 16次NumPy CPU reference fits，每次73個Brier objective、5,110筆train輸入評估；CUDA-Q training calls=0。
- MLP／VQC／Hybrid同用train-only PCA2；logistic13使用全部標準化13維。
- 每模型每split只依final validation Brier選seed，test不參與當次選擇。

## 選定模型

| Split | Model | Seed | Test Brier | Accuracy | Confusion matrix | Clipped train/val/test |
|---|---|---:|---:|---:|---|---|
| 2030 | mlp | 43 | 0.028478 | 96.15% | [[14, 1], [0, 11]] | [0, 2, 1] |
| 2030 | vqc | 42 | 0.032404 | 96.15% | [[14, 1], [0, 11]] | [0, 2, 1] |
| 2030 | hybrid | 42 | 0.034820 | 96.15% | [[14, 1], [0, 11]] | [0, 2, 1] |
| 2030 | logistic13 | 43 | 0.040234 | 92.31% | [[13, 2], [0, 11]] | [0, 0, 0] |
| 2031 | mlp | 42 | 0.041693 | 92.31% | [[13, 2], [0, 11]] | [0, 0, 1] |
| 2031 | vqc | 42 | 0.057050 | 92.31% | [[13, 2], [0, 11]] | [0, 0, 1] |
| 2031 | hybrid | 42 | 0.041495 | 92.31% | [[13, 2], [0, 11]] | [0, 0, 1] |
| 2031 | logistic13 | 43 | 0.032224 | 92.31% | [[13, 2], [0, 11]] | [0, 0, 0] |

Train-prior baseline p1=28/70=0.4：test Brier=0.244615、accuracy=57.69%，兩split相同。Confusion row=true／column=predicted；clipping按座標計數。

![Comparison](comparison.png)

## PCA與資料完整性

| Split | PC1 variance | PC2 variance | Sum |
|---|---:|---:|---:|
| 2030 | 0.372251 | 0.138856 | 0.511107 |
| 2031 | 0.379600 | 0.159848 | 0.539448 |

原始檔SHA-256：`6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659`。沒有完全重複feature＋label rows。Variance ratio針對標準化train，不等於保留的分類資訊比例。

## 全部候選與成本

| Split | Model | Seed | Train Brier | Validation Brier | NumPy fit seconds | Full sweeps |
|---|---|---:|---:|---:|---:|---:|
| 2030 | mlp | 42 | 0.029996 | 0.033030 | 0.010 | 4 |
| 2030 | mlp | 43 | 0.028448 | 0.031751 | 0.010 | 4 |
| 2030 | vqc | 42 | 0.040014 | 0.037809 | 0.398 | 9 |
| 2030 | vqc | 43 | 0.061833 | 0.078268 | 0.399 | 9 |
| 2030 | hybrid | 42 | 0.030592 | 0.022207 | 0.311 | 3 |
| 2030 | hybrid | 43 | 0.028877 | 0.022667 | 0.310 | 3 |
| 2030 | logistic13 | 42 | 0.012122 | 0.037450 | 0.007 | 2 |
| 2030 | logistic13 | 43 | 0.006819 | 0.023418 | 0.008 | 2 |
| 2031 | mlp | 42 | 0.023493 | 0.026533 | 0.010 | 4 |
| 2031 | mlp | 43 | 0.023863 | 0.028144 | 0.010 | 4 |
| 2031 | vqc | 42 | 0.035976 | 0.040932 | 0.396 | 9 |
| 2031 | vqc | 43 | 0.047176 | 0.045563 | 0.396 | 9 |
| 2031 | hybrid | 42 | 0.027362 | 0.026825 | 0.308 | 3 |
| 2031 | hybrid | 43 | 0.030532 | 0.032533 | 0.310 | 3 |
| 2031 | logistic13 | 42 | 0.007995 | 0.022182 | 0.008 | 2 |
| 2031 | logistic13 | 43 | 0.004160 | 0.019166 | 0.008 | 2 |

## 凍結模型驗證

| Backend | Records | Observe calls | Max probability error | Passed |
|---|---:|---:|---:|---|
| qpp-cpu | 24 | 476 | 3.331e-16 | True |
| nvidia | 24 | 476 | 1.090e-07 | True |

每backend：2splits×119筆×2量子模型=476次exact observe；classical模型保持NumPy CPU，合計24筆partition records。
GPU退出有cudaErrorCudartUnloading，驗證與測試exit code=0；根因未定位。時間可能含compilation／cache，非GPU training或隔離speedup測量。

## 解讀與限制

MLP與VQC／Hybrid有相同二維輸入；logistic13可見更多資訊，所以是實用完整輸入對照，非隔離量子層效果的配對。
73次objective不等於相同參數更新次數或充分收斂；沒有early stopping或依test追加預算。
兩個test splits重疊，每組僅26筆；不給顯著性或量子優勢結論，也不把Iris與Wine不同任務的accuracy直接排行。
本日沒有重跑Day23 quantum kernel或Day24梯度分布；第二份benchmark聚焦原定Classical／VQC／Hybrid交付。

## 原始紀錄

- [Protocol](protocol.json)、[資料／preprocessor](datasets.json)、[Candidates](candidates.json)、[Selected](selected.json)
- [Training summary](training_summary.json)
- [CPU summary](qpp-cpu/summary.json)、[CPU predictions](qpp-cpu/verification.json)
- [GPU summary](nvidia/summary.json)、[GPU predictions](nvidia/verification.json)
- [資料來源與授權](../../data/day25/README.md)、[教學與重跑](../../articles/day25/README.md)
