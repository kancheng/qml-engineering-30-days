# Day21 實驗結果

本檔由 `articles/day21/plot_results.py` 依保存的 JSON 產生。

資料：Day20 去重 binary Iris 99 筆，split seeds 2028／2029，train／validation／test=59／19／21。
16 次 fit 全部是 NumPy CPU exact reference；每次 73 個 Brier objectives、4,307 筆 train 輸入評估。CUDA-Q 僅驗證凍結模型。

## 選定模型

| Split | Model | Seed | Test Brier | Accuracy | Confusion matrix |
|---|---|---:|---:|---:|---|
| 2028 | pca_vqc | 43 | 0.105396 | 85.71% | [[8, 2], [1, 10]] |
| 2028 | selection_vqc | 42 | 0.053619 | 90.48% | [[9, 1], [1, 10]] |
| 2028 | bottleneck_vqc | 43 | 0.038293 | 95.24% | [[9, 1], [0, 11]] |
| 2028 | logistic4 | 43 | 0.037677 | 95.24% | [[9, 1], [0, 11]] |
| 2029 | pca_vqc | 43 | 0.102869 | 85.71% | [[9, 1], [2, 9]] |
| 2029 | selection_vqc | 42 | 0.059739 | 90.48% | [[10, 0], [2, 9]] |
| 2029 | bottleneck_vqc | 43 | 0.018951 | 100.00% | [[10, 0], [0, 11]] |
| 2029 | logistic4 | 43 | 0.031191 | 95.24% | [[9, 1], [0, 11]] |

Train-prior baseline p1=29/59：test Brier=0.250475，accuracy=47.62%（兩組 test 類別數相同）。Confusion matrix row=true、column=predicted。

![Comparison](comparison.png)

## 表示與範圍

| Split | 選取欄位（zero-based，依 score 排序） | PCA2 variance ratio 合計 |
|---|---|---:|
| 2028 | [3, 2] | 0.891425 |
| 2029 | [3, 2] | 0.884689 |

欄位 0／1／2／3 為 sepal length／sepal width／petal length／petal width；完整 train correlations 與 scaler 見 datasets.json。

| Split | Model | Train / validation / test clipped values | Bottleneck test abs(h)>0.99 |
|---|---|---|---:|
| 2028 | pca_vqc | [0, 0, 1] | — |
| 2028 | selection_vqc | [0, 1, 1] | — |
| 2028 | bottleneck_vqc | [0, 0, 0] | 10 |
| 2028 | logistic4 | [0, 0, 0] | — |
| 2029 | pca_vqc | [0, 0, 0] | — |
| 2029 | selection_vqc | [0, 0, 0] | — |
| 2029 | bottleneck_vqc | [0, 0, 0] | 9 |
| 2029 | logistic4 | [0, 0, 0] | — |

Clipping 按座標值計數；bottleneck 使用 tanh，logistic4 不做 clipping。PCA variance 是標準化 train variance，不是分類資訊保留率。

![Representations](representations.png)

## 所有訓練候選

| Split | Model | Seed | Train Brier | Validation Brier | NumPy fit seconds | Full sweeps |
|---|---|---:|---:|---:|---:|---:|
| 2028 | pca_vqc | 42 | 0.105924 | 0.099047 | 0.344 | 9 |
| 2028 | pca_vqc | 43 | 0.101977 | 0.087006 | 0.340 | 9 |
| 2028 | selection_vqc | 42 | 0.054485 | 0.051643 | 0.342 | 9 |
| 2028 | selection_vqc | 43 | 0.055565 | 0.055266 | 0.346 | 9 |
| 2028 | bottleneck_vqc | 42 | 0.015715 | 0.063645 | 0.344 | 2 |
| 2028 | bottleneck_vqc | 43 | 0.016970 | 0.034839 | 0.340 | 2 |
| 2028 | logistic4 | 42 | 0.018055 | 0.048099 | 0.006 | 7 |
| 2028 | logistic4 | 43 | 0.020294 | 0.040918 | 0.006 | 7 |
| 2029 | pca_vqc | 42 | 0.099425 | 0.133513 | 0.335 | 9 |
| 2029 | pca_vqc | 43 | 0.094092 | 0.125680 | 0.336 | 9 |
| 2029 | selection_vqc | 42 | 0.052040 | 0.063476 | 0.334 | 9 |
| 2029 | selection_vqc | 43 | 0.053150 | 0.068977 | 0.329 | 9 |
| 2029 | bottleneck_vqc | 42 | 0.009465 | 0.091613 | 0.330 | 2 |
| 2029 | bottleneck_vqc | 43 | 0.012345 | 0.058478 | 0.331 | 2 |
| 2029 | logistic4 | 42 | 0.010679 | 0.071606 | 0.006 | 7 |
| 2029 | logistic4 | 43 | 0.013616 | 0.066215 | 0.006 | 7 |

## CUDA-Q 驗證

| Backend | Records | Observe calls | Max probability error | Passed |
|---|---:|---:|---:|---|
| qpp-cpu | 24 | 594 | 3.331e-16 | True |
| nvidia | 24 | 594 | 1.340e-07 | True |

每 backend：2 splits × 99 筆 × 3 量子模型=594 observe；classical baseline 仍用 NumPy CPU。shots=-1、無噪聲，時間包含可能的 compilation／cache，不是隔離效能測量。

環境版本、precision、UTC timestamp 與 artifact SHA-256 見各 summary。

## 範圍與解讀

這是固定小預算、兩個重疊 splits 的示範。PCA／selection 固定表示，bottleneck 同時訓練 10 個 encoder 與 4 個量子參數；objective 次數相同不等於相同 sweeps 或模型容量。
完整四維 logistic-link baseline 使用 Brier＋座標搜尋，並非標準 cross-entropy LogisticRegression solver。結果不證明量子優勢，也不能單獨把準確率差異歸因於降維方法。
沿用 Day20 已公開的 test，屬探索性系列實驗；正式模型選擇需新的 holdout 或 nested cross-validation。

## 原始紀錄

- [Protocol](protocol.json)、[資料與 preprocessing](datasets.json)、[全部 candidates](candidates.json)
- [Selected models](selected.json)、[Training summary](training_summary.json)
- [CPU summary](qpp-cpu/summary.json)、[CPU predictions](qpp-cpu/verification.json)
- [GPU summary](nvidia/summary.json)、[GPU predictions](nvidia/verification.json)
- [教學與重跑指令](../../articles/day21/README.md)
