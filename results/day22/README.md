# Day22 實驗結果

由 `articles/day22/plot_results.py` 依 JSON 產生。

Day20 binary Iris 99 筆，兩個 splits，各 train／validation／test=59／19／21。20 次 NumPy CPU reference fits，每次 73 個 objective、4,307 筆 train 輸入評估；CUDA-Q training calls=0。

## 電路資源

| Model | Parameters | Upload blocks | RY | CNOT | Logical depth |
|---|---:|---:|---:|---:|---:|
| pca_once2 | 8 | 1 | 10 | 2 | 7 |
| pca_repeat2 | 8 | 2 | 12 | 2 | 8 |
| chunks_once4 | 16 | 2 | 20 | 4 | 14 |
| chunks_repeat4 | 16 | 4 | 24 | 4 | 16 |
| logistic4 | 5 | 0 | 0 | 0 | 0 |

所有量子模型為兩 qubit。Depth 是未合併 gates、平行單 qubit 層計算的邏輯深度，不含 readout；不是編譯器或 QPU 實測深度。

## Validation 選定模型

| Split | Model | Seed | Test Brier | Accuracy | Confusion matrix | Clipped train/val/test |
|---|---|---:|---:|---:|---|---|
| 2028 | pca_once2 | 43 | 0.106940 | 85.71% | [[8, 2], [1, 10]] | [0, 0, 1] |
| 2028 | pca_repeat2 | 43 | 0.095350 | 85.71% | [[9, 1], [2, 9]] | [0, 0, 1] |
| 2028 | chunks_once4 | 43 | 0.067473 | 95.24% | [[10, 0], [1, 10]] | [0, 1, 1] |
| 2028 | chunks_repeat4 | 42 | 0.074330 | 85.71% | [[7, 3], [0, 11]] | [0, 1, 1] |
| 2028 | logistic4 | 43 | 0.037677 | 95.24% | [[9, 1], [0, 11]] | [0, 0, 0] |
| 2029 | pca_once2 | 42 | 0.108847 | 85.71% | [[9, 1], [2, 9]] | [0, 0, 0] |
| 2029 | pca_repeat2 | 42 | 0.119821 | 80.95% | [[7, 3], [1, 10]] | [0, 0, 0] |
| 2029 | chunks_once4 | 42 | 0.090362 | 90.48% | [[10, 0], [2, 9]] | [0, 1, 0] |
| 2029 | chunks_repeat4 | 42 | 0.096516 | 85.71% | [[8, 2], [1, 10]] | [0, 1, 0] |
| 2029 | logistic4 | 43 | 0.031191 | 95.24% | [[9, 1], [0, 11]] | [0, 0, 0] |

Train-prior baseline p1=29/59：test Brier=0.250475、accuracy=47.62%，兩 split 相同。Confusion matrix row=true、column=predicted；clipping 按輸入座標值計數，不乘 upload 次數。

![Comparison](comparison.png)

## 全部候選與訓練成本

| Split | Model | Seed | Train Brier | Validation Brier | NumPy fit seconds | Full sweeps |
|---|---|---:|---:|---:|---:|---:|
| 2028 | pca_once2 | 42 | 0.109331 | 0.089984 | 0.437 | 4 |
| 2028 | pca_once2 | 43 | 0.108820 | 0.088482 | 0.426 | 4 |
| 2028 | pca_repeat2 | 42 | 0.134590 | 0.119919 | 0.492 | 4 |
| 2028 | pca_repeat2 | 43 | 0.108501 | 0.098928 | 0.490 | 4 |
| 2028 | chunks_once4 | 42 | 0.074296 | 0.081954 | 0.764 | 2 |
| 2028 | chunks_once4 | 43 | 0.063564 | 0.066266 | 0.760 | 2 |
| 2028 | chunks_repeat4 | 42 | 0.066722 | 0.087423 | 0.901 | 2 |
| 2028 | chunks_repeat4 | 43 | 0.081910 | 0.099639 | 0.902 | 2 |
| 2028 | logistic4 | 42 | 0.018055 | 0.048099 | 0.006 | 7 |
| 2028 | logistic4 | 43 | 0.020294 | 0.040918 | 0.006 | 7 |
| 2029 | pca_once2 | 42 | 0.094536 | 0.122438 | 0.413 | 4 |
| 2029 | pca_once2 | 43 | 0.098006 | 0.129218 | 0.421 | 4 |
| 2029 | pca_repeat2 | 42 | 0.131569 | 0.141051 | 0.496 | 4 |
| 2029 | pca_repeat2 | 43 | 0.118289 | 0.144426 | 0.495 | 4 |
| 2029 | chunks_once4 | 42 | 0.077309 | 0.087475 | 0.773 | 2 |
| 2029 | chunks_once4 | 43 | 0.050284 | 0.093357 | 0.767 | 2 |
| 2029 | chunks_repeat4 | 42 | 0.088285 | 0.098164 | 0.900 | 2 |
| 2029 | chunks_repeat4 | 43 | 0.094432 | 0.147369 | 0.898 | 2 |
| 2029 | logistic4 | 42 | 0.010679 | 0.071606 | 0.006 | 7 |
| 2029 | logistic4 | 43 | 0.013616 | 0.066215 | 0.006 | 7 |

## 凍結模型驗證

| Backend | Records | Observe calls | Max probability error | Passed |
|---|---:|---:|---:|---|
| qpp-cpu | 30 | 792 | 6.661e-16 | True |
| nvidia | 30 | 792 | 2.719e-07 | True |

每 backend：2 splits×99 筆×4 量子模型=792 次 exact observe。另核對 logistic4 NumPy CPU，合計 30 個 partition records。版本、precision、UTC timestamp 與 artifact SHA-256 見 summaries。
GPU 程序結束有 cudaErrorCudartUnloading 訊息；驗證及測試 exit code=0，根因未定位。時間可能含 compilation／cache，不用於 GPU speedup 結論。

## 相同權重的輸入反應

![Schedule probe](schedule_probe.png)

兩張圖使用相同 seed42 的八個未訓練參數。只改 upload schedule；這不是 test decision boundary，也不是 expressibility 的統計測量。

在 scaled inputs (±0.5, ±0.5) 的 mixed contrast：
- pca_once2: -0.95769392
- pca_repeat2: -0.24858023

非零 contrast 表示在這四點無法寫成 f(x0)+g(x1)。單次 upload 也可能有 interaction；re-uploading 改變輸入反應，不保證 interaction 或 accuracy 必然增加。

## 解讀與限制

配對內 trainable blocks、初始化與 objective 預算相同，額外 data gates 仍增加操作量；配對間的 PCA2／四維 chunks 表示及參數數量不同，不能直接隔離單一原因。
這次結果沒有顯示 re-uploading 必然優於 once controls。固定 73 次 objective 對 8／16 參數只有 4／2 次 full sweeps；未證明充分收斂。
兩個重疊的小 splits、Day20 已公開的 test、exact 無噪聲 simulator，只支持探索性示範。沒有論文完整復現、universal approximation 或量子優勢結論。

## 原始紀錄

- [Protocol](protocol.json)、[資料與 preprocessing](datasets.json)、[全部 candidates](candidates.json)
- [選定模型](selected.json)、[Training summary](training_summary.json)、[固定權重 probe](schedule_probe.json)
- [CPU summary](qpp-cpu/summary.json)、[CPU predictions](qpp-cpu/verification.json)
- [GPU summary](nvidia/summary.json)、[GPU predictions](nvidia/verification.json)
- [教學及重跑指令](../../articles/day22/README.md)
