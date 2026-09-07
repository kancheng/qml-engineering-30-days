# Day 20 Iris Binary Report

[文章與重跑指令](../../articles/day20/README.md)。資料是 UCI Iris 的 versicolor／virginica，去重後 99 筆，兩組 stratified splits 各 train59／validation19／test21。四原始特徵共用 train-only standardization、PCA2 與 PC scaling。

## 選定模型的 Test 結果

| Split | Model | 選定 seed | Accuracy | Brier | Confusion matrix |
|---|---|---:|---:|---:|---|
| 2028 | mlp | 42 | 90.48% | 0.07088430 | [[8, 2], [0, 11]] |
| 2028 | vqc | 43 | 85.71% | 0.10539648 | [[8, 2], [1, 10]] |
| 2028 | hybrid | 42 | 90.48% | 0.07686679 | [[8, 2], [0, 11]] |
| 2029 | mlp | 42 | 85.71% | 0.08854314 | [[8, 2], [1, 10]] |
| 2029 | vqc | 43 | 85.71% | 0.10286942 | [[9, 1], [2, 9]] |
| 2029 | hybrid | 42 | 90.48% | 0.08169103 | [[9, 1], [1, 10]] |

Confusion matrix row=true、column=predicted。Constant baseline p1=29/59，test accuracy=47.62%、Brier=0.25047537。沒有以 test 挑 model family 或調整預算。

## 穩定性與預處理

- Split 2028：train PCA2 解釋變異比例 0.7736＋0.1179＝0.8914。
- Split 2029：train PCA2 解釋變異比例 0.7749＋0.1098＝0.8847。

兩個 split 的 test 資料可能重疊，兩個 initialization seeds 也不是新的 dataset 樣本。結果只提供有限的穩定性觀察，沒有信賴區間、顯著性或整個 Iris 三分類排名。各 split clipping 數量見 selected.json。

## 全部候選與訓練成本

| Split | Model | Seed | Train Brier | Validation Brier | NumPy fit seconds |
|---|---|---:|---:|---:|---:|
| 2028 | mlp | 42 | 0.08587578 | 0.06198313 | 0.006 |
| 2028 | mlp | 43 | 0.08698257 | 0.06305570 | 0.006 |
| 2028 | vqc | 42 | 0.10592397 | 0.09904705 | 0.354 |
| 2028 | vqc | 43 | 0.10197727 | 0.08700578 | 0.366 |
| 2028 | hybrid | 42 | 0.09157060 | 0.06886511 | 0.256 |
| 2028 | hybrid | 43 | 0.09183091 | 0.07778847 | 0.261 |
| 2029 | mlp | 42 | 0.06489028 | 0.12241147 | 0.005 |
| 2029 | mlp | 43 | 0.06664357 | 0.12563977 | 0.006 |
| 2029 | vqc | 42 | 0.09942525 | 0.13351256 | 0.332 |
| 2029 | vqc | 43 | 0.09409202 | 0.12568005 | 0.340 |
| 2029 | hybrid | 42 | 0.07596432 | 0.11414531 | 0.262 |
| 2029 | hybrid | 43 | 0.07264903 | 0.11874862 | 0.269 |

12 次 fit 各 73 objective evaluations、4,307 筆 train example evaluations，均耗盡固定預算停止，不宣稱收斂。所有訓練皆為 NumPy CPU，含 VQC／hybrid 的 exact statevector reference；CUDA-Q training calls=0。

## CUDA-Q 凍結模型驗證

| Backend | 紀錄 | Observe calls | 最大 probability 誤差 | 合計驗證秒數（含編譯） |
|---|---:|---:|---:|---:|
| qpp-cpu | 18 | 396 | 3.33e-16 | 17.363 |
| nvidia | 18 | 396 | 1.34e-07 | 17.411 |

CPU／GPU 各四個測試通過、18 筆 split/model 驗證通過 1e-5。這是保存模型的 inference 核對，不是 CUDA-Q 訓練；時間包含 JIT、cache 與 host 開銷，執行可能重疊，不能與 NumPy fit seconds 相除宣稱 speedup。

## 檔案

- [協定與資料雜湊](protocol.json)、[原始 split／preprocessing](datasets.json)、[12 個 candidates](candidates.json)、[選定模型與預測](selected.json)、[訓練摘要](training_summary.json)、[圖表](comparison.png)。
- CPU：[摘要](qpp-cpu/summary.json)、[逐 split 驗證](qpp-cpu/verification.json)。
- GPU：[摘要](nvidia/summary.json)、[逐 split 驗證](nvidia/verification.json)。

## 環境與限制

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6、Matplotlib 3.10.6，OMP_NUM_THREADS=1；RTX 3060 Laptop 6 GiB、driver 570.211.01。CPU fp64／GPU fp32，無新增依賴。時間戳為 UTC。

GPU 測試與驗證程序結束時有 cudaErrorCudartUnloading，退出碼為 0，數值核對通過，根因尚未確認。本日沒有三分類、finite shots、QPU 或量子優勢結論；所有模型都受二維 PCA 與小型固定預算限制。
