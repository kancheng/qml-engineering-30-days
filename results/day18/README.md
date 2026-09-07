# Day 18 固定協定 Benchmark

[文章與程式](../../articles/day18/README.md)。同一 XOR split（dataset seed=2027），三模型每個 seed 各 109 次 train objective；seeds 42／43 僅由 final validation Brier 選擇。每 backend 六次 fit、四個測試，全部通過。

| Backend | Model | 選定 seed | Test accuracy | Test Brier |
|---|---|---:|---:|---:|
| qpp-cpu | logistic | 42 | 50.00% | 0.24586002 |
| qpp-cpu | mlp | 42 | 58.33% | 0.13770608 |
| qpp-cpu | vqc | 43 | 100.00% | 0.03393890 |
| nvidia | logistic | 42 | 50.00% | 0.24586002 |
| nvidia | mlp | 42 | 58.33% | 0.13770608 |
| nvidia | vqc | 43 | 100.00% | 0.03393890 |

constant baseline：accuracy=50%、Brier=0.25。logistic 使用共同 Brier 訓練，是 logistic-link baseline，不是標準 log-loss Logistic Regression。12 筆 test 的 accuracy 每錯一筆約差 8.33 個百分點；本日不進行模型家族排名推論。

## 所有候選與成本

| Backend | Model | Seed | Train Brier | Validation Brier | Objective calls | Fit seconds |
|---|---|---:|---:|---:|---:|---:|
| qpp-cpu | logistic | 42 | 0.24899659 | 0.24842703 | 109 | 0.007 |
| qpp-cpu | logistic | 43 | 0.24899661 | 0.24844725 | 109 | 0.007 |
| qpp-cpu | mlp | 42 | 0.13664496 | 0.13114495 | 109 | 0.008 |
| qpp-cpu | mlp | 43 | 0.15153498 | 0.14672271 | 109 | 0.008 |
| qpp-cpu | vqc | 42 | 0.03176780 | 0.02166892 | 109 | 70.628 |
| qpp-cpu | vqc | 43 | 0.03173719 | 0.02158283 | 109 | 72.140 |
| nvidia | logistic | 42 | 0.24899659 | 0.24842703 | 109 | 0.008 |
| nvidia | logistic | 43 | 0.24899661 | 0.24844725 | 109 | 0.008 |
| nvidia | mlp | 42 | 0.13664496 | 0.13114495 | 109 | 0.008 |
| nvidia | mlp | 43 | 0.15153498 | 0.14672271 | 109 | 0.009 |
| nvidia | vqc | 42 | 0.03176780 | 0.02166893 | 109 | 71.539 |
| nvidia | vqc | 43 | 0.03173718 | 0.02158282 | 109 | 71.523 |

每個 VQC fit 1,308 次訓練 observe；兩個 seeds 共 2,616 次／backend。所有模型預算耗盡停止，沒有收斂保證。時間含 fit 內首次編譯，CPU／GPU 執行可能重疊，非隔離速度 benchmark；classical 模型在兩次 run 都使用 NumPy CPU。

## 原始資料

- qpp-cpu：[協定](qpp-cpu/protocol.json)、[資料](qpp-cpu/dataset.json)、[scaler](qpp-cpu/scaler.json)、[所有候選](qpp-cpu/candidates.json)、[選定模型與評分](qpp-cpu/selected.json)、[摘要](qpp-cpu/summary.json)、[圖表](qpp-cpu/benchmark.png)。
- nvidia：[協定](nvidia/protocol.json)、[資料](nvidia/dataset.json)、[scaler](nvidia/scaler.json)、[所有候選](nvidia/candidates.json)、[選定模型與評分](nvidia/selected.json)、[摘要](nvidia/summary.json)、[圖表](nvidia/benchmark.png)。

protocol 保存資料雜湊、budget、scaler 政策與選模規則；selected 保存 weights 與每個 split 的 probabilities／metrics。圖表橫軸為 objective evaluations，不是迭代輪數或秒。

## 限制與環境

XOR 特意需要非線性；參數量、每座標更新機會與 optimizer 適配度不同。共同 protocol 是可核查的資源控制，不代表各模型都達到最佳能力。只有一個 dataset split，不把 initialization seeds 或 CPU／GPU 重跑當獨立統計樣本。沒有量子優勢、QPU 或 finite-shot 結論。

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6、Matplotlib 3.10.6；OMP_NUM_THREADS=1。RTX 3060 Laptop 6 GiB，driver 570.211.01，CPU fp64／GPU fp32。無新增依賴。UTC 2026-09-06 22:xx 為台灣 2026-09-07 06:xx。

GPU 測試與實驗退出時出現 cudaErrorCudartUnloading，退出碼 0、數值核對通過，根因尚未確認。
