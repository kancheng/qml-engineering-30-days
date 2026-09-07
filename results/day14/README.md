# Day 14 實驗紀錄

[教學與執行指令](../../articles/day14/README.md)。兩種編碼共用 Day 9 Ansatz 與 ZZ 讀出，各 backend 執行 24 組 forward、2 組短訓練與 6 個 unittest，全部通過。

| Backend | Encoding | 初始 MSE | 四輪後 MSE |
|---|---|---:|---:|
| qpp-cpu | angle | 0.377338315 | 0.0153427612 |
| qpp-cpu | amplitude | 0.35795077 | 0.0169261092 |
| nvidia | angle | 0.377338298 | 0.0153427534 |
| nvidia | amplitude | 0.357950779 | 0.0169261078 |

每個短訓練使用三筆合成資料，labels 由該模型的 NumPy teacher 產生。四輪皆因 max_sweeps 停止，各 33 次 objective／99 次訓練 observe。沒有 holdout 或分類 accuracy；兩種 encoding 的資料不同，不能以 MSE 排名。

## 正確性與原始紀錄

CPU 最大 exact ZZ 誤差約 7.22e-16，最大 fidelity 與 1 的差異約 1.11e-15。GPU 分別約 2.37e-7、1.31e-7，均低於 1e-5 門檻。每組 forward 有 1,000 shots parity 讀出與完整 counts，未用於 optimizer。

- CPU：[summary](qpp-cpu/summary.json)、[predictions](qpp-cpu/predictions.json)、[training history](qpp-cpu/training.json)、[angle 電路](qpp-cpu/angle_circuit.txt)、[amplitude 電路](qpp-cpu/amplitude_circuit.txt)。
- GPU：[summary](nvidia/summary.json)、[predictions](nvidia/predictions.json)、[training history](nvidia/training.json)、[angle 電路](nvidia/angle_circuit.txt)、[amplitude 電路](nvidia/amplitude_circuit.txt)。

config、features、初始／最終 weights、teacher labels 與每輪 loss 均保存。forward 同時核對 NumPy state fidelity 和 expectation。

## 環境與限制

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6、OMP_NUM_THREADS=1；RTX 3060 Laptop 6 GiB，driver 570.211.01。CPU fp64、GPU fp32，沿用 .venv，無新增依賴。時間戳使用 UTC，2026-09-06 18:xx 對應台灣 2026-09-07 02:xx。

GPU 測試與實驗退出時出現 cudaErrorCudartUnloading，退出碼為 0，數值核對通過；根因尚未確認。全部為無噪聲 simulator 結果，沒有 QPU、速度、模型泛化或量子優勢結論。
