# Day 12 實驗紀錄

[教學與重跑方式](../../articles/day12/README.md)。沿用 Day 11 的 train-only scaler 與八筆 2D 資料，比較三種角度映射及 RY／RX／RZ／H→RZ。

| Backend | 設定 | Pair 比較 | 最大 observable 誤差 | 最大 fidelity 與 1 的差異 | 測試 |
|---|---:|---:|---:|---:|---:|
| qpp-cpu（fp64） | 96 | 6 | 5.55e-16 | 8.88e-16 | 7 通過 |
| nvidia（fp32） | 96 | 6 | 2.38e-7 | 1.51e-7 | 7 通過 |

所有設定通過 1e-5 數值容許誤差。每個 backend 另保留八組 positive_half／RY 的 sampling，各 1,000 shots。此抽樣不參與訓練或編碼策略選擇。

## 主要觀察

RY 端點 pair `[-1,0]` 與 `[1,0]` 的 fidelity：centered_full 約為 1，兩種 half 映射約為 0。正負 pair `[-0.25,0]` 與 `[0.25,0]` 則分別約為 0.5、0.853553、0.853553，與解析公式一致。

兩種 half 映射的 pair fidelity 相同，但 Z 讀出不同。直接 RZ 在 |0⟩ 上只產生 global phase；H→RZ 才能產生相對相位，且需 X／Y 讀出才看得到本例的角度變化。這些是固定電路的數值與解析核對，不是 encoding accuracy 排名。

## 原始檔案

- CPU：[summary](qpp-cpu/summary.json)、[逐筆設定](qpp-cpu/predictions.json)、[pair 比較](qpp-cpu/pairs.json)、[CSV](qpp-cpu/angle_sweep.csv)、[RY 電路](qpp-cpu/circuit.txt)。
- GPU：[summary](nvidia/summary.json)、[逐筆設定](nvidia/predictions.json)、[pair 比較](nvidia/pairs.json)、[CSV](nvidia/angle_sweep.csv)、[RY 電路](nvidia/circuit.txt)。

predictions 包含 raw、scaled、clipping flags、mapping、axis、radians、完整 probabilities、q0 Bloch expectations 與 NumPy reference。pair 檔案保存兩端 features 和 RY fidelity；state 順序明確為 |q0 q1⟩，以 labels 取得 amplitudes。

## 環境與範圍

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6、OMP_NUM_THREADS=1。GPU 為 RTX 3060 Laptop 6 GiB，driver 570.211.01；沿用 `.venv`，沒有新增依賴。

CPU／GPU 程序均以退出碼 0 完成，本次工具輸出未出現先前的 cudaErrorCudartUnloading 訊息。全部為無噪聲模擬器結果，未進行 QPU 實測、訓練、分類評分或速度比較。
