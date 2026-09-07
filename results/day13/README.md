# Day 13 實驗紀錄

[教學與重跑指令](../../articles/day13/README.md)。本日比較 signed-real amplitude encoding 的 simulator state loading 與明確 RY／CNOT 準備。

| Backend | 案例 | 最大 fidelity 與 1 的差異（兩路徑） | 最大 X expectation 誤差 | 測試 |
|---|---:|---:|---:|---:|
| qpp-cpu（fp64） | 12 | 4.44e-16 | 3.33e-16 | 6 通過 |
| nvidia（fp32） | 12 | 1.19e-7 | 1.57e-7 | 6 通過 |

每組案例的兩條路徑均與 normalized target state 核對；明確電路另有 exact X 與 1,000 shots 的 Z 基底 counts。所有數值通過 1e-5 容許誤差，shots 總數正確。

## 案例與成本

涵蓋 2、3、4 維、尺度倍數、global sign、relative sign、basis01／basis10、零分支、Bell state 與 signed four-vector。測試額外驗證四個 basis states、固定 seed 隨機向量及極大／極小有限值 normalization。

本例一個 qubit 使用 1 RY，兩個 qubit 使用 3 RY＋2 CNOT。這是 source-level 計數，不含量測、硬體路由或編譯後消除；direct loading 路徑沒有提供硬體 gate 成本。

[3,4,0] 和 [30,40,0] 產生相同 state；[-3,-4,0] 只差 global phase。[3,-4,0] 的 Z 機率不變，但 X1 從約 +0.96 變成 −0.96，顯示只看 counts 不能檢查相對符號。

## 原始資料

- CPU：[summary](qpp-cpu/summary.json)、[predictions](qpp-cpu/predictions.json)、[signed four-vector 電路](qpp-cpu/circuit.txt)。
- GPU：[summary](nvidia/summary.json)、[predictions](nvidia/predictions.json)、[signed four-vector 電路](nvidia/circuit.txt)。

predictions 保存原始 values、維度、norm 因子、normalized amplitudes、angles、兩路徑 fidelity／probabilities、X reference、counts、seed 與 logical gate counts。時間戳使用 UTC；2026-09-06 16:xx UTC 對應台灣 2026-09-07 00:xx。

## 環境與限制

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6、OMP_NUM_THREADS=1；RTX 3060 Laptop 6 GiB，driver 570.211.01。沿用 `.venv`，沒有新增依賴。

GPU 實驗結束時出現 cudaErrorCudartUnloading，退出碼為 0，數值核對通過；根因尚未確認。結果僅涵蓋無噪聲 simulator、2–4 個實數 features，不包含任意複數、高維 synthesis、QPU 或速度／分類效能比較。
