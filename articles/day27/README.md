# Day 27｜CPU vs GPU Quantum Simulation

前幾天的CPU／GPU主要用於結果核對。今天正式測量同一電路的host-visible latency，區分首次呼叫、暖機與精度。**這是exact statevector inference benchmark，不是QML訓練、QPU或Day26 noisy trajectory效能比較。**

程式：[benchmark.py](benchmark.py)、[run_all.py](run_all.py)、[plot_results.py](plot_results.py)、[結果核對](audit_results.py)。數值與圖表見[實測報告](../../results/day27/README.md)。沿用`.venv`，不新增套件。

## 1. 比較什麼才有意義？

| 比較項目 | 本日選擇 |
|---|---|
| 相同任務 | RY／RZ／CZ電路＋全域Z parity expectation |
| 相同輸入 | seed42產生相同weights，保存SHA-256 |
| Qubits | 4、8、12、16 |
| Circuit blocks | 4、12 |
| CPU | qpp-cpu、fp64、OMP_NUM_THREADS=1 |
| GPU主比較 | nvidia、option=fp64 |
| GPU補充比較 | nvidia預設fp32 |
| Noise／shots | 無noise、shots_count=-1 |
| 每組 | 首次1次、暖機2次、正式7次 |

官方文件說明`nvidia`的precision options。[D24] 程式以`nvidia-fp64`作結果目錄標籤，實際呼叫`cudaq.set_target('nvidia', option='fp64')`，並保存runtime回報precision。CPU fp64不能與GPU fp32混稱為完全相同數值條件。

## 2. 固定Circuit與Observable

重用Day24的kernel，每block按順序執行：每qubit RY→RZ，再CZ(0,1)、CZ(1,2)…。n個qubits、L個blocks共有2nL個rotation weights與(n−1)L個CZ。

Observable為Z0Z1…Z(n−1)，輸出一個expectation，不回傳完整statevector到Python。L是block數，不是compiled depth；backend可自行gate fusion，沒有刻意關掉某一邊的最佳化。固定seed不是訓練結果，較大qubit的數值不一定有分類意義。

## 3. 計時邊界

```python
start = time.perf_counter()
value = float(cudaq.observe(circuit, observable, n, blocks, weights,
                            shots_count=-1).expectation())
elapsed = time.perf_counter() - start
```

`observe`與expectation取得是同步結果路徑，計時結束時輸出已可用；不是只量非同步kernel enqueue。權重產生、observable建構在timer外。時間包含host呼叫、runtime與計算，不能稱為純GPU kernel time。

每case保留第一次呼叫，接著2次warmups，再保留7個獨立時間。報告median與Q25／Q75，不挑最短的一次當主要結果。IQR只描述這7次的分散，並非confidence interval。

每backend獨立process，但同backend的cases共享process。各case的first time不是保證cold JIT；只有首個case是該process第一次執行kernel，disk cache仍可能存在。程式沒有刪除使用者cache，也不以破壞cache來製造啟動時間。另保存process wall time，包含Python imports、target setup、檔案輸出及參考驗證。

## 4. 依序執行，避免互相干擾

`run_all.py`依CPU→GPU fp64→GPU fp32啟動子程序，上一個結束才執行下一個。每邊case依n遞增，再4／12blocks。計時期間不額外平行執行其他benchmark。

CPU使用一個OpenMP thread，目的是固定可重現條件，不代表最佳CPU設定。系統背景負載、動態clock、散熱與執行順序仍可能影響結果；本日未鎖頻或控制所有OS活動。保存CPU型號、CUDA-Q／Python／NumPy版本與GPU run後快照，GPU溫度不是全程telemetry。

## 5. 先確認答案，再看Speed Ratio

所有重複執行都保存expectation，檢查finite與spread。小case(4／8qubits、4blocks)另與Day24 NumPy reference核對；其餘case由CPU／GPU同電路交叉比對，不宣稱所有16qubit結果都有第三份獨立reference。

報告產生器確認protocol、weights hash、fp64主比較與輸出誤差<1e-5後，才產生：

```text
ratio = CPU warm median / GPU warm median
ratio > 1：此設定下GPU較快
ratio < 1：此設定下CPU較快
```

小電路的固定呼叫成本可能佔主要比例；較大statevector有更多可平行計算工作。但交叉點由本機實測決定，不保證GPU對所有n或所有電路都更快。

![Latency](../../results/day27/latency.png)

完整first／warm時間、ratio與數值誤差在[結果報告](../../results/day27/README.md)。不把最好的單個ratio當作「CUDA-Q GPU普遍快幾倍」。

## 6. 記憶體只是下限，不是容量保證

Statevector需要2^n個complex values。單份buffer下限：

```text
fp32 complex：8 × 2^n bytes
fp64 complex：16 × 2^n bytes
```

本日16qubits分別0.5MiB／1MiB，遠低於RTX3060的6GiB總VRAM；但runtime還可能配置workspace、額外state與其他buffer，所以不能用6GiB除上式就宣稱最大可跑qubit數。沒有量測peak RAM／VRAM、OOM上限或multi-GPU容量。

Day26 density matrix需要4^n entries，noisy trajectory又有重複路徑的成本；不要把那些時間與本日exact single-observe直接混成同一曲線。

## 7. 重跑與單一Backend範例

使用[requirements-day27.txt](../../requirements-day27.txt)，建議在沒有其他重負載時依序執行：

```bash
source .venv/bin/activate
# 三backend依序執行，完成後產生報告與圖表。
OMP_NUM_THREADS=1 python articles/day27/run_all.py

# 也可只測一邊；結果寫入同名目錄。
OMP_NUM_THREADS=1 python articles/day27/benchmark.py --backend qpp-cpu
OMP_NUM_THREADS=1 python articles/day27/benchmark.py --backend nvidia-fp64
OMP_NUM_THREADS=1 python articles/day27/benchmark.py --backend nvidia
OMP_NUM_THREADS=1 python articles/day27/plot_results.py
OMP_NUM_THREADS=1 python articles/day27/audit_results.py
```

`run_all.py`包含CPU單獨量測及GPU量測，不需重跑Day25訓練。沒有GPU可單獨執行qpp-cpu；完整比較報告需要三邊資料。結果在`results/day27/`，包含protocol、process_times、各backend原始records與summary。單獨重跑某backend後，舊process_times不代表新run；完整重跑請用run_all。

每backend8cases×10calls=80次observe，三backend合計240次。所有數值檢查在報告前完成；benchmark不以「GPU必須贏」作為通過條件。重跑會覆寫同名檔案，請不要同時啟動多個run_all。

GPU結束時若出現`cudaErrorCudartUnloading`，本日以process exit與保存的結果檢查判斷完成狀態；未定位該訊息根因。

## 8. 下一步與來源

Day28檢視multi-GPU的介面、資源與單GPU硬體限制，不把單張RTX3060假裝成多GPU實測。

[D24] [NVIDIA State Vector Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html)：CPU／GPU simulator與precision options，查閱2026-09-07。另沿用[Day24](../day24/README.md)的kernel與NumPy reference。完整[參考索引](../../REFERENCES.md)。
