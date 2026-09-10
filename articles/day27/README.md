# Day 27｜CPU 與 GPU：如何比較量子模擬效能

[Day26](../day26/README.md) 加入量子噪聲，區分噪聲造成的分布改變與有限次量測的抽樣波動，並觀察已訓練模型的預測如何受到影響。Day27 回到無噪聲模擬，在固定任務與數值精度下，比較 CPU 與 GPU 的執行時間。

**有意義的效能比較，需要固定工作內容，並說清楚時間從哪裡開始、到哪裡結束。** 量子機器學習（QML）會反覆執行電路，了解單次計算的成本，有助於規劃實驗規模。CPU 是中央處理器，GPU 是擅長平行運算的圖形處理器；兩者在這裡都用來模擬量子電路。本章固定電路、參數與量測目標，從完整量子狀態直接計算期望值，也就是量測結果依機率計算的平均，不進行有限次抽樣。主要比較使用相同的雙精度數值，另列 GPU 單精度結果作補充。計時涵蓋程式發出呼叫到結果可用的等待時間，首次呼叫與暖機後的時間分開記錄，並先核對數值一致性，再解讀速度比。小電路可能主要受固定呼叫成本影響，GPU 不一定較快；增加量子位元雖能提供更多平行工作，也會讓記憶體需求快速成長。結果只描述本次單次電路求值，不能直接推論完整訓練、含噪聲模擬或真實量子處理器的速度。

[Day26](../day26/README.md) introduced quantum noise channels, distinguished changes in measurement distributions from finite-shot sampling fluctuations, and examined their effects on a frozen model's predictions. Day27 shifts to execution efficiency, returning to noiseless statevector simulation to compare CPU and GPU latency under fixed tasks and numerical conditions.

**A useful CPU/GPU comparison keeps the task and numerical precision fixed and states exactly what is timed.** QML experiments repeatedly evaluate quantum circuits, so understanding the cost of one evaluation helps with planning experiment sizes and computing resources. The circuit, parameters, and observable remain fixed, and expectation values are calculated directly from the statevector without finite-shot sampling; this exact evaluation still has floating-point limitations. The main comparison uses double precision (fp64) on both CPU and GPU, with GPU single precision (fp32) reported separately. Timing covers the wait from the host call to the available result, including call and runtime overhead. First-call latency is recorded separately from measurements after warmup, and repeated measurements are described through their median and spread. Numerical agreement is checked before interpreting speed ratios. The key lesson is that GPU parallelism needs sufficient work: fixed overhead can dominate small circuits, so a GPU is not automatically faster. Meanwhile, statevector memory requirements grow as 2^n with the number of qubits. Timing boundaries, precision, and hardware conditions determine how these results should be interpreted. These circuit-evaluation measurements do not directly establish the speed of complete QML training, noisy simulation, or real quantum processors.


---

程式：[benchmark.py](benchmark.py)、[run_all.py](run_all.py)、[plot_results.py](plot_results.py)、[結果核對](audit_results.py)。數值與圖表見 [實測報告](../../results/day27/README.md)。沿用 `.venv` 虛擬環境，也就是專案獨立保存套件的目錄，不新增套件。

## 1. 固定比較條件

狀態向量用一組複數振幅描述量子狀態，振幅的絕對值平方決定量測機率。本章直接從狀態向量求值，省去有限次量測的抽樣誤差，但仍有浮點數的捨入誤差。浮點數以有限位數保存數值；fp32 使用 32 位元，fp64 使用 64 位元，後者能保存較高精度，也需要更多記憶體。

| 比較項目 | 本章設定 |
|---|---|
| 相同任務 | RY／RZ／CZ 電路，加上全域 Z 奇偶性期望值 |
| 相同輸入 | 隨機種子 42 產生的權重，保存 SHA-256 摘要 |
| 量子位元數 | 4、8、12、16 |
| 電路區塊數 | 4、12 |
| CPU | `qpp-cpu`、fp64、`OMP_NUM_THREADS=1` |
| GPU 主要比較 | `nvidia`，指定 `option='fp64'` |
| GPU 補充比較 | `nvidia` 預設 fp32 |
| 噪聲與抽樣 | 無噪聲，`shots_count=-1`，直接計算期望值 |
| 每組呼叫次數 | 首次 1 次、暖機 2 次、正式計時 7 次 |

後端（backend）是實際執行模擬的工具，`qpp-cpu` 與 `nvidia` 是 CUDA-Q 的後端名稱。CUDA-Q 是執行量子程式的工具，其文件列出 `nvidia` 的精度選項。[D24] 程式以 `nvidia-fp64` 作為結果目錄標籤，實際呼叫 `cudaq.set_target('nvidia', option='fp64')`，並保存執行環境回報的精度。

隨機種子用來重現權重的產生序列；SHA-256 是由內容計算的摘要，用來確認兩邊使用相同權重。CPU fp64 與 GPU fp32 的數值條件不同，因此分開呈現。

## 2. 固定電路與量測目標

重用 Day24 的量子電路程式，每個區塊先在各量子位元執行 RY、RZ，再依序執行 CZ(0,1)、CZ(1,2) 等相鄰位元操作。RY 與 RZ 分別繞 Y 軸、Z 軸旋轉；CZ 是受控相位閘，對兩位元皆為 1 的狀態分量改變正負號。n 個量子位元、L 個區塊共有 `2nL` 個旋轉角度與 `(n−1)L` 個 CZ。

量測量為 `Z0Z1…Z(n−1)`。每個位元的 Z 量測將 0 記為 +1、1 記為 −1，全部相乘後，偶數個 1 得到 +1、奇數個 1 得到 −1，這就是奇偶性。程式只回傳這個量測量的期望值，不將完整狀態向量傳回 Python。

L 是區塊數，不是編譯後的電路深度；深度計算必須依序完成的操作層數。後端可以合併量子閘，將多個操作一起處理以減少執行成本，實驗未刻意關閉某一邊的最佳化。權重是固定隨機產生的數值，不是訓練結果，輸出也不一定具有分類意義。

## 3. 計時範圍與暖機

```python
start = time.perf_counter()
value = float(cudaq.observe(circuit, observable, n, blocks, weights,
                            shots_count=-1).expectation())
elapsed = time.perf_counter() - start
```

`time.perf_counter()` 提供適合量測經過時間的計時器。此路徑在取得期望值後才停止計時，結果此時已可用，不是只記錄把工作送入等待佇列所需的時間。權重產生與量測量建構在計時範圍之外；範圍內包含主機端呼叫、執行環境處理與實際計算，不能稱為純 GPU 運算時間。主機端在此指發出呼叫、接收結果的 Python 程式。

每組設定保留首次呼叫時間，接著執行兩次暖機，再保存七次正式計時。暖機讓可能只在初次發生的準備工作先完成，有助於觀察後續重複呼叫的成本。

主要結果使用中位數，也就是七個時間排序後的中間值，並列出 Q25 與 Q75（第 25 與第 75 百分位數）。兩者差距稱為四分位距（IQR），描述這批時間的分散程度，不是估計統計不確定性的信賴區間。報告不以最快的一次作為主要結果。

每個後端在獨立程序執行，但同一後端的八組設定共用程序；程序是作業系統中正在執行的一份程式。因此各組的首次呼叫，不保證都包含全新的 JIT（執行時編譯）成本。只有第一組是該程序首次執行電路，磁碟上仍可能保留快取，也就是可重用的編譯結果。

實驗保留既有快取，另記錄整個程序從開始到結束的時間，包含 Python 套件載入、後端設定、檔案輸出與參考計算核對。這個總時間與單次電路計時代表不同範圍。

## 4. 依序執行與硬體條件

`run_all.py` 依 CPU、GPU fp64、GPU fp32 順序啟動子程序，前一個結束後才執行下一個。每個後端先依量子位元數遞增，再比較 4 與 12 個區塊，計時期間不另外平行執行其他效能測試。

`OMP_NUM_THREADS=1` 將 OpenMP 的 CPU 執行緒數設為 1。OpenMP 是讓 CPU 程式平行工作的機制，執行緒是可被排程執行的工作單位。這裡固定一個執行緒以重現條件，不代表已找到最快的 CPU 設定。

背景程式負載、處理器動態時脈、散熱與執行順序仍可能影響結果。時脈是處理器運作的頻率，本章沒有將其固定，也沒有控制作業系統的所有活動。報告保存 CPU 型號、CUDA-Q／Python／NumPy 版本與 GPU 執行後的狀態快照；單次溫度快照不等於全程監測紀錄。NumPy 是參考計算使用的 Python 數值運算套件。

## 5. 先確認答案，再看速度比

每次執行都保存期望值，檢查數值是否有限，以及重複執行間的差異；有限值檢查可找出無限大或非數值等異常。4／8 個量子位元、4 個區塊的小型設定另與 Day24 的 NumPy 參考計算核對。其餘設定由 CPU 與 GPU 交叉比對，並非每個 16 位元結果都有第三份獨立實作驗證。

報告產生器核對實驗設定、權重摘要、fp64 主要比較條件與小於 `1e-5` 的輸出誤差後，才計算速度比：

```text
ratio = CPU warm median / GPU warm median
ratio > 1：此設定下 GPU 較快
ratio < 1：此設定下 CPU 較快
```

分子與分母都是暖機後的時間中位數。比值大於 1，表示該設定下 GPU 等待時間較短；小於 1 則表示 CPU 較短。

小電路可能主要受固定呼叫成本影響，狀態向量較大時才有更多工作可平行處理。GPU 從哪個規模開始較快，需要本機實測，不能預設對所有電路都成立。

![CPU 與 GPU 的計時比較](../../results/day27/latency.png)

完整首次與暖機後時間、速度比及數值誤差見 [結果報告](../../results/day27/README.md)。單一設定的最高速度比不能代表 CUDA-Q 的普遍加速幅度。

## 6. 記憶體估算只是下限

狀態向量需要 `2^n` 個複數，每個複數包含實部與虛部兩個數值。只計算一份狀態向量的儲存空間：

```text
fp32 complex：8 × 2^n bytes
fp64 complex：16 × 2^n bytes
```

每增加一個量子位元，儲存需求就加倍。本章 16 個量子位元分別需要 0.5 MiB／1 MiB。MiB 是 `2^20` 位元組，GiB 是 `2^30` 位元組；一個位元組等於 8 位元。

這些數字遠低於本機 RTX 3060 的 6 GiB 顯示記憶體（VRAM），但執行環境還可能配置暫存工作空間、額外狀態副本與其他緩衝區。因此不能只將總記憶體容量除以上式，就宣稱最大可執行的量子位元數。本章沒有量測主記憶體（RAM）或 VRAM 的最高用量、記憶體不足（OOM）的邊界，也未測試多張 GPU 的容量。

Day26 的密度矩陣以矩陣描述可能混合的量子狀態，需要 `4^n` 個元素；含噪聲軌跡模擬則要重複抽取噪聲作用並演化狀態。兩者與本章單次無噪聲求值的工作量不同，不能直接混在同一條效能曲線中比較。

## 7. 重跑與單一後端範例

使用 [requirements-day27.txt](../../requirements-day27.txt)，在沒有其他重負載時依序執行：

```bash
source .venv/bin/activate
# 三個後端依序執行，完成後產生報告與圖表。
OMP_NUM_THREADS=1 python articles/day27/run_all.py

# 也可只測一邊；結果寫入同名目錄。
OMP_NUM_THREADS=1 python articles/day27/benchmark.py --backend qpp-cpu
OMP_NUM_THREADS=1 python articles/day27/benchmark.py --backend nvidia-fp64
OMP_NUM_THREADS=1 python articles/day27/benchmark.py --backend nvidia
OMP_NUM_THREADS=1 python articles/day27/plot_results.py
OMP_NUM_THREADS=1 python articles/day27/audit_results.py
```

`run_all.py` 包含 CPU 與 GPU 量測，不需重跑 Day25 訓練。沒有 GPU 時可單獨執行 `qpp-cpu`；完整比較報告需要三組後端資料。

結果位於 `results/day27/`，包含實驗設定 `protocol`、程序總時間 `process_times`，以及各後端的原始紀錄與摘要。單獨重跑某個後端後，舊的程序總時間不代表新一次執行；需要完整更新時使用 `run_all.py`。

每個後端 8 組設定 × 10 次呼叫，共 80 次 `observe`；三個後端合計 240 次。`observe` 是計算量測量期望值的呼叫。數值檢查在報告產生前完成，通過條件不要求 GPU 必須較快。重跑會覆寫同名檔案，應避免同時啟動多個 `run_all.py`。

若 GPU 結束時出現 `cudaErrorCudartUnloading`，需一併檢查程序退出狀態與保存結果，才能判斷量測是否完成；該訊息的根因尚未定位。

## 8. 下一步與來源

Day28 將檢視多 GPU 模擬的程式介面與資源需求，並說明單 GPU 環境能驗證的範圍。本機只有一張 RTX 3060，因此沒有多 GPU 實測結果。

[D24] [NVIDIA State Vector Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html)：CPU／GPU 狀態向量模擬器與精度選項，查閱日期 2026-09-07。另沿用 [Day24](../day24/README.md) 的電路與 NumPy 參考計算。完整 [參考索引](../../REFERENCES.md)。

## 延伸研究

[N10] W. Michael Brown et al. “Multi-GPU Quantum Circuit Simulation and the Impact of Network Performance.” Computer Physics Communications 324, 110126 (2026)；研究論文。[原始來源](https://doi.org/10.1016/j.cpc.2026.110126)；[完整書目](../../REFERENCES.md#n10)。

本章測量單 GPU 模擬時間；這篇研究可銜接多 GPU 系統的通訊成本，外部硬體的加速比不能直接套用到 RTX 3060。
