# Day 12｜角度編碼：角度範圍、旋轉軸與量測方式

[Day11](../day11/README.md) 建立從原始資料到量子態的前處理與編碼流程，並區分縮放、編碼及量測階段可能造成的資訊損失。Day12 接著深入角度編碼，在固定資料與縮放規則下，觀察角度範圍、旋轉軸與初始態如何影響資料的表示與讀出。

**把資料轉成不同角度，不保證得到不同量子態；得到不同量子態，也不保證目前的量測能分辨。** 角度範圍、初始狀態與量測方式，需要一起考慮。本章將縮放後的數值分別映到三種角度範圍，比較哪些輸入會變成相同物理態，再用 保真度（fidelity，量子態重疊程度）檢查資料之間的關係；這個數值描述狀態的相似程度，並不是分類準確率。另一個重點是旋轉閘必須連同初始態與讀出一起考慮：直接對 `|0⟩` 施加 RZ，只會改變無法觀測的整體相位；先用 H 準備疊加態再施加 RZ，角度才會進入相對相位，但仍需適合的 X／Y 讀出才能看見變化。實作沿用只由訓練資料決定的 縮放器，比較 NumPy 與 CUDA-Q 的狀態及期望值，另以有限次量測 觀察抽樣結果，尚未訓練模型。縮放決定數值範圍，角度映射決定電路操作，而量測決定能讀到哪些差異。更換編碼設定，仍無法恢復先前截斷操作已丟失的資訊。

[Day11](../day11/README.md) established a preprocessing and encoding workflow from raw data to quantum states, distinguishing information loss during scaling, encoding, and measurement. Day12 examines angle encoding in more detail, keeping the data and scaling rules fixed while exploring how the angle range, rotation axis, and initial state affect representation and readout.

**Different angles need not produce different physical states, and different states need not be distinguishable by a chosen measurement.** The angle range, initial state, and measurement must be considered together. Scaled values are mapped into three angle ranges to identify inputs that become the same physical state. Fidelity, a measure of quantum-state overlap, is used to examine relationships between encoded inputs; it measures state similarity rather than classification accuracy. Another key point is that rotation gates must be considered together with the initial state and readout. Applying RZ directly to `|0⟩` changes only an unobservable global phase. Preparing a superposition with H before applying RZ places the angle in the relative phase, but suitable X/Y readouts are still needed to reveal the change. The implementation reuses a scaler fitted only on training data, compares NumPy and CUDA-Q states and expectations, and examines finite-shot samples without training a model. Scaling sets the numerical range, angle mapping sets the circuit operations, and measurement determines which differences become visible. Changing the encoding cannot recover information already lost through clipping.

---

Day 11 示範 `θ=πx`，也發現 −1 與 +1 編碼後碰撞。今天保留同一份兩個特徵的資料與僅由訓練資料決定的縮放器，改變 **角度映射、旋轉軸、初始態**，看量子態與讀出如何改變。

實作：[angle_encoding.py](angle_encoding.py)；單筆示範：[demo.py](demo.py)；完整實驗：[experiment.py](experiment.py)。本日沒有訓練權重。

## 1. 一個特徵對應一個旋轉操作

角度編碼（angle encoding）將資料當成量子閘的旋轉角度。特徵是描述樣本的數值，例如長度與寬度。布洛赫球是單一量子位元狀態的幾何表示，RX、RY、RZ 分別對應繞 x、y、z 軸旋轉。

CUDA-Q 的 Pauli 旋轉閘慣例為 `R_P(θ)=exp(−iθP/2)`，角度使用弧度。[D5] 其中 `P` 代表 X、Y 或 Z 矩陣，`i` 是滿足 `i²=-1` 的虛數單位，`exp` 是指數運算，`θ` 讀作 theta。弧度是角度單位，`π` 弧度等於半圈；下方展開式使用半角 `θ/2`。最簡單的兩個特徵編碼：

```text
q0: |0⟩ ──RY(θ0)──
q1: |0⟩ ──RY(θ1)──
```

```python
@cudaq.kernel
def feature_map(q: cudaq.qview, angles: list[float]):
    for i in range(2):
        ry(angles[i], q[i])
```

程式中的 `qview` 用來操作已建立的量子位元，`list[float]` 表示浮點數清單；浮點數是電腦以有限位數表示的小數。`for` 重複操作兩個位元，`q[i]` 指向編號為 `i` 的位元。

這裡沒有連接兩個位元的操作，結果是乘積態（product state），也就是能拆成兩個獨立狀態的組合。`⊗` 是組合向量的張量積，`cos` 與 `sin` 是餘弦、正弦函數：

```text
|ψ(x)⟩ = [cos(θ0/2)|0⟩ + sin(θ0/2)|1⟩]
       ⊗ [cos(θ1/2)|0⟩ + sin(θ1/2)|1⟩]
```

邏輯深度計算需要依序執行的操作層數，可同時進行的操作算在同一層。H 是 Hadamard 閘，將 `|0⟩` 轉成 `|+⟩ = (|0⟩ + |1⟩)/√2`。

本例兩個特徵用兩個量子位元、兩個旋轉量子閘，理想可平行邏輯深度為 1。H→RZ 版本多兩個 H，深度為 2；不包含量測基底轉換、硬體路由或編譯最佳化。硬體路由是因裝置連接限制而安排額外操作；編譯最佳化則是在維持結果的前提下調整電路。這些是本例結構的計數，不是所有資料載入方式的成本結論。

## 2. 縮放與角度映射分開

縮放器（scaler）保存每欄的數值範圍，將原始值轉到指定區間。保留資料（holdout）不參與規則的計算；截斷（clipping）把越界值改成上下界，旗標記錄哪些值被改動。

沿用 Day 11 的 `fit_scaler(train)` 與 `transform(raw, scaler)`；只有訓練資料決定最小值／最大值，保留資料越界依原政策截斷並保留旗標。輸出 x 在 `[-1,1]`，之後再選映射：

| 命令列映射選項 | θ(x) | 角度範圍 | 本日 RY 端點的保真度 |
|---|---|---|---:|
| `centered_full` | πx | [−π,π] | 1：相同物理態 |
| `centered_half` | πx/2 | [−π/2,π/2] | 0：正交 |
| `positive_half` | π(x+1)/2 | [0,π] | 0：正交 |

命令列選項是終端機傳給程式的設定；這三個名稱分別表示以 0 為中心的全範圍、以 0 為中心的半範圍，以及非負的半範圍。保真度（fidelity）衡量兩個狀態的重疊，1 表示相同物理態，0 表示正交，也就是有合適量測方式能完全區分的狀態。

端點比較只改 x0：`[-1,0]` 與 `[1,0]`。預設示範改用 `positive_half`，是為了展示 Z 讀出單調的情況；既有 Day 9–11 的程式介面與模型沒有改動。這不是經過分類準確率比較後的最佳設定。

```python
scaler = fit_scaler(train)
scaled, clipped = transform(holdout, scaler)
angles = to_angles(scaled[0], mapping='positive_half')
```

`to_angles` 接受**已縮放**的兩個特徵，拒絕 NaN（無效數值）、Inf（無限大）、越界或錯誤形狀；形狀描述輸入的維度與元素數量。不要把原始 `[25,150]` 直接傳給它，也不要把已是弧度的資料再乘一次 π。`validate` 是低階弧度程式介面，可接受範圍外的有限角度，以保留旋轉週期性。

## 3. 範圍如何影響資料的可區分性？

對 RY 乘積態編碼，兩筆角度向量的保真度可直接推導：

```text
F(a,b) = |⟨ψ(a)|ψ(b)⟩|²
       = ∏_j cos²((θ_j(a)−θ_j(b))/2)
```

公式中的 `a`、`b` 是兩筆資料，`j` 是特徵編號，`∏` 表示將各特徵的項相乘，`⟨ψ(a)|ψ(b)⟩` 是兩個狀態向量的內積；對複數向量，先對第一個向量取共軛，再將對應分量相乘加總。

這裡的保真度是量子態重疊，不是模型準確率。F=1 表示只差整體相位，也就是所有振幅乘上同一個長度為 1 的複數，不改變物理預測；F=0 表示正交。縮短角度範圍可以避開本例端點重合，但也會改變其他資料間的距離：

| 只改 x0 的資料配對 | centered_full | centered_half | positive_half |
|---|---:|---:|---:|
| −1 與 +1 | 1 | 0 | 0 |
| −0.25 與 +0.25 | 0.5 | 約 0.853553 | 約 0.853553 |

兩種半範圍映射只差共同角度偏移，所以資料配對保真度相同；但讀出不同。期望值是依機率計算的平均值，準確率則是分類正確的比例。Z 量測將 0 記為 +1、1 記為 −1。RY 的 Z 期望值是 cosθ：`centered_half` 的正負 x 仍有相同 Z 值，`positive_half` 則在 x∈[−1,1] 上嚴格遞減，也就是 x 越大，理論輸出越小。精確 Z 可以區分本例的單一縮放後特徵，但有限量測次數仍有估計誤差，不能當成無限精度資料還原。

任何角度策略都無法恢復 Day 11 截斷已經丟失的差異。改映射與修復原始資料資訊損失是不同問題。

## 4. RX、RY、RZ 不只是替換函式名稱

對每個量子位元，從 |0⟩ 出發可推得：

| 旋轉軸選項 | 操作順序 | ⟨X⟩ | ⟨Y⟩ | ⟨Z⟩ |
|---|---|---|---|---|
| `ry` | RY(θ) | sinθ | 0 | cosθ |
| `rx` | RX(θ) | 0 | −sinθ | cosθ |
| `rz` | RZ(θ) | 0 | 0 | 1 |
| `h_rz` | 先 H，再 RZ(θ) | cosθ | sinθ | 0 |

RX 的 Y 符號是負號，符合上述旋轉閘定義；測試使用 Day 3 矩陣獨立核對。

直接 RZ(θ)|0⟩ 只產生整體相位，不管角度如何改，物理態都相同。先 H 建立 `|+⟩`，再 RZ 才會把角度放進相對相位；相對相位是同一狀態內不同振幅的方向差，會影響後續運算。`h_rz` 的矩陣乘法順序是 `RZ(θ) @ H @ |0⟩`，不要寫反。

即使 h_rz 已有相對相位，Z 基底機率仍不依賴角度；需要 X／Y 等讀出才能看見。讀出是選擇量測方式並將結果轉成數值；不同基底相當於以不同的一組基本狀態區分結果。實驗對每個設定計算 q0 的 X、Y、Z 期望值，並核對完整兩個量子位元狀態，因此 q1 也包含在正確性檢查中。

## 5. CUDA-Q 實作邊界

主控端（host）是一般 Python 程式，負責檢查輸入與安排執行；量子核心程式（quantum kernel）描述電路操作。`prepare(q, angles, axis)` 是共用子函式，旋轉軸的主控端字串轉成整數：0=RY、1=RX、2=RZ、3=H→RZ。主控端的 `validate` 拒絕不支援的選項；直接使用底層量子核心程式時須遵守這個契約。

`encoded` 只做狀態準備，供 `get_state` 與 `observe` 使用；`measured` 共用 prepare，再加 `mz(q0)`、`mz(q1)`。狀態以 `State.amplitude('00')` 等標籤取出，明確使用 `|q0 q1⟩` 的 NumPy 順序。

```python
angles = to_angles([0.25, -0.4], 'positive_half')
_, axis_code = validate(angles, 'ry')
value = cudaq.observe(encoded, cudaq.spin.z(0),
                      angles, axis_code, shots_count=-1).expectation()
```

NVIDIA 文件也提供 `cudaq.contrib.angular_encode`；本日明寫量子閘，方便看清 H 與 RZ 的順序，沒有依賴該輔助函式。[D5]

## 6. 執行單筆與完整實驗

CPU 是一般電腦的中央處理器，GPU 是擅長平行運算的圖形處理器。執行後端（backend）指定實際使用的模擬器；本章預設使用 CPU，`nvidia` 使用 GPU。`.venv` 是專案獨立保存 Python 套件的虛擬環境；`OMP_NUM_THREADS=1` 指定 CPU 平行工作的執行緒數為 1。

沿用 `.venv`，沒有新增依賴；[requirements-day12.txt](../../requirements-day12.txt) 延續固定版本鏈。在專案根目錄：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day12/demo.py
OMP_NUM_THREADS=1 python articles/day12/demo.py --features -0.25 0 --mapping centered_half --axis rx
OMP_NUM_THREADS=1 python articles/day12/demo.py --axis h_rz --backend nvidia

OMP_NUM_THREADS=1 python articles/day12/experiment.py
OMP_NUM_THREADS=1 python articles/day12/experiment.py --backend nvidia
```

預設示範的縮放後特徵是 `[0.25,-0.4]`，positive_half 角度約為 `[1.963495,0.942478]`。RY 的 q0 預期 X≈0.923880、Y=0、Z≈−0.382683。

每個執行後端的實驗包含：

- Day 11 的 4 筆訓練資料＋4 筆保留資料 × 3 映射方式 × 4 旋轉軸，共 96 組設定。
- 2 組資料資料配對 × 3 映射方式，6 組 RY 保真度比較。
- 8 筆 positive_half／RY 設定各 1,000 量測次數的 Z 基底抽樣。

JSON 用欄位名稱保存結構化資料，CSV 是表格文字檔，TXT 保存純文字。

保存 `predictions.json`、`pairs.json`、`summary.json`、`angle_sweep.csv` 與預設 RY `circuit.txt`。預設目錄為 `results/day12/<backend>/`；重跑會更新，可加 `--output-dir /tmp/day12-check` 另存。

## 7. 驗證與限制

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day12 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY12_TARGET=nvidia python -m unittest discover -s articles/day12 -p 'test_*.py' -v
```

七個測試涵蓋角度端點與錯誤輸入、Day 11 相容性、資料配對的狀態重疊、布洛赫球符號、各軸複數態、RZ 與 H→RZ 差異，以及 positive_half 的 Z 單調性。完整 CPU／GPU 數字見 [結果紀錄](../../results/day12/README.md)。

NumPy 是 Python 的數值運算套件，用來核對公式。無噪聲表示未加入使演化或量測偏離理想情況的干擾。這是無噪聲模擬器上的表示與正確性實驗。完整狀態與精確期望值不等同量子處理器（QPU，實際執行量子操作的硬體）的直接輸出。精確值由模擬器保存的向量直接計算，不是有限次抽樣估計，仍可能有浮點誤差；沒有分類器、訓練、編碼效能排名或 GPU 加速宣稱。下一篇 [Day 13](../day13/README.md) 改看振幅編碼的正規化與狀態準備。

## 8. 來源

[D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：角度編碼與 Pauli 旋轉閘 convention。查閱日期 2026-09-06，實際 CUDA-Q 0.15.1。映射、布洛赫球表格與保真度公式由 Day 3 量子閘矩陣推導並測試；來源索引見 [REFERENCES.md](../../REFERENCES.md)。
