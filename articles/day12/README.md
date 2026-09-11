# Day 12｜角度編碼：角度範圍、旋轉軸與量測方式

Day 11 用 `θ = πx` 把縮放後的特徵轉成角度，也看到 −1 與 +1 可能變成同一個物理態。今天固定同一份資料與同一把「只看訓練集」的縮放尺，改動三件事：**角度怎麼映、繞哪一軸轉、從什麼初始態出發**，看狀態與讀出會怎麼變。

用音量旋鈕打比方：把數字刻在旋鈕上，不保證兩個刻度真的對應兩種聽得見的差別。刻度範圍太寬，兩端可能繞回同一點；轉錯軸，或一開始就不在能聽見變化的位置，指針動了你也可能聽不出來。量測方式就像你用哪種耳朵去聽。

實作：[angle_encoding.py](angle_encoding.py)；單筆示範：[demo.py](demo.py)；完整實驗：[experiment.py](experiment.py)。本日**沒有**訓練權重。

Day12 keeps Day11's train-only scaler and varies angle mappings, rotation axes, and initial states for two-feature angle encoding. Fidelity tables and Bloch expectations separate encoding collisions from readout blind spots. NumPy and CUDA-Q checks verify the circuits without claiming classifier accuracy, encoding rankings, or QPU results.

---

## 1. 一個特徵，對應一個旋轉

**角度編碼**把資料當成量子閘的旋轉角度。單一量子位元的狀態可用**布洛赫球**想像成球面上的一點；RX、RY、RZ 分別繞 x、y、z 軸轉。

CUDA-Q 的慣例是 `R_P(θ) = exp(−iθP/2)`，角度用弧度。[D5] `P` 是 X／Y／Z；`i` 滿足 `i² = −1`；π 是半圈。最簡單的兩個特徵：

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

`qview` 操作已建立的位元；`list[float]` 是浮點數清單。這裡沒有連接兩個位元的閘，結果是**乘積態**：能拆成兩個獨立狀態。

```text
|ψ(x)⟩ = [cos(θ0/2)|0⟩ + sin(θ0/2)|1⟩]
       ⊗ [cos(θ1/2)|0⟩ + sin(θ1/2)|1⟩]
```

`⊗` 是張量積。兩個特徵、兩個旋轉，理想上可平行，邏輯深度為 1。若改成先 H 再 RZ，深度變 2。這是本例結構計數，不含硬體繞線或編譯合併後的成本，也不能推成所有資料載入方式的結論。

## 2. 縮放歸縮放，角度映射歸映射

縮放器仍只用訓練資料訂最小／最大值；保留資料越界依 Day 11 政策截斷並留旗標。縮放後的 `x` 落在 `[-1, 1]`，**之後**再選怎麼變成角度：

| 命令列選項 | θ(x) | 角度範圍 | RY 下 −1 與 +1 的保真度 |
|---|---|---|---:|
| `centered_full` | πx | [−π, π] | 1：相同物理態 |
| `centered_half` | πx/2 | [−π/2, π/2] | 0：正交（可完全區分） |
| `positive_half` | π(x+1)/2 | [0, π] | 0：正交 |

**保真度（fidelity）**衡量兩個狀態有多像：1 表示相同物理態，0 表示正交。它不是分類準確率。

端點比較只改第一個特徵：`[-1, 0]` 與 `[1, 0]`。預設示範改用 `positive_half`，是為了展示 Z 讀出隨 x 單調變化；Day 9–11 既有介面沒有改。這**不是**用分類成績挑出來的最佳設定。

```python
scaler = fit_scaler(train)
scaled, clipped = transform(holdout, scaler)
angles = to_angles(scaled[0], mapping='positive_half')
```

`to_angles` 只接受**已縮放**的兩個特徵；拒絕無效數值、無限大、越界或形狀錯誤。不要把原始 `[25, 150]` 直接丟進去，也不要把已經是弧度的數再乘一次 π。低階的 `validate` 可接受範圍外的有限角度，以保留旋轉的週期性。

## 3. 範圍一改，誰跟誰比較像也會改

對本日這種 RY 乘積態，兩筆資料的保真度可寫成：

```text
F(a,b) = |⟨ψ(a)|ψ(b)⟩|²
       = ∏_j cos²((θ_j(a)−θ_j(b))/2)
```

`a`、`b` 是兩筆資料；`∏` 表示各特徵貢獻相乘。只改 `x0` 時：

| 資料配對 | centered_full | centered_half | positive_half |
|---|---:|---:|---:|
| −1 與 +1 | 1 | 0 | 0 |
| −0.25 與 +0.25 | 0.5 | 約 0.853553 | 約 0.853553 |

兩種「半範圍」映射只差共同角度偏移，所以配對保真度相同，但**讀出不同**。RY 的 Z 期望值是 `cos θ`：`centered_half` 下正負 x 仍可能得到相同 Z；`positive_half` 在 `x ∈ [−1, 1]` 上嚴格遞減（x 越大，理論 Z 越小）。精算 Z 能區分本例單一縮放特徵，但有限次量測仍有估計誤差，不能當成無限精度還原。

任何角度策略都**救不回** Day 11 截斷已經弄丟的差異。改映射，與修復原始資料損失，是兩件事。

## 4. RX、RY、RZ：不是換函式名稱而已

從 `|0⟩` 出發，各軸讀出不同：

| 選項 | 操作順序 | ⟨X⟩ | ⟨Y⟩ | ⟨Z⟩ |
|---|---|---|---|---|
| `ry` | RY(θ) | sin θ | 0 | cos θ |
| `rx` | RX(θ) | 0 | −sin θ | cos θ |
| `rz` | RZ(θ) | 0 | 0 | 1 |
| `h_rz` | 先 H，再 RZ(θ) | cos θ | sin θ | 0 |

直覺：

- 直接對 `|0⟩` 做 RZ，只轉出**整體相位**（所有振幅乘同一個長度為 1 的複數）——物理態沒變，角度怎麼改都一樣
- 先用 H 把 `|0⟩` 變成 `|+⟩ = (|0⟩+|1⟩)/√2`，再 RZ，角度才進入**相對相位**（同一個狀態裡，不同振幅的方向差）
- 即使有相對相位，Z 基底機率仍可能看不出角度；需要 X／Y 等讀出才看得見

`h_rz` 的矩陣順序是 `RZ(θ) @ H @ |0⟩`，不要寫反。實驗對每個設定算 q0 的 X／Y／Z，並核對完整兩位元狀態，因此 q1 也在正確性檢查內。RX 的 Y 為負號，符合上述旋轉定義；測試用 Day 3 矩陣獨立核對。

## 5. CUDA-Q 實作怎麼切邊界

主控端檢查輸入並安排執行；量子核心程式描述電路。共用子函式 `prepare(q, angles, axis)` 把軸的字串編成整數：0=RY、1=RX、2=RZ、3=H→RZ。不支援的選項在主控端拒絕。

`encoded` 只準備狀態，給 `get_state`／`observe` 用；`measured` 共用 prepare，再加 `mz(q0)`、`mz(q1)`。狀態用 `State.amplitude('00')` 等標籤取出，順序明確為 `|q0 q1⟩`。

```python
angles = to_angles([0.25, -0.4], 'positive_half')
_, axis_code = validate(angles, 'ry')
value = cudaq.observe(encoded, cudaq.spin.z(0),
                      angles, axis_code, shots_count=-1).expectation()
```

文件另有 `cudaq.contrib.angular_encode`；本日明寫閘順序，方便看清 H 與 RZ，沒有依賴該輔助函式。[D5]

## 6. 怎麼跑單筆與完整實驗

沿用 `.venv`；固定依賴見 [requirements-day12.txt](../../requirements-day12.txt)。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day12/demo.py
OMP_NUM_THREADS=1 python articles/day12/demo.py --features -0.25 0 --mapping centered_half --axis rx
OMP_NUM_THREADS=1 python articles/day12/demo.py --axis h_rz --backend nvidia

OMP_NUM_THREADS=1 python articles/day12/experiment.py
OMP_NUM_THREADS=1 python articles/day12/experiment.py --backend nvidia
```

預設示範的縮放後特徵是 `[0.25, -0.4]`，`positive_half` 角度約 `[1.963495, 0.942478]`。RY 下 q0 預期約 X≈0.923880、Y=0、Z≈−0.382683。

每個後端實驗包含：

- Day 11 的 4 筆訓練＋4 筆保留 × 3 映射 × 4 旋轉軸 → **96** 組設定
- 2 組資料配對 × 3 映射 → **6** 組 RY 保真度比較
- 8 筆 `positive_half`／RY 各 1,000 次 Z 基底抽樣

結果：`predictions.json`、`pairs.json`、`summary.json`、`angle_sweep.csv`、預設 RY 的 `circuit.txt`。目錄為 `results/day12/<backend>/`；可用 `--output-dir /tmp/day12-check` 另存。

## 7. 驗證與限制

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day12 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY12_TARGET=nvidia python -m unittest discover -s articles/day12 -p 'test_*.py' -v
```

七個測試涵蓋：角度端點與錯誤輸入、Day 11 相容性、資料配對保真度、布洛赫符號、各軸複數態、RZ 與 H→RZ 差異，以及 `positive_half` 的 Z 單調性。完整數字見 [結果紀錄](../../results/day12/README.md)。

這是無雜訊模擬器上的**表示與正確性**實驗：縮放範圍、角度映射、旋轉軸、初始態與讀出分開核對；保真度與期望值表用來區分編碼碰撞與量測盲點。換映射也救不回截斷損失。完整狀態與精算期望值 ≠ QPU 的直接輸出。沒有分類器、沒有訓練、沒有編碼效能排名，也沒有 GPU 加速宣稱。

下一篇 [Day 13](../day13/README.md) 改看振幅編碼的正規化與狀態準備。

## 8. 來源

[D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：角度編碼與 Pauli 旋轉閘慣例。查閱日期 2026-09-06，實際 CUDA-Q 0.15.1。映射、布洛赫表與保真度公式由 Day 3 閘矩陣推導並測試；索引見 [REFERENCES.md](../../REFERENCES.md)。

## 延伸研究

[N3] Seungcheol Oh et al. “Fourier Analysis Perspective on Quantum Neural Networks.” Communications Physics 9, 176 (2026)；觀點論文。[原始來源](https://doi.org/10.1038/s42005-026-02680-x)；[完整書目](../../REFERENCES.md#n3)。

本章比較角度映射；延伸閱讀可從輸出隨輸入變化的週期，理解縮放範圍與旋轉操作為何會影響哪些資料仍可區分。
