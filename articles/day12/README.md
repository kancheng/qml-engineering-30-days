# Day 12｜Angle Encoding：角度範圍、旋轉軸與讀出

Day 11 示範 `θ=πx`，也發現 −1 與 +1 編碼後碰撞。今天保留同一份 2D 資料與 train-only scaler，改變 **角度映射、旋轉軸、初始態**，看量子態與讀出如何改變。

實作：[angle_encoding.py](angle_encoding.py)；單筆示範：[demo.py](demo.py)；完整實驗：[experiment.py](experiment.py)。本日沒有訓練 weights。

## 1. 一個 Feature 對應一個 Rotation

CUDA-Q 的 Pauli rotation 慣例為 `R_P(θ)=exp(−iθP/2)`，角度使用 radians。[D5] 最簡單的兩個 feature 編碼：

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

沒有 entangling gate，結果是 product state：

```text
|ψ(x)⟩ = [cos(θ0/2)|0⟩ + sin(θ0/2)|1⟩]
       ⊗ [cos(θ1/2)|0⟩ + sin(θ1/2)|1⟩]
```

本例兩個 feature 用兩個 qubit、兩個旋轉 gate，理想可平行 logical depth 為 1。H→RZ 版本多兩個 H，depth 為 2；不包含量測基底轉換、硬體路由或 compiler optimization。這些是本例結構的計數，不是所有資料載入方式的成本結論。

## 2. 縮放與角度映射分開

沿用 Day 11 的 `fit_scaler(train)` 與 `transform(raw, scaler)`；只有 train 決定 min/max，holdout 越界依原政策 clipping 並保留旗標。輸出 x 在 `[-1,1]`，之後再選映射：

| CLI mapping | θ(x) | 角度範圍 | 本日 RY 端點的 fidelity |
|---|---|---|---:|
| `centered_full` | πx | [−π,π] | 1：相同物理態 |
| `centered_half` | πx/2 | [−π/2,π/2] | 0：正交 |
| `positive_half` | π(x+1)/2 | [0,π] | 0：正交 |

端點比較只改 x0：`[-1,0]` 與 `[1,0]`。預設示範改用 `positive_half`，是為了展示 Z 讀出單調的情況；既有 Day 9–11 的 API 與模型沒有改動。這不是經過分類準確率比較後的最佳設定。

```python
scaler = fit_scaler(train)
scaled, clipped = transform(holdout, scaler)
angles = to_angles(scaled[0], mapping='positive_half')
```

`to_angles` 接受**已縮放**的兩個 features，拒絕 NaN、Inf、越界或錯誤 shape。不要把原始 `[25,150]` 直接傳給它，也不要把已是 radians 的資料再乘一次 π。`validate` 是低階 radians API，可接受範圍外的有限角度，以保留旋轉週期性。

## 3. 範圍如何影響資料的可區分性？

對 RY product encoding，兩筆角度向量的 fidelity 可直接推導：

```text
F(a,b) = |⟨ψ(a)|ψ(b)⟩|²
       = ∏_j cos²((θ_j(a)−θ_j(b))/2)
```

這裡的 fidelity 是量子態重疊，不是模型 accuracy。F=1 表示只差 global phase；F=0 表示正交。縮短角度範圍可以避開本例端點重合，但也會改變其他資料間的距離：

| 只改 x0 的 pair | centered_full | centered_half | positive_half |
|---|---:|---:|---:|
| −1 與 +1 | 1 | 0 | 0 |
| −0.25 與 +0.25 | 0.5 | 約 0.853553 | 約 0.853553 |

兩種 half 映射只差共同角度偏移，所以 pair fidelity 相同；但讀出不同。RY 的 Z expectation 是 cosθ：`centered_half` 的正負 x 仍有相同 Z 值，`positive_half` 則在 x∈[−1,1] 上嚴格遞減。Exact Z 可以區分本例的單一 scaled feature，但有限 shots 仍有估計誤差，不能當成無限精度資料還原。

任何角度策略都無法恢復 Day 11 clipping 已經丟失的差異。改映射與修復原始資料資訊損失是不同問題。

## 4. RX、RY、RZ 不只是替換函式名稱

對每個 qubit，從 |0⟩ 出發可推得：

| axis 選項 | 操作順序 | ⟨X⟩ | ⟨Y⟩ | ⟨Z⟩ |
|---|---|---|---|---|
| `ry` | RY(θ) | sinθ | 0 | cosθ |
| `rx` | RX(θ) | 0 | −sinθ | cosθ |
| `rz` | RZ(θ) | 0 | 0 | 1 |
| `h_rz` | 先 H，再 RZ(θ) | cosθ | sinθ | 0 |

RX 的 Y 符號是負號，符合上述 rotation convention；測試使用 Day 3 矩陣獨立核對。

直接 RZ(θ)|0⟩ 只產生 global phase，不管角度如何改，物理態都相同。先 H 建立 |+⟩，再 RZ 才會把角度放進相對相位。`h_rz` 的矩陣乘法順序是 `RZ(θ) @ H @ |0⟩`，不要寫反。

即使 h_rz 已有相對相位，Z 基底 probabilities 仍不依賴角度；需要 X／Y 等讀出才能看見。實驗對每個設定計算 q0 的 X、Y、Z expectation，並核對完整兩個 qubit state，因此 q1 也包含在 correctness check 中。

## 5. CUDA-Q 實作邊界

`prepare(q, angles, axis)` 是共用 sub-kernel，axis 的 host 字串轉成整數：0=RY、1=RX、2=RZ、3=H→RZ。host 的 `validate` 拒絕不支援的選項；直接使用底層 kernel 時須遵守這個契約。

`encoded` 只做 state preparation，供 `get_state` 與 `observe` 使用；`measured` 共用 prepare，再加 `mz(q0)`、`mz(q1)`。state 以 `State.amplitude('00')` 等 labels 取出，明確使用 `|q0 q1⟩` 的 NumPy 順序。

```python
angles = to_angles([0.25, -0.4], 'positive_half')
_, axis_code = validate(angles, 'ry')
value = cudaq.observe(encoded, cudaq.spin.z(0),
                      angles, axis_code, shots_count=-1).expectation()
```

NVIDIA 文件也提供 `cudaq.contrib.angular_encode`；本日明寫 gates，方便看清 H 與 RZ 的順序，沒有依賴該 helper。[D5]

## 6. 執行單筆與完整實驗

沿用 `.venv`，沒有新增依賴；[requirements-day12.txt](../../requirements-day12.txt) 延續固定版本鏈。在專案根目錄：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day12/demo.py
OMP_NUM_THREADS=1 python articles/day12/demo.py --features -0.25 0 --mapping centered_half --axis rx
OMP_NUM_THREADS=1 python articles/day12/demo.py --axis h_rz --backend nvidia

OMP_NUM_THREADS=1 python articles/day12/experiment.py
OMP_NUM_THREADS=1 python articles/day12/experiment.py --backend nvidia
```

預設 demo 的 scaled features 是 `[0.25,-0.4]`，positive_half angles 約為 `[1.963495,0.942478]`。RY 的 q0 預期 X≈0.923880、Y=0、Z≈−0.382683。

每個 backend 的實驗包含：

- Day 11 的 4 筆 train＋4 筆 holdout × 3 mappings × 4 axis，共 96 組設定。
- 2 組資料 pair × 3 mappings，6 組 RY fidelity 比較。
- 8 筆 positive_half／RY 設定各 1,000 shots 的 Z 基底 sampling。

保存 `predictions.json`、`pairs.json`、`summary.json`、`angle_sweep.csv` 與預設 RY `circuit.txt`。預設目錄為 `results/day12/<backend>/`；重跑會更新，可加 `--output-dir /tmp/day12-check` 另存。

## 7. 驗證與限制

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day12 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY12_TARGET=nvidia python -m unittest discover -s articles/day12 -p 'test_*.py' -v
```

七個測試涵蓋角度端點與錯誤輸入、Day 11 相容性、pair overlap、Bloch 符號、各軸複數態、RZ 與 H→RZ 差異，以及 positive_half 的 Z 單調性。完整 CPU／GPU 數字見 [結果紀錄](../../results/day12/README.md)。

這是無噪聲 simulator 上的表示與正確性實驗。完整 state 與 exact expectation 不等同硬體 QPU 的直接輸出；沒有 classifier、訓練、encoding 效能排名或 GPU 加速宣稱。下一篇 [Day 13](../day13/README.md) 改看 Amplitude Encoding 的正規化與 state preparation。

## 8. 來源

[D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：angular encoding 與 Pauli rotation convention。查閱日期 2026-09-06，實際 CUDA-Q 0.15.1。映射、Bloch 表格與 fidelity 公式由 Day 3 gate matrices 推導並測試；來源索引見 [REFERENCES.md](../../REFERENCES.md)。
