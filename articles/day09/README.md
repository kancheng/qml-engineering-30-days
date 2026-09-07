# Day 09｜Parameterized Quantum Circuit：把資料與可訓練參數分開

Day 8 已經能用 `observe` 取得電路的數值輸出。今天要把電路變成一個具有明確參數介面的模型：**輸入資料 x 決定 feature map，weights 決定 Ansatz，observable 決定讀出方式。**

本日建立可調參數的 forward model，驗證參數如何影響輸出。Day 10 才加入 loss 與 optimizer，因此今天的 weights 都是指定或初始化的，尚未經過訓練。

## 1. Kernel Argument 不一定是 Trainable Parameter

前幾天已經傳過角度、qubit 數與 bool。這些都是 kernel arguments，但用途不同：

| 項目 | 本日表示 | 是否交給 optimizer 更新 |
|---|---|---|
| 輸入資料 | `features=[x0,x1]` | 否 |
| 資料編碼角度 | `angles=π*features` | 否，由資料決定 |
| 模型權重 | `weights` | 預留給 Day 10 更新 |
| 結構設定 | `layers` | 本日固定後執行 |
| 量測次數 | `shots` | 否，是執行設定 |

CUDA-Q 可以以 `list[float]` 接收參數化 kernel 的角度。[D10] 但「有可調角度」與「已經學會某個任務」是兩回事。

## 2. Ansatz 是我們選擇的電路結構

本日固定兩個 Qubit，每層使用：

```text
q0: ──RY(w[4l])────●──RY(w[4l+2])──
                   │
q1: ──RY(w[4l+1])───X──RY(w[4l+3])──
```

每層有四個 weights，L 層共有 `4L` 個。排列規則固定為：

| 層內 offset | 操作 |
|---:|---|
| 0 | CNOT 前的 q0 RY |
| 1 | CNOT 前的 q1 RY |
| 2 | CNOT 後的 q0 RY |
| 3 | CNOT 後的 q1 RY |

兩個不同 qubit 上的 RY 可視為同一 logical layer，因此每個 block 深度為 3；加上最前面的 feature map，未經 compiler optimization 的準備深度為 `1+3L`。相鄰同軸旋轉可能合併，不能把這個教學層數當成編譯後的硬體深度。

這個 RY-only Ansatz 方便與 NumPy 矩陣逐步核對。它從實數 amplitudes 出發，不能產生任意複數態；我們沒有宣稱它是通用電路或最佳模型架構。

## 3. Feature Map 與 Ansatz 的程式邊界

完整代碼位於 [pqc.py](pqc.py)。輸入 features 限定兩個有限數值，範圍 `[-1,1]`，host 轉成 radians：

```python
angles = (np.pi * np.asarray(features)).tolist()
```

正式版本另外檢查 features shape、weights 長度與 finite values；weights 可超過 `[-π,π]`，不會靜默 clipping。示範 layers 限為 1–3，這是本日程式的範圍限制。

```python
@cudaq.kernel
def feature_map(q: cudaq.qview, angles: list[float]):
    ry(angles[0], q[0])
    ry(angles[1], q[1])

@cudaq.kernel
def ansatz(q: cudaq.qview, weights: list[float], layers: int):
    for layer in range(layers):
        offset = 4 * layer
        ry(weights[offset], q[0])
        ry(weights[offset + 1], q[1])
        x.ctrl(q[0], q[1])
        ry(weights[offset + 2], q[0])
        ry(weights[offset + 3], q[1])
```

外層 model 只配置 register 並組合它們：

```python
@cudaq.kernel
def model(angles: list[float], weights: list[float], layers: int):
    q = cudaq.qvector(2)
    feature_map(q, angles)
    ansatz(q, weights, layers)
```

同一組 weights 用於不同的輸入資料；資料不是 optimizer 每次任意重寫的參數。第一層 feature map RY 與 Ansatz RY 在同一 qubit 上相鄰，可合成角度相加；此處分開寫，是為了維持資料與模型權重的語意邊界，不是宣稱兩者形成不可合併的物理操作。

## 4. Readout：本日選擇 Z0 Z1

模型定義為：

```text
|ψ(x,w)⟩ = U_ansatz(w) U_encoding(x)|00⟩
f(x,w) = ⟨ψ(x,w)|Z0 Z1|ψ(x,w)⟩
```

Exact simulator forward pass：

```python
value = cudaq.observe(model, cudaq.spin.z(0) * cudaq.spin.z(1),
                      angles, weights, layers, shots_count=-1).expectation()
```

本日輸出位於 `[-1,1]`，是 parity expectation。它不是現成的分類機率或 accuracy；若後續需要分類輸出，仍須定義 label、mapping 與 loss。

finite-shot 版本使用獨立的 `sampled_model`，共用同一份 feature map／Ansatz，再加上 `mz(q[0])` 與 `mz(q[1])`。後處理為：

```text
f_sample = (n00 + n11 − n01 − n10) / shots
```

注意這和 Day 5 的 Z0 公式不同。observable 是模型的一部分，不能只換電路而沿用不相符的後處理。

延續 Day 8，`model` 本身不含量測。開發時確認此版本的 `observe` 會拒絕含量測分支的 PQC，即使呼叫時把 bool 傳成 false；因此最終實作使用兩個 readout wrappers，不依賴 compiler 是否消除某個分支。[D3][D5]

## 5. 初始化與可重用的 Forward API

```python
weights = np.random.default_rng(42).normal(0, 0.2, size=4 * layers)
```

這是本日固定 seed 的小角度初始化，不是經過比較後得出的最佳策略。實驗另外保留零向量與 seed 43 作對照。

`pqc.py` 對 host 提供：

| 函式 | 用途 |
|---|---|
| `parameter_count(layers)` | 計算 weights 長度 |
| `initialize(layers, seed)` | 可重現的初始 weights |
| `predict(features, weights, layers)` | CUDA-Q exact forward scalar |
| `reference_prediction(...)` | NumPy 矩陣參考 |
| `sample_prediction(...)` | counts、shots、seed 與 finite-shot scalar |

呼叫端先選 target，再評估多組資料；forward API 不會在背後切換 backend，也不修改傳入 weights。Day 10 的 optimizer 可以產生新的候選 weights，再交給這個介面計算輸出。

## 6. 可執行示範

沿用既有 `.venv`，沒有新增套件；全新環境可由 [requirements-day09.txt](../../requirements-day09.txt) 安裝固定 stack。

```bash
source .venv/bin/activate

# 一層、四個 weights，預設 features=[0.25,-0.4]、初始化 seed=42。
OMP_NUM_THREADS=1 python articles/day09/demo.py

# 手動指定 weights，確認資料和 weights 是不同引數。
OMP_NUM_THREADS=1 python articles/day09/demo.py --features 0.25 -0.4 --weights 0 0 0 0

# 兩層、八個 weights，改用 RTX 3060。
OMP_NUM_THREADS=1 python articles/day09/demo.py --backend nvidia --layers 2
```

完整 CLI 在 [demo.py](demo.py)，會印出電路圖、features、encoded angles、weights、exact 與 NumPy reference，以及 1,000 shots 的 counts。

預設 CPU 示範：

```text
weights ≈ [0.0609434, -0.2079968, 0.1500902, 0.1881129]
exact ZZ ≈ 0.24571394
NumPy reference ≈ 0.24571394
sampled ZZ = 0.252
counts = {00:568, 01:301, 10:73, 11:58}
```

finite-shot 結果與 exact 的差異是本次抽樣的波動，不是 weights 被更新了。

## 7. 只改一個 Weight，輸出會怎麼變？

固定 features、L=1、seed 42 的初始 weights，依序對每個 weight 加上 −0.5、0、+0.5，其餘不動。CPU 的 exact 結果約為：

| Weight | −0.5 | 原始值 | +0.5 |
|---|---:|---:|---:|
| w0 | 0.285360 | 0.245714 | 0.171108 |
| w1 | −0.238628 | 0.245714 | 0.675024 |
| w2 | 0.165546 | 0.245714 | 0.265722 |
| w3 | −0.134586 | 0.245714 | 0.565854 |

這說明在選定設定下，四個參數都能影響輸出。這是有限改動的 response scan，不是梯度估計，也不能證明所有初始值、資料或深度都容易訓練。Barren Plateau 等問題留待後續章節。

RY 的角度增加 `2π` 可帶來 global phase，但 observable 不變；測試會逐個 weight 核對輸出週期性。這也提醒我們：參數向量不同，不一定代表不同的可觀察模型。

## 8. 零 Weights 不代表整個 Ansatz 是 Identity

當 weights 全為 0，RY 都是 identity，但 CNOT 仍存在：

```text
一層零 weights：仍有一個 CNOT
兩層零 weights：CNOT × CNOT = I
```

對本日 feature map 與 ZZ readout，可推得：

| 零 weights 層數 | Exact ZZ |
|---|---|
| 奇數 L | `cos(π*x1)` |
| 偶數 L | `cos(π*x0)*cos(π*x1)` |

測試使用 L=1、2、3 核對這個獨立解析結果。它既能檢查迴圈，也能避免把「所有 rotation 都是 0」誤解成整個電路沒有作用。零初始化是對照案例，不保證每個參數在該點都對輸出敏感。

## 9. 實驗與結果檔案

```bash
OMP_NUM_THREADS=1 python articles/day09/experiment.py --backend qpp-cpu
OMP_NUM_THREADS=1 python articles/day09/experiment.py --backend nvidia
```

每個 backend 包含：

- L=1、2 × 零／seed42／seed43 weights × 4 組 features，共 24 組 forward cases。
- 一層模型的 4 個 weights × 3 種 delta，共 12 組 response cases。
- 每組 forward case 使用 exact observe、NumPy state／expectation reference，以及 1,000 shots、seed 42 的 sampling。

資料保存在 `results/day09/<backend>/`：`pqc_sweep.csv`、`raw_results.json`、`summary.json` 與 `circuit.txt`。response 的 features、base weights、完整候選 weights 與輸出都寫入 summary。重跑更新相同 backend，可用 `--output-dir /tmp/day09-check` 另存。

測試指令：

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day09 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY09_TARGET=nvidia python -m unittest discover -s articles/day09 -p 'test_*.py' -v
```

本日兩個 backend 各 7 個測試通過，forward 與 response 也全數通過。完整數字、precision 與執行限制見 [結果紀錄](../../results/day09/README.md)。

NumPy 是本日的 classical correctness reference；本日沒有學習任務或資料集成效比較。有限幾個設定的 fidelity／expectation 檢查不代表所有可能參數都已驗證，計時也不作 GPU 效能宣稱。

## 10. 本日來源與下一篇

- [D10] [NVIDIA Building Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html)：`list[float]` 參數化 kernel。
- [D3] [NVIDIA Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)：execution 與 observable。
- [D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：observe、sample 與 State.amplitude。

本日 Ansatz 是教學設計，NumPy reference 重用 Day 3 的 RY 與 Day 4 的 CNOT。未引用它來宣稱特定研究方法的效果。查閱日期：2026-09-06，實際版本 CUDA-Q 0.15.1，共用索引見 [REFERENCES.md](../../REFERENCES.md)。

[Day 10](../day10/README.md) 會在這個 forward API 外加入 loss 與 classical optimizer，第一次讓 weights 由最佳化過程更新，完成第二階段的 Hybrid Optimization Loop。
