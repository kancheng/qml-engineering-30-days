# Day 09｜參數化量子電路：把資料與可訓練參數分開

Day 8 已經能從電路拿到數字。接下來常見的問題是：**這些數字裡，哪些該跟著這筆資料變，哪些該留給之後的訓練一起調？**

想像一個簡單的天氣預報器：今天的濕度、氣溫是「這次要看的資料」；機器內部那組旋鈕，則是「看過很多天之後才慢慢調好的設定」。旋鈕一換，同一筆濕度會得到不同預測；但濕度本身不該被訓練程序改寫成另一天的濕度。

量子電路也一樣。今天把輸入整理成明確介面：資料決定編碼角度，權重決定電路模板，可觀測量決定讀出方式。我們會核對 CUDA-Q 與 NumPy 的結果，並固定資料、逐一改動權重，觀察輸出怎麼變。權重能影響輸出，只表示電路有可調的部分；還沒有損失函數與最佳化器，因此**今天還不算在學習任務**。

Day9 turns the circuit into a model interface that separates data-dependent angles from shared trainable weights. A two-qubit feature map and layered RY–CNOT ansatz produce a ZZ expectation in `[-1,1]`. CUDA-Q outputs are checked against a NumPy reference; single-weight response scans and zero-weight analytic cases verify the forward pass without claiming training success, universal expressivity, or quantum advantage.

---

## 1. 傳進電路的設定，用途並不一樣

量子核心程式（quantum kernel）描述量子操作；主控端（host）是安排呼叫與整理結果的一般 Python 程式。引數是呼叫時傳入的值，例如角度或整數。

前幾天已經傳過角度、位元數與開關。它們都是輸入，但角色不同：

| 項目 | 本日怎麼寫 | 之後會不會交給最佳化器改？ |
|---|---|---|
| 輸入資料 | `features=[x0,x1]` | 否 |
| 資料編碼角度 | `angles = π × features` | 否，由這筆資料決定 |
| 模型權重 | `weights` | 預留給 Day 10 |
| 結構設定 | `layers`（層數） | 本日先固定再執行 |
| 量測次數 | `shots` | 否，是執行設定 |

`list[float]` 是浮點數清單；浮點數是電腦用有限位數表示的小數。CUDA-Q 可用它接收參數化電路的角度。[D10]

一句話：**資料描述「這次要處理誰」；權重是「很多筆資料共用的旋鈕」。** 有可調角度，不等於已經學會某個任務。

## 2. 先選好電路模板，再談參數

**電路模板（ansatz）**是預先排好的操作組合，像食譜步驟；角度可以改，步驟順序今天先固定。含有可調參數的量子電路，常稱為**參數化量子電路（Parameterized Quantum Circuit，PQC）**。

本章用 RY 旋轉振幅，用 CNOT（受控反相閘）在控制位元為 1 時翻轉目標位元。固定兩個量子位元，每層長這樣：

```text
q0: ──RY(w[4l])────●──RY(w[4l+2])──
                   │
q1: ──RY(w[4l+1])───X──RY(w[4l+3])──
```

`l` 是從 0 開始的層編號，`w` 是權重清單。每層四個權重；L 層共有 `4L` 個：

| 層內位置 | 做什麼 |
|---:|---|
| 0 | CNOT 前，旋轉 q0 |
| 1 | CNOT 前，旋轉 q1 |
| 2 | CNOT 後，旋轉 q0 |
| 3 | CNOT 後，旋轉 q1 |

兩個不同位元上的 RY 可算同一邏輯操作層，因此每個區塊深度為 3；再加上最前面的資料編碼，未經編譯整理前的準備深度為 `1 + 3L`。編譯工具之後可能合併相鄰同軸旋轉，所以**教學層數 ≠ 硬體最終深度**。

這個只含 RY 的模板，方便用 NumPy 矩陣逐步核對。它從實數振幅出發，不能產生任意複數態；本章不宣稱它是通用電路，也不宣稱它是最佳架構。

## 3. 資料編碼與模板：程式上分兩段

**特徵（feature）**是描述樣本的數值，例如兩個長度。**資料編碼電路（feature map）**把它們轉成量子操作。本章規則很直接：乘上 π（半圈）變成角度。

完整代碼在 [pqc.py](pqc.py)。特徵必須是兩個有限數值，範圍 `[-1, 1]`；主控端轉換：

```python
angles = (np.pi * np.asarray(features)).tolist()
```

`np.asarray` 轉成數值陣列，`.tolist()` 再變回 Python 清單。正式版本會檢查形狀、權重長度與有限數值；權重可以超出 `[−π, π]`，不會默默截成邊界。示範層數限為 1–3，是本日程式範圍。

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

`qview` 操作既有位元，不另開暫存器。`for` 依層數重複。外層模型先配置兩個位元，再依序呼叫兩段：

```python
@cudaq.kernel
def model(angles: list[float], weights: list[float], layers: int):
    q = cudaq.qvector(2)
    feature_map(q, angles)
    ansatz(q, weights, layers)
```

同一組權重可以處理不同資料；資料不是最佳化器每次重寫的參數。編碼 RY 與模板 RY 若相鄰，工具理論上可把角度相加合併；這裡分開寫，是為了**語意邊界清楚**，不是宣稱物理上不能合併。

## 4. 讀出：今天用 `Z0 Z1`

**可觀測量（observable）**指定你關心的量。`Z0 Z1`（簡寫 ZZ）把兩個位元的 Z 結果各記成 +1 或 −1，再相乘：兩個結果相同得 +1，不同得 −1。**期望值**是依機率加權後的平均。

把資料寫成 `x`、權重寫成 `w`，電路先編碼再套模板：

```text
|ψ(x,w)⟩ = U_ansatz(w) U_encoding(x)|00⟩
f(x,w) = ⟨ψ(x,w)|Z0 Z1|ψ(x,w)⟩
```

模擬器直接算這個輸出：

```python
value = cudaq.observe(model, cudaq.spin.z(0) * cudaq.spin.z(1),
                      angles, weights, layers, shots_count=-1).expectation()
```

`shots_count=-1` 表示由完整狀態精算期望值，不做有限次抽樣；仍可能有浮點誤差。

輸出落在 `[-1, 1]`。它是「相同／不同」的加權平均，**不是**現成的分類機率或準確率。若之後要分類，還要定義標籤、如何把分數對應到類別，以及衡量對錯的損失；準確率則是答對比例。

有限次抽樣版本使用獨立的 `sampled_model`：同一套編碼與模板，再加 `mz(q[0])`、`mz(q[1])`。後處理為：

```text
f_sample = (n00 + n11 − n01 − n10) / shots
```

`n00` 是結果 `00` 的次數，其餘同理。注意這和 Day 5、Day 8 的「只看第一個位元 Z0」公式不同。**可觀測量是模型的一部分**；換了讀出規則，就不能沿用舊的計分方式。

延續 Day 8：`model` 本身不寫末端量測。開發時確認此版本的 `observe` 會拒絕含量測分支的 PQC，即使呼叫時把開關設成 false；因此最終用兩個讀出包裝函式，不依賴編譯器是否刪掉某段分支。[D3][D5]

## 5. 初始化，以及可重用的前向介面

訓練開始前，權重要先有一組起始值：

```python
weights = np.random.default_rng(42).normal(0, 0.2, size=4 * layers)
```

`default_rng(42)` 用種子 42 產生可重現的隨機數；`normal(0, 0.2, …)` 從平均 0、標準差 0.2 的鐘形分布抽樣。這是本日固定的小角度起點，不是比較過後的最佳策略。實驗另保留「全零」與種子 43 作對照。

[pqc.py](pqc.py) 對主控端提供：

| 函式 | 用途 |
|---|---|
| `parameter_count(layers)` | 權重該有多長 |
| `initialize(layers, seed)` | 可重現的初始權重 |
| `predict(...)` | CUDA-Q 精算的單一輸出 |
| `reference_prediction(...)` | NumPy 矩陣參考 |
| `sample_prediction(...)` | 有限次量測的估計輸出 |

呼叫端先選執行後端（CPU 或 GPU 模擬器），再評估多筆資料。這個前向介面不會偷偷換後端，也不會改掉你傳入的權重。Day 10 的最佳化器會產生新的候選權重，再呼叫同一介面算輸出。

## 6. 可執行示範

沿用既有 `.venv`；全新環境見 [requirements-day09.txt](../../requirements-day09.txt)。

```bash
source .venv/bin/activate

# 一層、四個 weights；預設 features=[0.25,-0.4]、初始化 seed=42
OMP_NUM_THREADS=1 python articles/day09/demo.py

# 手動指定 weights，確認資料與權重是不同引數
OMP_NUM_THREADS=1 python articles/day09/demo.py --features 0.25 -0.4 --weights 0 0 0 0

# 兩層、八個 weights，改用 GPU 模擬
OMP_NUM_THREADS=1 python articles/day09/demo.py --backend nvidia --layers 2
```

`qpp-cpu` 用 CPU，`nvidia` 用 GPU；`OMP_NUM_THREADS=1` 固定 CPU 執行緒數。完整選項在 [demo.py](demo.py)，會印出電路圖、特徵、編碼角度、權重、精算與 NumPy 參考，以及 1,000 次量測的計數。

預設 CPU 示範大約是：

```text
weights ≈ [0.0609434, -0.2079968, 0.1500902, 0.1881129]
exact ZZ ≈ 0.24571394
NumPy reference ≈ 0.24571394
sampled ZZ = 0.252
counts = {00:568, 01:301, 10:73, 11:58}
```

抽樣與精算之間的差距，是這次有限次量測的波動，**不是**權重被更新了。

## 7. 只改一個旋鈕，輸出會怎麼動？

固定特徵 `[0.25, -0.4]`、一層、種子 42 的初始權重。每次只對一個權重加上 −0.5、0 或 +0.5，其餘不動。CPU 精算約為：

| 權重 | −0.5 | 原始值 | +0.5 |
|---|---:|---:|---:|
| w0 | 0.285360 | 0.245714 | 0.171108 |
| w1 | −0.238628 | 0.245714 | 0.675024 |
| w2 | 0.165546 | 0.245714 | 0.265722 |
| w3 | −0.134586 | 0.245714 | 0.565854 |

四個參數在這個設定下都能拉動輸出。這是有限步長的掃描，**不是**梯度估計，也不能證明任意初始值、資料或深度都容易訓練。**梯度**描述參數微微改變時，輸出往哪邊、改多少；今天還沒算它。特定條件下梯度變得極小、訓練很困難的現象，稱為貧瘠高原（barren plateau），後續章節再談。

RY 角度加 `2π` 可能只多一個整體相位（所有振幅乘上同一個長度為 1 的複數），可觀測量不變；測試會逐個權重檢查這種週期性。因此：**參數數字不同，不一定代表可觀察行為不同。**

## 8. 權重全是零，電路也不一定「什麼都沒做」

單位操作不改變狀態，常記成 `I`。權重全為 0 時，所有 RY 都是單位操作，但 CNOT 仍在：

```text
一層零權重：還有一個 CNOT
兩層零權重：CNOT × CNOT = I
```

搭配本日編碼與 ZZ 讀出，可推得：

| 零權重層數 | 精算 ZZ |
|---|---|
| 奇數 L | `cos(π × x1)` |
| 偶數 L | `cos(π × x0) × cos(π × x1)` |

測試用 L = 1、2、3 核對。這能檢查迴圈有沒有寫錯，也避免把「旋轉角度都是 0」誤解成整條電路沒作用。零初始化是對照案例，不保證每個參數在該點都對輸出敏感。

## 9. 實驗、檔案與測試

```bash
OMP_NUM_THREADS=1 python articles/day09/experiment.py --backend qpp-cpu
OMP_NUM_THREADS=1 python articles/day09/experiment.py --backend nvidia
```

每個後端包含：

- L = 1、2 × 零／種子 42／種子 43 權重 × 4 組特徵，共 24 組前向計算
- 一層模型、4 個權重 × 3 種改動量，共 12 組輸出變化掃描
- 每組前向計算：精算 `observe`、NumPy 參考，以及 1,000 次、種子 42 的抽樣

結果在 `results/day09/<backend>/`：`pqc_sweep.csv`、`raw_results.json`、`summary.json`、`circuit.txt`。重跑會更新同目錄；可用 `--output-dir /tmp/day09-check` 另存。

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day09 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY09_TARGET=nvidia python -m unittest discover -s articles/day09 -p 'test_*.py' -v
```

兩個後端各 7 個測試通過；完整數字見 [結果紀錄](../../results/day09/README.md)。

NumPy 是今日的經典數值核對基準。**保真度**衡量兩個量子狀態有多接近，1 表示相同；本日幾個設定的檢查，不代表所有可能參數都已驗證。計時也不作 GPU 效能宣稱。沒有學習任務，也沒有資料集成效比較。

本日把資料編碼與可訓練權重分開，並以模擬器／NumPy 與單參數掃描核對前向介面。權重尚未訓練，也不宣稱通用表達能力；它不是已收斂的分類器報告。

## 10. 本日來源與下一篇

- [D10] [NVIDIA Building Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html)：`list[float]` 參數化量子核心程式。
- [D3] [NVIDIA Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)：執行與可觀測量。
- [D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：observe、sample 與 State.amplitude。

本日電路模板是教學設計；NumPy 參考重用 Day 3 的 RY 與 Day 4 的 CNOT。查閱日期：2026-09-06，實際版本 CUDA-Q 0.15.1，共用索引見 [REFERENCES.md](../../REFERENCES.md)。

[Day 10](../day10/README.md) 會在這個前向介面外加入損失函數與一般電腦上的最佳化器，第一次讓權重依預測誤差更新。

## 延伸研究

[N3] Seungcheol Oh et al. “Fourier Analysis Perspective on Quantum Neural Networks.” Communications Physics 9, 176 (2026)；觀點論文。[原始來源](https://doi.org/10.1038/s42005-026-02680-x)；[完整書目](../../REFERENCES.md#n3)。

本章區分資料角度與可訓練權重；這篇觀點論文可延伸理解兩者如何共同限制模型能表示的函數形狀。
