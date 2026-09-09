# Day 09｜參數化量子電路：把資料與可訓練參數分開

[Day8](../day08/README.md) 比較 `sample`、`run` 與 `observe` 的輸出形式，釐清量測分布、逐次回傳值與期望值的差別。Day9 接著把量子電路整理成具有明確輸入與權重的模型介面，為後續以最佳化程序更新參數做準備。

**資料決定的角度與模型可調整的權重，雖然都會影響電路，卻有不同用途。** 資料描述這次要處理的樣本，權重則是之後由訓練調整、並共用於不同樣本的設定。本章以兩個數值特徵作為輸入，透過 feature map（資料編碼電路）轉成旋轉角度，再接上 Ansatz（預先選定、含可調權重的電路結構）。每層 Ansatz 包含四個旋轉權重與一個 CNOT；同一組權重可用來處理不同資料，而層數與 shots 則屬於結構或執行設定。讀出選擇 `Z0Z1` 的期望值，也就是把兩個量測位元相同記為 +1、不同記為 −1 後的平均，輸出範圍為 `[-1,1]`，尚未定義成分類機率。實作先核對 CUDA-Q 與 NumPy 的結果，再固定資料、逐一改動權重，觀察輸出反應。參數能影響輸出，只表示電路有可調整的部分；加入衡量預測誤差的損失函數，以及依誤差更新權重的最佳化器，才會開始學習任務。

[Day8](../day08/README.md) compared the outputs of `sample`, `run`, and `observe`, distinguishing measurement distributions, per-execution return values, and expectation values. Day9 organizes the quantum circuit into a model interface with explicit inputs and weights, preparing for parameter updates through optimization.

**Data-dependent angles and adjustable model weights both affect a circuit, but serve different purposes.** Data describes the current example, while weights are shared settings that training will later adjust. Two numerical features enter a feature map—a data-encoding circuit—that converts the inputs into rotation angles. An Ansatz, a chosen circuit structure with adjustable weights, follows the encoding. Each Ansatz block contains four rotation weights and one CNOT. The same weights can process different inputs, while the number of blocks and the shot count are structural or execution settings. The readout is the expectation of `Z0Z1`: the average obtained by assigning +1 to matching measurement bits and −1 to differing bits. Its range is `[-1,1]`, and it has not been defined as a classification probability. The implementation checks CUDA-Q outputs against NumPy references, then holds the data fixed and varies one weight at a time to examine output responses. A response to parameter changes does not establish successful learning. A loss function and an optimizer are still needed to update weights toward a task objective.

---

Day 8 已經能用 `observe` 取得電路的數值輸出。今天要把電路變成一個具有明確參數介面的模型：**輸入資料 x 決定資料編碼電路，權重決定電路模板，可觀測量決定讀出方式。**

本日建立可調參數的前向計算模型，驗證參數如何影響輸出。Day 10 才加入損失函數與最佳化器，因此今天的權重都是指定或初始化的，尚未經過訓練。

## 1. 傳入電路的設定，不一定都要學習

量子核心程式（quantum kernel）是描述量子操作的函式。引數是呼叫函式時傳入的值，例如角度、整數或布林值；布林值只有真與假兩種可能。主控端（host）則是安排呼叫與處理結果的一般 Python 程式。

前幾天已經傳過角度、量子位元數與布林值。這些都是電路輸入引數，但用途不同：

| 項目 | 本日表示 | 是否交給最佳化器更新 |
|---|---|---|
| 輸入資料 | `features=[x0,x1]` | 否 |
| 資料編碼角度 | `angles=π*features` | 否，由資料決定 |
| 模型權重 | `weights` | 預留給 Day 10 更新 |
| 結構設定 | `layers` | 本日固定後執行 |
| 量測次數 | `shots` | 否，是執行設定 |

`list[float]` 表示浮點數清單，浮點數是電腦以有限位數表示的小數。CUDA-Q 可以以 `list[float]` 接收參數化量子核心程式的角度。[D10] 但「有可調角度」與「已經學會某個任務」是兩回事。

## 2. 電路模板是預先選定的電路結構

電路模板（ansatz）是預先安排的操作組合；參數化量子電路（Parameterized Quantum Circuit，PQC）則是包含可調整參數的量子電路。本章的 RY 閘以角度改變振幅，CNOT 是受控反相閘，在控制位元為 1 時翻轉目標位元。q0、q1 是兩個量子位元的編號。

本日固定兩個量子位元，每層使用：

```text
q0: ──RY(w[4l])────●──RY(w[4l+2])──
                   │
q1: ──RY(w[4l+1])───X──RY(w[4l+3])──
```

圖中的 `l` 是從 0 開始的層編號，`w` 是權重清單；`w[4l]` 取出該層第一個權重。偏移量是相對於這個起點的位置。

每層有四個權重，L 層共有 `4L` 個。排列規則固定為：

| 層內偏移量 | 操作 |
|---:|---|
| 0 | CNOT 前的 q0 RY |
| 1 | CNOT 前的 q1 RY |
| 2 | CNOT 後的 q0 RY |
| 3 | CNOT 後的 q1 RY |

電路深度計算必須依序完成的操作層數；可同時執行的操作算在同一層。

兩個不同量子位元上的 RY 可視為同一邏輯操作層，因此每個區塊深度為 3；加上最前面的資料編碼電路，未經編譯最佳化的準備深度為 `1+3L`。編譯最佳化是工具在維持運算結果的前提下調整操作安排，例如相鄰同軸旋轉可能合併，不能把這個教學層數當成編譯後的硬體深度。

這個僅使用 RY 旋轉閘的電路模板方便與 NumPy 矩陣逐步核對。振幅是計算量測機率的係數，絕對值平方才是機率。實數振幅不含虛部，複數振幅則可含虛部。這個電路從實數振幅出發，不能產生任意複數態；本章沒有宣稱這是通用電路或最佳模型架構。

## 3. 資料編碼電路與電路模板的程式邊界

特徵（feature）是描述樣本的數值，例如長度與寬度；資料編碼電路（feature map）將它們轉成量子操作。本章用乘上 `π` 的方式轉成角度，`π` 弧度等於半圈。NumPy 是 Python 的數值運算套件。

完整代碼位於 [pqc.py](pqc.py)。輸入特徵限定兩個有限數值，範圍 `[-1,1]`，主控端轉成弧度：

```python
angles = (np.pi * np.asarray(features)).tolist()
```

`np.asarray` 將輸入轉成數值陣列，`.tolist()` 再轉回 Python 清單。陣列是按順序排列的數值，形狀描述它的維度與大小。有限數值排除無限大與無效數值；截斷是將超出範圍的值改成邊界值。

正式版本另外檢查特徵形狀、權重長度與有限數值；權重可超過 `[-π,π]`，不會靜默截斷。示範層數限為 1–3，這是本日程式的範圍限制。

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

`qview` 提供操作既有量子位元的介面，不另外建立量子位元。`for` 迴圈重複執行每層操作，`range(layers)` 依序提供 0 到 `layers−1`。外層模型配置量子暫存器，也就是一組量子位元，再依序呼叫兩段操作：

```python
@cudaq.kernel
def model(angles: list[float], weights: list[float], layers: int):
    q = cudaq.qvector(2)
    feature_map(q, angles)
    ansatz(q, weights, layers)
```

同一組權重用於不同的輸入資料；資料不是最佳化器每次任意重寫的參數。第一層資料編碼電路 RY 與電路模板 RY 在同一量子位元上相鄰，可合成角度相加；此處分開寫，是為了維持資料與模型權重的語意邊界，不是宣稱兩者形成不可合併的物理操作。

## 4. 輸出讀取：本日選擇 Z0 Z1

可觀測量（observable）指定量測所關心的量。`Z0 Z1` 將兩個位元的 Z 量測結果各記為 +1 或 −1，再相乘：相同結果得到 +1，不同結果得到 −1。期望值是依機率計算的平均值。

下式的 `x` 是資料，`w` 是權重，`U` 表示電路操作；由右往左先編碼，再套用模板。`ψ` 是最後的狀態名稱，`f(x,w)` 是模型輸出：

```text
|ψ(x,w)⟩ = U_ansatz(w) U_encoding(x)|00⟩
f(x,w) = ⟨ψ(x,w)|Z0 Z1|ψ(x,w)⟩
```

模擬器直接計算輸出：

```python
value = cudaq.observe(model, cudaq.spin.z(0) * cudaq.spin.z(1),
                      angles, weights, layers, shots_count=-1).expectation()
```

`shots_count=-1` 在本章模擬器中表示直接由狀態計算期望值，不做有限次抽樣；數值仍可能有浮點誤差。

本日輸出位於 `[-1,1]`，是同位性期望值，也就是上述相同與不同結果的加權平均。它不是現成的分類機率或準確率；若後續需要分類輸出，仍須定義類別標籤（每筆資料的正確類別）、數值轉換規則與損失函數（衡量預測誤差的方法）。準確率則是分類正確的比例。

有限次抽樣版本使用獨立的 `sampled_model`，共用同一份資料編碼電路／電路模板，再加上 `mz(q[0])` 與 `mz(q[1])`。後處理為：

```text
f_sample = (n00 + n11 − n01 − n10) / shots
```

`n00` 表示 `00` 出現的次數，其餘符號同理，`shots` 是總量測次數。

注意這和 Day 5 的 Z0 公式不同。可觀測量是模型的一部分，不能只換電路而沿用不相符的後處理。

延續 Day 8，`model` 本身不含量測。開發時確認此版本的 `observe` 會拒絕含量測分支的 PQC，即使呼叫時把布林值傳成 false；因此最終實作使用兩個輸出讀取 wrappers，不依賴 編譯器是否消除某個分支。[D3][D5]

## 5. 初始化與可重用的前向計算介面

```python
weights = np.random.default_rng(42).normal(0, 0.2, size=4 * layers)
```

初始化是先指定訓練開始前的權重。`default_rng(42)` 建立隨機數產生器，42 是方便重現結果的隨機種子；`normal(0, 0.2, ...)` 從平均值 0、標準差 0.2 的常態分布抽樣。常態分布呈鐘形，標準差描述數值的分散尺度。

這是本日固定隨機種子的小角度初始化，不是經過比較後得出的最佳策略。實驗另外保留零向量與隨機種子 43 作對照。

`pqc.py` 對主控端提供：

| 函式 | 用途 |
|---|---|
| `parameter_count(layers)` | 計算權重長度 |
| `initialize(layers, seed)` | 可重現的初始權重 |
| `predict(features, weights, layers)` | CUDA-Q 直接計算的單一數值輸出 |
| `reference_prediction(...)` | NumPy 矩陣參考 |
| `sample_prediction(...)` | 量測計數、量測次數、隨機種子與抽樣估計的單一數值輸出 |

API 是程式呼叫功能的介面。執行後端（backend）指定實際計算的模擬器或硬體。呼叫端先選執行後端，再評估多組資料；前向計算介面不會在背後切換執行後端，也不修改傳入權重。Day 10 的最佳化器可以產生新的候選權重，再交給這個介面計算輸出。

## 6. 可執行示範

CPU 是中央處理器，GPU 是擅長平行運算的圖形處理器；`qpp-cpu` 使用 CPU 模擬，`nvidia` 使用 GPU 模擬。RTX 3060 是本次的 GPU 型號。`.venv` 是專案獨立保存套件的虛擬環境，`OMP_NUM_THREADS=1` 將 CPU 平行工作的執行緒數設為 1。

沿用既有 `.venv`，沒有新增套件；全新環境可由 [requirements-day09.txt](../../requirements-day09.txt) 安裝固定套件組合。

```bash
source .venv/bin/activate

# 一層、四個 weights，預設 features=[0.25,-0.4]、初始化 seed=42。
OMP_NUM_THREADS=1 python articles/day09/demo.py

# 手動指定 weights，確認資料和 weights 是不同引數。
OMP_NUM_THREADS=1 python articles/day09/demo.py --features 0.25 -0.4 --weights 0 0 0 0

# 兩層、八個 weights，改用 RTX 3060。
OMP_NUM_THREADS=1 python articles/day09/demo.py --backend nvidia --layers 2
```

CLI 是命令列介面，讓終端機選項轉成程式輸入。完整 CLI 在 [demo.py](demo.py)，會印出電路圖、特徵、編碼角度、權重、精確與 NumPy 參考值，以及 1,000 量測次數的量測計數。

預設 CPU 示範：

```text
weights ≈ [0.0609434, -0.2079968, 0.1500902, 0.1881129]
exact ZZ ≈ 0.24571394
NumPy reference ≈ 0.24571394
sampled ZZ = 0.252
counts = {00:568, 01:301, 10:73, 11:58}
```

有限次抽樣結果與精確的差異是本次抽樣的波動，不是權重被更新了。

## 7. 只改一個權重，輸出會怎麼變？

固定特徵、L=1、隨機種子 42 的初始權重，依序對每個權重加上 −0.5、0、+0.5，其餘不動。CPU 的精確結果約為：

| 權重 | −0.5 | 原始值 | +0.5 |
|---|---:|---:|---:|
| w0 | 0.285360 | 0.245714 | 0.171108 |
| w1 | −0.238628 | 0.245714 | 0.675024 |
| w2 | 0.165546 | 0.245714 | 0.265722 |
| w3 | −0.134586 | 0.245714 | 0.565854 |

這說明在選定設定下，四個參數都能影響輸出。這是有限改動的輸出變化掃描，不是梯度估計，也不能證明所有初始值、資料或深度都容易訓練。梯度描述參數微小改變時輸出的變化方向與幅度；此處只測試幾個有限改動，並未計算梯度。貧瘠高原（barren plateau）指特定條件下梯度隨規模快速縮小、使訓練困難的現象，會在後續討論。

RY 的角度增加 `2π` 可帶來整體相位，也就是所有振幅乘上同一個長度為 1 的複數，但可觀測量不變；測試會逐個權重核對輸出週期性。因此，參數向量不同，不一定代表不同的可觀察模型。

## 8. 零權重不代表整個電路模板是單位操作

單位操作不改變狀態，以 `I` 表示。當權重全為 0，RY 都是單位操作，但 CNOT 仍存在：

```text
一層零 weights：仍有一個 CNOT
兩層零 weights：CNOT × CNOT = I
```

對本日資料編碼電路與 ZZ 輸出讀取，可推得：

| 零權重層數 | Exact ZZ |
|---|---|
| 奇數 L | `cos(π*x1)` |
| 偶數 L | `cos(π*x0)*cos(π*x1)` |

測試使用 L=1、2、3 核對這個獨立解析結果。它既能檢查迴圈，也能避免把「所有旋轉角度都是 0」誤解成整個電路沒有作用。零初始化是對照案例，不保證每個參數在該點都對輸出敏感。

## 9. 實驗與結果檔案

```bash
OMP_NUM_THREADS=1 python articles/day09/experiment.py --backend qpp-cpu
OMP_NUM_THREADS=1 python articles/day09/experiment.py --backend nvidia
```

CSV 是表格文字檔，JSON 用欄位名稱保存結構化資料，TXT 保存純文字。

每個執行後端包含：

- L=1、2 × 零／隨機種子 42／隨機種子 43 權重 × 4 組特徵，共 24 組前向計算設定。
- 一層模型的 4 個權重 × 3 種改動量，共 12 組輸出變化設定。
- 每組前向計算設定使用精確 observe、NumPy 狀態／期望值參考值，以及 1,000 量測次數、隨機種子 42 的抽樣。

資料保存在 `results/day09/<backend>/`：`pqc_sweep.csv`、`raw_results.json`、`summary.json` 與 `circuit.txt`。輸出變化的特徵、基準權重、完整候選權重與輸出都寫入 summary。重跑更新相同執行後端，可用 `--output-dir /tmp/day09-check` 另存。

測試指令：

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day09 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY09_TARGET=nvidia python -m unittest discover -s articles/day09 -p 'test_*.py' -v
```

本日兩個執行後端各 7 個測試通過，前向計算與輸出變化也全數通過。完整數字、數值精度與執行限制見 [結果紀錄](../../results/day09/README.md)。

NumPy 是本日的經典數值核對基準；本日沒有學習任務或資料集成效比較。保真度衡量兩個量子狀態的接近程度，1 表示相同物理狀態；數值精度則描述電腦用多少位數表示與計算數值。有限幾個設定的保真度／期望值檢查不代表所有可能參數都已驗證，計時也不作 GPU 效能宣稱。

## 10. 本日來源與下一篇

- [D10] [NVIDIA Building Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html)：`list[float]` 參數化量子核心程式。
- [D3] [NVIDIA Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)：執行與可觀測量。
- [D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：observe、sample 與 State.amplitude。

本日電路模板是教學設計，NumPy 參考值重用 Day 3 的 RY 與 Day 4 的 CNOT。未引用它來宣稱特定研究方法的效果。查閱日期：2026-09-06，實際版本 CUDA-Q 0.15.1，共用索引見 [REFERENCES.md](../../REFERENCES.md)。

[Day 10](../day10/README.md) 會在這個前向計算介面外加入損失函數與一般電腦上的最佳化器，第一次讓權重由最佳化過程更新，完成第二階段的量子與經典混合的參數更新流程。
