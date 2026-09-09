# Day 04｜疊加、糾纏與量測：建立第一個貝爾態

[Day3](../day03/README.md) 用 NumPy 矩陣運算，觀察單一量子位元經過量子閘後，振幅、相位與量測機率如何改變。Day4 將範圍擴展到兩個量子位元，建立 Bell state（貝爾態），並以 CUDA-Q 實作電路與重複量測。

**兩個量子位元可以形成無法各自獨立描述的狀態，但量測結果相關，還不足以證明糾纏。** 分清楚這兩件事，才能正確解讀後續量子機器學習（Quantum Machine Learning，QML）電路的輸出。本章先用 tensor product（張量積）組合兩個單量子位元狀態，再從 `|00⟩` 出發，依序施加 Hadamard 與 CNOT，建立無法拆成兩個單量子位元純態乘積的 Bell state。在理想的 Z 基底量測下，每次只會得到 `00` 或 `11`；然而，以公平硬幣決定準備 `|00⟩` 或 `|11⟩` 的經典混合，也能產生相同統計。因此，本章另外比較 X 基底量測，觀察這兩種已知準備的差異，釐清疊加、糾纏與一般相關性的區別。實作先以 NumPy 核對理論分布，再用 CUDA-Q 的 CPU／GPU 模擬取得 counts（各結果出現次數），並比較不同 shots（重複抽樣次數）帶來的波動。量測方式與抽樣次數都會影響結果判讀，而這些程式是在一般電腦上模擬量子系統，並未使用真實量子硬體。

[Day3](../day03/README.md) used NumPy matrix operations to examine how single-qubit gates change amplitudes, phases, and measurement probabilities. Day4 extends the scope to two qubits, constructs a Bell state, and implements circuits and repeated measurements with CUDA-Q.

**Two qubits can form a state that cannot be described as two independent pure states, but correlated measurement outcomes alone do not establish entanglement.** This distinction matters when interpreting quantum machine learning (QML) circuits. The chapter first uses the tensor product to combine single-qubit states. Starting from `|00⟩`, a Hadamard gate followed by a CNOT creates a Bell state that cannot be factored into a product of two single-qubit pure states. Ideal Z-basis measurements produce only `00` or `11`. However, a classical mixture prepared by using a fair coin to choose between `|00⟩` and `|11⟩` produces the same statistics. Comparing X-basis measurements distinguishes these two known preparations and clarifies the differences between superposition, entanglement, and ordinary correlation. The implementation uses NumPy to verify theoretical distributions, then CUDA-Q CPU/GPU simulation to obtain counts and examine sampling fluctuations at different shot counts. The measurement basis and number of samples affect interpretation. These programs simulate quantum systems on conventional computers; they do not run on physical quantum hardware.

---

Day 3 用矩陣改變一個量子位元；今天把狀態擴展成兩個量子位元，先用 NumPy 看清楚每一步，再以 CUDA-Q 在 CPU 與 RTX 3060 上執行相同概念。

今天的核心問題是：**兩個量子位元各自都像隨機結果，為什麼合在一起卻有固定的關係？**

## 1. 程式與實驗內容

- [bell_state.py](bell_state.py)：張量積、CNOT、貝爾態與經典混合態的 NumPy 參考實作。
- [cudaq_bell.py](cudaq_bell.py)：CUDA-Q 電路、Z／X 基底量測、CPU／GPU 示範。
- [experiment.py](experiment.py)：3 種狀態 × 2 種基底 × 3 種量測次數 × 5 個隨機種子，共 90 筆／執行後端。
- [test_bell_state.py](test_bell_state.py)、[test_cudaq_bell.py](test_cudaq_bell.py)：理論性質與框架整合測試。
- [results/day04](../../results/day04/)：分執行後端保存 CSV 與 JSON。

NumPy 是 Python 的數值運算套件；CUDA-Q 是描述量子電路並安排執行的工具。CPU 是一般電腦的中央處理器，GPU 是擅長平行運算的圖形處理器，RTX 3060 是本次使用的 GPU 型號。兩者都屬於經典硬體，也就是一般電腦的運算設備。執行後端（backend）指定實際使用的模擬器或硬體；本章的三個後端都是模擬。CSV 是表格文字檔，JSON 則以欄位名稱保存結構化資料。

這是量子基礎與模擬器驗證實驗。經典混合態是概念對照，NumPy 是數值參考；今天尚未建立 機器學習模型或比較分類效能。

## 2. 延續專案獨立環境

虛擬環境 `.venv` 是專案獨立保存 Python 套件的資料夾，避免與其他專案的版本互相影響。CUDA 是 NVIDIA GPU 的運算平台，cuQuantum 是協助量子模擬的運算函式庫；兩者與 CUDA-Q 的分工會在 Day 06 展開。

沿用 Day 3 的 Python 3.12 `.venv`，增加 CUDA 12 版 CUDA-Q 0.15.1，保留 NumPy 2.2.6。在專案根目錄執行：

```bash
source .venv/bin/activate
python -m pip install -r requirements-day04.txt
python -m pip check
```

若是全新專案副本，先執行 `python3.12 -m venv .venv`。實際驗證使用的完整依賴清單見 [requirements-day04-lock.txt](../../requirements-day04-lock.txt)，需要相同依賴版本時改用此檔安裝。

本專案明確選用 `cuda-quantum-cu12` 發行套件，不另外疊裝 `cudaq` 或其他 CUDA-Q 二進位發行套件。官方提醒不同發行套件可能衝突。[D1]

Day 4 先完成必要的安裝與貝爾態驗證；Day 6 再系統整理 CUDA-Q、CUDA、cuQuantum 與環境設定。這樣 Day 4 的程式可立即執行。

## 3. 兩個量子位元：用張量積組合狀態

本日 NumPy 向量順序固定為 `|q0 q1⟩`，也就是 `[00, 01, 10, 11]`：

```text
|00⟩ = |0⟩ ⊗ |0⟩ = [1, 0, 0, 0]ᵀ
|01⟩ = |0⟩ ⊗ |1⟩ = [0, 1, 0, 0]ᵀ
|10⟩ = |1⟩ ⊗ |0⟩ = [0, 0, 1, 0]ᵀ
|11⟩ = |1⟩ ⊗ |1⟩ = [0, 0, 0, 1]ᵀ
```

`⊗` 是張量積（tensor product），將兩個向量的分量逐一相乘，組合成更大的向量。例如 `[a, b]ᵀ ⊗ [c, d]ᵀ = [ac, ad, bc, bd]ᵀ`；`ᵀ` 表示把橫向排列轉成直向排列。在 NumPy 使用 `np.kron` 計算。`q0`、`q1` 是兩個量子位元的編號；`I` 是不改變狀態的單位矩陣。H 是 Hadamard 閘，會將 `|0⟩` 轉成等振幅疊加態 `|+⟩ = (|0⟩ + |1⟩)/√2`。若只對 q0 施加 H，整個系統的操作為 `H ⊗ I`：

```text
(H ⊗ I)|00⟩ = (|00⟩ + |10⟩)/√2 = |+⟩ ⊗ |0⟩
```

這個狀態已經有疊加，但仍能拆成兩個單量子位元狀態的乘積。對兩個量子位元都施加 H，得到的 `|+⟩ ⊗ |+⟩` 也仍是乘積態。

## 4. CNOT：讓 q0 控制 q1 的翻轉

CNOT 是受控反相閘（controlled-NOT）：控制位元為 0 時保持目標位元，為 1 時翻轉目標位元。本章使用 q0 為控制位元、q1 為目標位元：

| 輸入 | 輸出 |
|---|---|
| `00` | `00` |
| `01` | `01` |
| `10` | `11` |
| `11` | `10` |

對應矩陣為：

```text
       [1 0 0 0]
CNOT = [0 1 0 0]
       [0 0 0 1]
       [0 0 1 0]
```

CNOT 不會量測控制位元，它對整個疊加態做線性、么正轉換：線性表示各項振幅依同一規則運算後再相加，么正則表示保留向量長度與內積，且能反向還原。把它接在 `H ⊗ I` 後面：

```text
|00⟩ → (|00⟩ + |10⟩)/√2 → (|00⟩ + |11⟩)/√2 = |Φ+⟩
```

`|Φ+⟩` 是這個狀態的名稱，讀作 phi plus，屬於貝爾態（Bell state）的一種。糾纏（entanglement）在這個純態例子中，表示整體狀態無法拆成兩個單量子位元純態的乘積。CNOT 是否產生糾纏取決於輸入；例如 `CNOT|00⟩` 仍是 `|00⟩`。

## 5. NumPy 範例：計算貝爾態的振幅

以下是 [bell_state.py](bell_state.py) 的核心計算：

```python
import numpy as np

zero = np.array([1, 0], dtype=np.complex128)
h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                 [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
bell = cnot @ np.kron(h, np.eye(2)) @ np.kron(zero, zero)
print(bell)
print(np.abs(bell) ** 2)
```

預期振幅約為 `[0.7071, 0, 0, 0.7071]`，Z 基底機率為 `[0.5, 0, 0, 0.5]`。完整示範：

```bash
python articles/day04/bell_state.py
```

純態是能用單一狀態向量完整描述的狀態。若能拆成 `|a⟩ ⊗ |b⟩`，就稱為乘積態（product state）；本章的貝爾態不能這樣拆開。

程式將四個振幅排成 2 × 2 矩陣，用秩（rank）檢查這件事。秩可以理解成矩陣中彼此獨立的列或欄有多少個；兩個向量的分量逐一相乘形成的外積矩陣，非零時秩為 1，貝爾態對應矩陣的秩則為 2。

測試也檢查約化狀態（reduced state），也就是只觀察其中一個量子位元時所需的狀態描述。貝爾態的單一位元約化狀態是 `I/2`，表示單獨看這個位元是完全混合的；完整的關係仍存在於兩個位元的聯合狀態。混合態是以機率準備不同狀態的情況，不能直接套用上述純態的秩檢查。[F2]

## 6. CUDA-Q 範例：用程式描述量子電路

量子核心程式（quantum kernel）是描述量子操作的程式區塊。以下範例用 `qvector(2)` 建立兩個量子位元，`h` 施加 H 閘，`x.ctrl` 施加 CNOT，`mz` 執行 Z 基底量測；Z 基底就是區分 `|0⟩` 與 `|1⟩` 的量測方式。最小貝爾態程式如下，對應 NVIDIA 官方入門範例的 H + 受控 X 閘 結構。[D1]

```python
import cudaq

cudaq.set_target("qpp-cpu")

@cudaq.kernel
def bell_measure():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q[0])
    mz(q[1])

cudaq.set_random_seed(42)
print(cudaq.sample(bell_measure, shots_count=1000,
                   explicit_measurements=True))
```

專案的 [cudaq_bell.py](cudaq_bell.py) 另支援乘積態、混合態準備與 X 基底。直接執行：

```bash
python articles/day04/cudaq_bell.py --backend qpp-cpu
python articles/day04/cudaq_bell.py --backend nvidia
```

本次兩個執行後端的 貝爾態的 Z 基底 示範都得到 `00: 493`、`11: 507`，`01`／`10` 為 0。這是指定環境、隨機種子的結果，不保證不同版本與執行後端都產生相同量測計數。

```text
q0: ──H──●──MZ
         │
q1: ─────X──MZ
```

`h`、`x.ctrl`、`mz` 是量子核心程式內由 CUDA-Q 編譯器識別的操作；編譯器會將程式描述轉成可執行的運算，這些操作不能當成普通 Python 函式在該區塊外呼叫。本日採 `explicit_measurements=True`，按 q0、q1 的量測次序串接輸出，並以只翻轉 q0 的非對稱案例測試得到 `10`。[D5]

## 7. 量測與抽樣次數：一個結果不是機率分布

貝爾態在 Z 基底下每次得到 `00` 或 `11`。單看任一量子位元，0／1 各半；合看兩個結果，則永遠相同。條件在 q0 已量到 0 時，q1 為 0；但無法選擇 q0 的隨機結果來傳送訊息。

真實硬體必須重新準備並量測以累積量測次數；模擬器可以先建立狀態，再從其分布抽樣，不必真的做相同次數的完整狀態演化。[D3]

隨機種子（seed）是產生隨機序列的起始設定，方便在相同環境重做抽樣。標準誤（standard error）描述反覆做同樣實驗時，估計值的典型波動尺度，並不是每次誤差的上限。`N` 是抽樣次數，`sqrt` 表示平方根。估計某個結果的機率 `p`，在每次抽樣彼此獨立且機率相同的條件下，標準誤為 `sqrt(p(1-p)/N)`。對 `p=0.5`：

| 量測次數 N | 機率估計的理論標準誤 |
|---:|---:|
| 100 | 0.0500 |
| 1,000 | 約 0.0158 |
| 10,000 | 0.0050 |

量測次數增加通常能縮小抽樣波動，但某一個隨機種子的誤差不保證單調下降。因此本日保留五組隨機種子，而不只挑最好看的結果。

## 8. 為什麼只看到 00／11，還不能認定糾纏？

想像每次先丟一枚經典公平硬幣，選擇準備 `|00⟩` 或 `|11⟩`。它也會產生相同的 Z 基底統計，但這是可分離混合態：每次準備的都是乘積態，再用一般機率將這些準備方式混合，並未形成糾纏。[F2]

密度矩陣（density matrix）可以同時描述純態與機率混合，用希臘字母 `ρ`（rho）表示。純態使用 `ρ = |ψ⟩⟨ψ|`；經典混合則把每種準備的密度矩陣依機率加權。

```text
ρ_bell = 1/2 [1 0 0 1]       ρ_mix = 1/2 [1 0 0 0]
             [0 0 0 0]                   [0 0 0 0]
             [0 0 0 0]                   [0 0 0 0]
             [1 0 0 1]                   [0 0 0 1]
```

兩者對角線相同，Z 基底機率自然相同；貝爾態還有連結 `00` 與 `11` 的非對角元素所描述的相干性（coherence），也就是不同基底分量之間保留了能在後續操作中產生干涉的相位關係。對角線指矩陣左上到右下的位置；非對角元素則在其餘位置。

把兩個量子位元都先施加 H，再做 Z 量測，就相當於量 X 基底。下面的分布可由本日矩陣實作直接計算，順序皆為 `[00, 01, 10, 11]`：

| 狀態 | Z 基底分布 | X 基底分布 |
|---|---|---|
| 乘積態 `|++⟩` | `[0.25, 0.25, 0.25, 0.25]` | `[1, 0, 0, 0]` |
| 貝爾態 `|Φ+⟩` | `[0.5, 0, 0, 0.5]` | `[0.5, 0, 0, 0.5]` |
| 經典混合態 | `[0.5, 0, 0, 0.5]` | `[0.25, 0.25, 0.25, 0.25]` |

X 基底的 0 代表 `+`，1 代表 `−`。這組比較區分了本日三個已知準備；貝爾不等式是檢查特定經典局域模型對量測關聯所設限制的工具；不依賴裝置內部模型的驗證還需要額外的量測設定與條件，本章未進行這類實驗。量子優勢則需要與適當的一般計算方法比較任務品質與成本，本章也未做這項比較。

CUDA-Q 版混合態用 NumPy 的二項分布抽樣（二項分布描述固定次數獨立擲硬幣時，某一面出現的總次數）決定本批量測次數中準備 `00`／`11` 的數量，再分批交給 CUDA-Q 量測。這等價於每次抽樣 獨立丟硬幣的聚合量測計數，並未把混合態假裝成 貝爾純態。

## 9. 實驗設計與執行

| 欄位 | 設定 |
|---|---|
| 量子位元數 | 2 |
| 狀態 | 乘積態、貝爾態、混合態 |
| 貝爾態準備 | H(q0) → CNOT(q0, q1)，準備深度 2 |
| 量測 | Z、X；X 額外一層 H⊗H |
| 量測次數 | 100、1,000、10,000 |
| 隨機種子 | 42–46 |
| 雜訊 | 無 |
| 執行後端 | numpy、qpp-cpu、nvidia |
| 評估指標 | 量測計數、相同結果比例、關聯值、總變異距離 |

```bash
python articles/day04/experiment.py --backend numpy
python articles/day04/experiment.py --backend qpp-cpu
python articles/day04/experiment.py --backend nvidia
```

每個執行後端獨立寫入 `results/day04/<backend>/`，不會互相覆蓋。重跑同一執行後端則更新其結果；可用 `--output-dir /tmp/day04-check` 指定另存位置。

`correlation = P(00) + P(11) − P(01) − P(10)`，對應本次基底的兩個 ±1 量測結果乘積平均；`TVD = 1/2 Σ|p_sample − p_exact|` 衡量整體分布誤差。總變異距離（total variation distance，TVD）將每個結果的機率差取絕對值後加總，再除以 2；0 表示兩個分布相同。公式中的 `Σ` 表示加總，`p_sample` 是抽樣比例，`p_exact` 是理論機率。關聯值是相同結果比例減去不同結果比例，範圍為 −1 到 1。樣本標準差描述五次結果彼此的分散程度。JSON 保存各設定五次執行的平均 TVD、樣本標準差與平均關聯值。`p_equal` 是相同結果的比例，不是糾纏程度的分數。

電路深度是考慮可同時執行的操作後，仍需依序完成的層數。執行時間包含首次即時編譯（JIT，執行時將程式轉成可用的運算形式），以及執行環境的初始化成本。只有兩個量子位元，資料不能用來判定 GPU 加速優勢。

## 10. 驗證與結果紀錄

```bash
python -m unittest discover -s articles/day04 -p 'test_*.py' -v
DAY04_TARGET=nvidia python -m unittest discover -s articles/day04 -p 'test_cudaq_bell.py' -v
```

第一個指令包含 NumPy 與 CUDA-Q CPU 測試；第二個明確切換 GPU。若只有 Day 3 的 NumPy 環境，可先執行 `-p 'test_bell_state.py'` 與 NumPy 實驗。

測試核對 CNOT 輸入與輸出對照表、貝爾態振幅、約化狀態、密度矩陣合法性、Z／X 分布、量測次數總數、固定隨機種子重跑與量測位元順序。CUDA-Q 的 `get_state` 僅用於模擬器的數值驗證，不代表量子處理器（QPU，實際執行量子操作的硬體）能直接讀出完整狀態向量。[D3]

本次環境與結果摘要見 [ENVIRONMENT.md](ENVIRONMENT.md)；完整數據見 [results/day04](../../results/day04/)。

## 11. 今天容易混淆的地方

- 疊加不必然是糾纏，例如 `|++⟩`。
- CNOT 不是量測控制位元後執行一般 Python `if`。
- `00`／`11` 相關性也可能由經典混合產生。
- 單一量子位元的 50%／50% 分布不能描述完整兩量子位元狀態。
- 不同框架可能有不同的向量與位元順序，貝爾態本身對稱，必須加非對稱案例驗證。
- 隨機種子用來追溯重跑，不是要求不同抽樣器輸出逐筆相同。
- GPU 量子模擬仍是經典硬體上的模擬。

## 12. 本日來源

- [D1] [NVIDIA CUDA-Q Quick Start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)：安裝與 H + controlled-X Bell 範例。
- [D3] [NVIDIA Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)：`sample`、`get_state` 與模擬器／硬體執行差異。
- [D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：`explicit_measurements` 的量測順序語意。
- [F2] [IBM Quantum Learning — Multiple systems and reduced states](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/multiple-systems)：可分離混合態、糾纏與約化狀態。

查閱日期：2026-09-06。官方教學用於基礎概念與程式介面（API）；表格的數值由本日代碼與測試交叉驗證。共用索引見 [REFERENCES.md](../../REFERENCES.md)。

## 13. 下一篇

[Day 05](../day05/README.md) 會把狀態準備、量子閘、量測與一般電腦上的結果處理串成完整流程，整理可將文字、程式與輸出放在一起的互動式筆記本，以及流程架構圖，完成第一個量子基礎實作。
