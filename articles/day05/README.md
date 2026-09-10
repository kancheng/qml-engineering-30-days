# Day 05｜從一般機器學習流程看懂 QML 流程

[Day4](../day04/README.md) 建立兩量子位元的 貝爾態（Bell state），透過不同量測基底與重複抽樣，理解糾纏、相關性與量測結果的差別。Day5 接著把輸入資料、量子電路與量測結果串起來，建立從一般數值輸入到可評估數值輸出的完整流程。

**一般數值需要先轉成量子電路的操作設定，量測結果也需要整理，才能成為機器學習可用的輸出。** 本章將前幾章的量子概念串成可以逐步核對的流程。本章用一個介於 0 與 1 的數值 `x` 作為輸入，透過 `θ = πx` 轉成 RY 旋轉閘的角度，再接上 CNOT，準備兩量子位元的狀態。量測後，將第一個量子位元的 0／1 結果分別換成 +1／−1，再取平均，得到 `⟨Z0⟩` 的估計值；這個輸出與 Day4 用來描述兩量子位元相關性的數值不同。由於理想答案可直接算成 `cos(πx)`，每個處理階段都能核對，並以 MSE（平均平方誤差）觀察有限 shots 帶來的抽樣誤差。這裡的角度由資料決定，尚未透過 optimizer（最佳化器）學習參數，因此本章完成的是 forward pass（前向計算）。這條流程呈現量子運算與一般程式的分工；後續加入可訓練參數與更新步驟，才會形成訓練流程。

[Day4](../day04/README.md) constructed a two-qubit Bell state and used different measurement bases and repeated sampling to distinguish entanglement, correlation, and measurement outcomes. Day5 connects input data, quantum circuits, and measurement results into a complete workflow from an ordinary numerical input to a numerical output that can be evaluated.

**Ordinary data must be converted into circuit settings, and measurement results must be processed into numerical outputs for machine learning.** This chapter connects the earlier concepts into a workflow that can be checked step by step. A scalar input `x` between 0 and 1 is converted into an RY rotation angle through `θ = πx`. A subsequent CNOT completes the two-qubit state preparation. After measurement, outcomes 0 and 1 for the first qubit are mapped to +1 and −1 and averaged to estimate `⟨Z0⟩`. This output differs from the two-qubit correlation examined in Day4. Since the ideal answer can be calculated directly as `cos(πx)`, each stage can be checked, and mean squared error (MSE) can quantify the sampling error from finite shots. The angle is determined by the input data; no optimizer learns parameters at this stage, so the implementation is a forward pass. The workflow shows how quantum operations and conventional code work together; trainable parameters and update steps are still needed to turn it into a training process.

---

Day 3 實作單量子位元量子閘，Day 4 用 H + CNOT 建立貝爾態。今天將這些元件串成一條完整路徑：**輸入資料 → 編碼 → 電路 → 量測 → 數值輸出 → 評估與保存**。

NumPy 是 Python 的數值運算套件，CUDA-Q 是描述量子電路並安排執行的工具。CPU 是一般電腦的中央處理器，GPU 是擅長平行運算的圖形處理器，RTX 3060 是本次使用的 GPU 型號。本章使用這些工具模擬量子運算，並以互動式筆記本（Notebook）將文字、程式與輸出放在一起。

## 1. 從熟悉的一般機器學習流程開始

監督式學習（supervised learning）使用附有正確答案的資料訓練模型。資料集是整理好的樣本集合；前處理負責清理資料與調整尺度；模型根據輸入產生預測。損失函數（loss function）把預測與答案的差距轉成數值，最佳化器（optimizer）則依據誤差或其變化方向調整參數。

量子機器學習（Quantum Machine Learning，QML）的模型部分可以包含量子電路，但資料整理、損失函數計算、參數更新與結果紀錄通常仍由一般程式負責。今天只先實作前向計算；可訓練電路與最佳化器會在 Day 9–10 加入。

![一般機器學習與量子機器學習流程架構圖](../../figures/day05_pipeline.svg)

| 階段 | 一般機器學習常見操作 | 本日示範 |
|---|---|---|
| 輸入 | 讀取數值特徵 | 9 個位於 `[0, 1]` 的單一數值輸入 |
| 前處理／編碼 | 數值縮放、特徵轉換 | `theta = πx`，以 RY 寫入 q0 |
| 模型／電路 | 參數化函數 | RY(q0) → CNOT(q0, q1) |
| 輸出讀取 | 數值陣列 | Z 基底量測計數 → `⟨Z0⟩` 估計 |
| 評估 | 比較預測與正確答案 | 比較已知函數 `cos(πx)`，計算 MSE |
| 訓練 | 最佳化器更新權重 | 本日沒有參數更新 |
| 紀錄 | 評估指標、設定、版本 | CSV、JSON、執行後端、量測次數、隨機種子 |

前向計算（forward pass）是固定設定下，從輸入算到輸出的一次流程。特徵是提供給模型的資訊；縮放是調整數值範圍，特徵轉換則是改變輸入的表示方式。陣列是按順序或多個維度排列的數值，權重是模型內可調整的係數。評估指標用數值描述結果品質。

這張表是工作流程的對照，不表示量子閘等同任意神經網路層。

## 2. 本日實驗問題與範圍

問題是：「給定單一數值輸入 `x`，能否用同一套介面完成 NumPy 與 CUDA-Q 的前向計算，並驗證有限量測次數的輸出誤差？」

資料由 `np.linspace(0, 1, 9)` 直接產生，目標值是已知的 `cos(πx)`。`np.linspace(0, 1, 9)` 會在 0 到 1 之間取 9 個等距數值。解析答案指能直接用公式計算的答案；`cos` 是餘弦函數。這裡沒有從資料調整參數的模型擬合，也沒有將資料切分成訓練與測試用途，因此不能評估泛化能力，也就是模型在未參與訓練的新資料上表現如何。

這個範圍讓每個階段都能獨立檢查，避免還不理解量測輸出，就開始追蹤複雜的 training 損失函數。

## 3. 環境與交付物

虛擬環境 `.venv` 是專案獨立保存套件的資料夾，避免其他專案的版本互相影響。以下在專案根目錄執行，沿用既有環境：

```bash
source .venv/bin/activate
python -m pip install -r requirements-day05.txt
python -m pip check
```

全新專案副本請先使用 `python3.12 -m venv .venv`。本日保留 NumPy 2.2.6 與 CUDA-Q 0.15.1，增加 Notebook 執行套件；完整依賴版本見 [requirements-day05-lock.txt](../../requirements-day05-lock.txt)。

| 檔案 | 用途 |
|---|---|
| [pipeline.py](pipeline.py) | 資料編碼、NumPy reference、量測計數後處理、評估指標與 CLI |
| [cudaq_pipeline.py](cudaq_pipeline.py) | CUDA-Q 量子核心程式與抽樣 |
| [Quantum Fundamentals Notebook](../../notebooks/day05_quantum_fundamentals.ipynb) | Day 2–5 概念回顧與完整 CPU 示範 |
| [架構圖 SVG](../../figures/day05_pipeline.svg) | 可在文章與 Notebook 顯示的獨立圖檔 |
| [execute_notebook.py](execute_notebook.py) | 使用目前 Python 重跑 Notebook 全部儲存格 |
| [results/day05](../../results/day05/) | 每個執行後端的`predictions.csv` 與 summary.json |

CLI 是命令列介面，表示透過終端機指令選擇設定與執行程式。CSV 是表格文字檔，JSON 以欄位名稱保存結構化資料；SVG 是可縮放而不因放大失去清晰度的向量圖格式。執行後端（backend）指定負責計算的模擬器或硬體：本章的 `numpy`、`qpp-cpu` 使用 CPU，`nvidia` 使用 GPU，三者都是模擬。

`pipeline.py` 直接重用 Day 3 的 RY／I／`|0⟩` 與 Day 4 的 CNOT／基底標籤（向量中各分量對應的位元組合）；未再複製一份量子閘定義。現階段仍以文章目錄為主，尚未將教學模組封裝成安裝套件。

## 4. 第一步：讓一般輸入決定角度

資料編碼（encoding）是將輸入轉成電路可處理的形式。這裡用 `θ = πx` 將數值轉成角度；`θ` 讀作 theta，角度單位是弧度，`π` 代表半圈。本日輸入已在 `[0, 1]`，定義：

```python
def encode(value: float) -> float:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("input must be finite and in [0, 1]")
    return float(np.pi * value)
```

`np.isfinite` 檢查數值不是無限大或無效數值。超出範圍時程式回報錯誤，不會把負數改成 0、把大於 1 的值改成 1；這種改到邊界的操作稱為截斷（clipping）。輸入契約就是程式事先約定的合法輸入條件。縮放器（scaler）則是依資料計算轉換尺度的工具。真正資料集若需要縮放器，只用訓練資料決定尺度，再套用到驗證與測試資料，避免提前使用評估資料的資訊。驗證資料協助選擇設定，測試資料評估選定模型。

`theta = πx` 是 **資料決定的角度**。它雖然是電路程式的輸入引數，卻不是本日學習出來的可訓練參數。之後才會把資料 `x` 和待訓練的權重分開。

## 5. 第二步：RY + CNOT 形成完整電路

q0、q1 是兩個量子位元的編號。RY 是繞布洛赫球 y 軸旋轉的量子閘，用角度改變振幅；CNOT 是受控反相閘，控制位元為 1 時翻轉目標位元。從 `|00⟩` 開始，先對 q0 使用 RY，再用 q0 控制 q1；圖中的 MZ 表示在區分 0 與 1 的 Z 基底下量測：

```text
q0: ──RY(πx)──●──MZ
              │
q1: ──────────X──MZ
```

依 Day 3 的旋轉閘定義與 Day 4 的 CNOT 輸入與輸出對照表：

```text
|ψ(x)⟩ = cos(πx/2)|00⟩ + sin(πx/2)|11⟩
```

三個可立即驗證的輸入：

| x | 輸出狀態 | Z0 精確期望值 |
|---:|---|---:|
| 0 | `|00⟩` | +1 |
| 0.5 | Day 4 的貝爾態 `(|00⟩ + |11⟩)/√2` | 0 |
| 1 | `|11⟩` | −1 |

量子核心程式（quantum kernel）是描述量子操作的程式區塊。本日程式如下，完整版本在 [cudaq_pipeline.py](cudaq_pipeline.py)：

```python
@cudaq.kernel
def encoded_pair(theta: float, measure: bool):
    q = cudaq.qvector(2)
    ry(theta, q[0])
    x.ctrl(q[0], q[1])
    if measure:
        mz(q[0])
        mz(q[1])
```

`measure=False` 僅用於模擬器 `get_state` 正確性測試；正式預測使用 `measure=True` 的量測計數。`get_state` 與 `sample` 分別提供狀態資訊與抽樣結果，兩者不能混為同一種硬體讀出方式。[D3]

## 6. 第三步：把 Counts 轉成數值輸出

可觀測量（observable）是量測所關心的量；期望值是依機率計算的平均值。`⟨Z0⟩` 表示第一個量子位元的 Z 期望值。輸出可觀測量選為 q0 的 Z，將首位 0 對應 +1、首位 1 對應 −1：

```text
prediction = (n00 + n01 − n10 − n11) / shots
```

`n00` 表示結果 `00` 出現的次數，其餘符號同理；`shots` 是總量測次數。位元字串是以 0、1 排列表示的結果，例如 `01`。

例如 `00:493`、`11:507`，估計值為 `−0.014`。這是 Day 4 的量測計數接到一般數值輸出的第一步。

注意不能用 `P(00)+P(11)−P(01)−P(10)` 代替它；那是 Day 4 的兩量子位元關聯值。對本日電路，關聯值始終為 1，無法表達輸入如何改變 `⟨Z0⟩`。

`prediction_from_counts` 會檢查位元字串、非負整數量測計數與非零量測次數，並保留 01／10。即使理想電路只出現 00／11，後處理也不應依賴這個特殊假設。

由狀態推導：

```text
⟨Z0⟩ = cos²(πx/2) − sin²(πx/2) = cos(πx)
```

這是今天用來核對程式的解析答案。CNOT 保留在示範中以銜接 Day 4，但對這個單獨的 Z0 輸出，移除 CNOT 也會得到相同結果。不能把兩量子位元的存在解讀成此任務需要糾纏。

## 7. 完整執行範例

```bash
python articles/day05/pipeline.py --backend numpy
python articles/day05/pipeline.py --backend qpp-cpu
python articles/day05/pipeline.py --backend nvidia
```

隨機種子（seed）是產生隨機序列的起始設定，方便在相同環境重做抽樣。

每個執行後端使用 9 個輸入、3 組 重複實驗的隨機種子（42、43、44）、每筆 量測 1,000 次，產生 27 筆資料。每個輸入透過 `SeedSequence([seed, input_index])` 衍生自己的抽樣隨機種子，實際隨機種子也寫進 CSV。

結果存於 `results/day05/<backend>/`；重跑更新同一目錄，或使用 `--output-dir /tmp/day05-check` 另存。增加量測次數的範例：

```bash
python articles/day05/pipeline.py --backend qpp-cpu --shots 10000 --output-dir /tmp/day05-more-shots
```

CSV 記錄輸入、角度、隨機種子、量測次數、四種量測計數、目標值、預測、平方誤差與時間。JSON 保存環境、資料編碼、電路、可觀測量、雜訊、MSE 與每個 重複實驗的隨機種子的 MSE。

## 8. 結果怎麼判讀？

平均平方誤差（mean squared error，MSE）先將每筆預測減去目標值，將差值平方，再對所有資料取平均；數值越小，代表這批輸出越接近目標。例如誤差為 0.1，該筆平方誤差就是 0.01。本章的誤差來自有限次抽樣，而非訓練不足。

本次預設設定的結果如下，來源為各執行後端的 `summary.json`：

| 方法 | MSE |
|---|---:|
| NumPy 有限次抽樣 | 約 0.00047117 |
| CUDA-Q qpp-cpu 有限次抽樣 | 約 0.00036461 |
| CUDA-Q nvidia 有限次抽樣 | 約 0.00036461 |
| 永遠輸出 0 的簡單對照 | 0.55555556 |
| 一般解析式 `cos(πx)` | 0（依定義） |

解析式就是目標值的生成規則，因此得到零誤差是預期結果。量子優勢需要證明量子方法在指定任務、品質與成本條件下優於適當的一般方法；擊敗固定輸出 0 的方法不足以支持這項結論。本例已有便宜、精確的一般解。

變異數（variance）描述估計值反覆抽樣時的波動大小，定義為偏離其平均值的差距平方再取平均。在每次量測彼此獨立、且相同輸入的機率不變時，±1 量測結果的樣本平均具有下列理論變異數：

```text
Var(prediction) = (1 − cos²(πx)) / shots
```

對本日九個輸入，量測 1,000 次時平均預期平方誤差為 `4/9000 ≈ 0.00044444`。實際 MSE 在此尺度附近波動，有限三組隨機種子 不保證精確等於期望值。

本次 CPU 與 GPU 的量測計數一致，但重現契約是設定與統計行為可追溯，不是所有框架／版本的亂數必須相同。時間包含編譯與初始化，沒有先重複執行以排除首次成本的暖機步驟，也未控制其他程式占用資源的負載或設計效能比較，不能當 GPU 效能評測。

## 9. Notebook 示範與驗證

在程式編輯器開啟 [Notebook](../../notebooks/day05_quantum_fundamentals.ipynb)，選取專案 `.venv/bin/python` 後執行全部儲存格。Notebook 包含已執行輸出，順序涵蓋量子閘、貝爾態、資料編碼、量測計數、NumPy 與 CUDA-Q CPU 流程。

也可不開筆記本介面，直接執行：

```bash
python articles/day05/execute_notebook.py
```

Python 直譯器是執行 Python 程式的程序；Jupyter kernel 則是替筆記本執行程式的背景程序，與量子核心程式是不同概念。儲存格是筆記本中的一段文字或程式。此指令用目前 Python 直譯器啟動暫時的 Jupyter kernel，遇到儲存格執行錯誤即失敗，成功後更新 Notebook 輸出；不註冊全域 kernel。Notebook 的 CSV／JSON 寫入示範使用暫存目錄，不覆蓋正式結果。[D6]

測試指令：

```bash
python -m unittest discover -s articles/day05 -p 'test_*.py' -v
DAY05_TARGET=nvidia python -m unittest discover -s articles/day05 -p 'test_cudaq_pipeline.py' -v
```

測試涵蓋輸入契約、解析振幅、非對稱量測計數的位元順序、隨機種子重現、保存指標與完整 CUDA-Q 流程。執行環境與限制見 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 10. 前五天串起了哪些步驟？

Day 1 確立研究與工程界線；Day 2 把量子位元寫成向量；Day 3 用么正矩陣改變狀態；Day 4 建立兩量子位元系統並理解量測相關性；Day 5 把輸入、電路、量測計數與評估指標接起來。

現在已有完整 前向計算流程，但沒有最佳化器、已訓練模型或真實資料集效能評測。之後的工作就是讓電路擁有獨立的可訓練參數，計算損失函數與梯度，再由一般最佳化器更新參數。梯度描述誤差隨各參數改變的方向與幅度，能協助決定調整方向。么正矩陣則是保持狀態向量長度與內積、可反向還原的矩陣，是前面理想量子閘的限制。

## 11. 本日來源與下一篇

- [D3] [NVIDIA CUDA-Q Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)：`sample` 與 `get_state` 的用途。
- [D6] [nbclient — Executing notebooks](https://nbclient.readthedocs.io/en/latest/client.html)：程式化執行 Notebook 與錯誤處理。
- 本日狀態、期望值與變異數公式由 Day 3–4 的定義推導，並以代碼驗證。文獻索引見 [REFERENCES.md](../../REFERENCES.md)。查閱日期：2026-09-06。

[Day 06](../day06/README.md) 將系統整理 CUDA-Q、CUDA、cuQuantum 的分工，以及已經跑通的 Ubuntu／RTX 3060 環境，附上可重跑的環境檢查與最小量子程式。CUDA 是 NVIDIA GPU 的運算平台，cuQuantum 是協助量子模擬的函式庫，Ubuntu 則是本專案主機使用的 Linux 作業系統。

## 延伸研究

[N1] Shreeya Sanjeev Gokhale et al. “A review of quantum machine learning algorithms, applications, and emerging advantages.” Discover Computing 29, 226 (2026)；綜述論文。[原始來源](https://doi.org/10.1007/s10791-026-10085-1)；[完整書目](../../REFERENCES.md#n1)。

本章建立可重跑的實驗流程；延伸閱讀時，可將研究中的資料處理、電路執行與結果分析分開記錄，建立完整成本觀念。
