# Day 13｜振幅編碼：從正規化向量到狀態準備電路

[Day12](../day12/README.md) 比較角度範圍、旋轉軸與初始態，說明資料轉成旋轉角度後，量子態與量測方式如何影響可區分性。Day13 接著改用振幅編碼，將資料向量轉成量子態的係數，並檢查正規化保留了哪些資訊，以及實際準備這個狀態需要哪些操作。

**把資料寫成合法的量子態，與用電路準備出這個狀態，是兩個不同步驟。** 前者決定狀態的數學表示，後者決定需要執行哪些操作。本章限定兩到四個有限實數，先將向量補零到適合一或兩個量子位元的長度，再除以向量長度，使振幅的絕對值平方總和為 1。例如 `[3,4,0]` 會變成 `[0.6,0.8,0,0]`；其中 0.6、0.8 是振幅，對應的量測機率是 0.36、0.64。重點之一是正規化會移除原始大小，而相對正負號仍可能影響後續量測，因此只核對 Z 基底機率並不足夠。實作分別使用模擬器直接載入狀態，以及明確的 RY／CNOT 電路準備狀態，核對兩條路徑的結果。這些比較將量子位元數、資料前處理與電路準備成本分開記錄。少量量子位元可以表示多個振幅，仍須計入資料載入的工作，也無法在單次量測中讀回完整向量。

[Day12](../day12/README.md) compared angle ranges, rotation axes, and initial states, explaining how encoded states and measurement choices affect the distinguishability of inputs. Day13 turns to amplitude encoding, converting a data vector into quantum-state coefficients and examining both the information retained after normalization and the operations needed to prepare the state.

**Expressing data as a valid quantum state and preparing it with a circuit are separate tasks.** The first defines the mathematical representation; the second determines the required operations. The scope is two to four finite real values. Each vector is padded with zeros to a length suitable for one or two qubits, then divided by its norm so that the squared magnitudes of the amplitudes sum to 1. For example, `[3,4,0]` becomes `[0.6,0.8,0,0]`; the amplitudes 0.6 and 0.8 correspond to measurement probabilities of 0.36 and 0.64. Normalization removes the original magnitude, while relative signs can still affect subsequent measurements, so checking Z-basis probabilities alone is insufficient. The implementation compares direct simulator-state loading with explicit RY/CNOT state-preparation circuits. The comparison separates qubit count from preprocessing and circuit-preparation costs. Representing multiple amplitudes with a small number of qubits still requires data-loading work, and a single measurement cannot recover the complete vector.

---

Day 12 把每個特徵映射成旋轉角度。今天改把資料放進**量子態的振幅**，並實作兩條可核對的路徑：模擬器狀態載入，以及明確的 RY／CNOT 電路。

程式：[amplitude_encoding.py](amplitude_encoding.py)、[demo.py](demo.py)、[experiment.py](experiment.py)。範圍限定 2–4 個有限實數，支援負數與零分支；不支援任意複數或更高維量子閘分解。

## 1. 從特徵向量到正規化狀態

特徵向量是按順序排列的一筆數值資料。振幅編碼（amplitude encoding）將這些數值轉成量子狀態的係數。L2 長度是各分量絕對值平方相加後開平方根，正規化則是將每個分量除以這個長度，使量測機率總和為 1。

給定長度 d 的非零向量，先補零至 `N=2^ceil(log2(d))`，再用 L2 長度正規化：

```text
a = pad(x) / ||pad(x)||₂
|ψ(x)⟩ = Σ_i a_i |i⟩
```

`pad` 表示在尾端補零，`ceil` 表示向上取整數，`log2` 是以 2 為底的對數。因此長度 3 會補到 4，長度 4 不需補零。`Σ` 表示加總，`a_i` 是第 i 個振幅，`|i⟩` 是該位置對應的基底狀態。API 是程式呼叫功能的介面。

這與 NVIDIA Python API 的振幅編碼定義一致。[D5] 本文保留 Day 11 的邏輯基底順序 `|q0 q1⟩`，因此：

```text
[3,4,0] → [3,4,0,0] → [0.6,0.8,0,0]
|ψ⟩ = 0.6|00⟩ + 0.8|01⟩
P(00)=0.36，P(01)=0.64
```

振幅不是機率，要平方絕對值才得到 Z 基底機率。每筆特徵向量各自做正規化，沒有把整個資料集合併成一個狀態，也不使用 Day 11 的逐欄最小值／最大值縮放器。

```python
amplitudes, metadata = normalize([3, 4, 0])
```

NumPy 是 Python 的數值運算套件。附加紀錄（metadata）保存轉換過程的資訊，並非新的振幅。溢位表示數值超過電腦可表示的範圍，下溢則表示數值太小而失去精度，甚至變成 0。

實作先除以最大絕對值，再求向量長度，避免直接平方很大或很小的有限輸入造成溢位／下溢。附加紀錄保存 `norm_scale` 與 `norm_scaled`，原始 L2 長度在數學上是兩者乘積；極端數值下該乘積未必能用浮點數表示，所以不強迫合併成一個浮點數。

全零向量的長度為 0，無法拿來當除數，因此不能正規化。NaN 表示無效數值，Inf 表示無限大；NaN、Inf、錯誤維度與複數輸入也明確拒絕。本日只接受 d=2、3、4，是量子閘實作範圍限制，不是振幅編碼的一般限制。

## 2. 正規化丟掉哪些資訊？

`[3,4]` 和 `[30,40]` 會得到相同狀態；`[-3,-4]` 只差整體相位，也就是所有振幅一起乘上 −1，不影響任何量測預測，因此也代表相同物理態。如果任務依賴向量大小，須在一般電腦的處理流程另外保留向量長度，或重新設計表示方式。

但 `[3,-4]` 改的是**相對符號**：它和 `[3,4]` 的機率相同，狀態卻不同。保真度（fidelity）衡量兩個狀態的重疊程度，本例使用正規化向量內積的絕對值平方，1 表示相同物理狀態。內積是對應分量相乘後加總；一般複數向量需先對第一個向量取共軛，也就是將虛部反號。兩態保真度為：

```text
|0.6² − 0.8²|² = 0.0784
```

期望值是依機率計算的平均值。Z 基底量測區分 0 與 1，X 基底則區分 `( |0⟩ + |1⟩ )/√2` 與 `( |0⟩ − |1⟩ )/√2`，分別記為 +1 與 −1。X 期望值可以看到相對符號的差異：一個量子位元的 `[3,±4]` 得到 ±0.96；補成 `[3,±4,0,0]` 時讀取最後一個邏輯量子位元 X1，也得到 ±0.96。實驗同時保存精確 X、量測計數與保真度，避免只核對機率而漏掉符號錯誤。

## 3. 路徑 A：直接載入模擬器狀態

```python
state = simulator_state(amplitudes)
result = cudaq.get_state(loaded, state)
```

量子核心程式（quantum kernel）描述量子操作，`@cudaq.kernel` 是交給 CUDA-Q 處理的標記。`cudaq.State` 是保存狀態的物件，`qvector(state)` 由它建立對應的量子位元組。底層使用：

```python
@cudaq.kernel
def loaded(state: cudaq.State):
    q = cudaq.qvector(state)
```

`simulator_state` 依後端數值精度建立 complex64 或 complex128 記憶體區塊，兩者分別用 64 或 128 位元儲存一個複數，包含實部與虛部，再呼叫 `cudaq.State.from_data`。這是本地模擬器的狀態初始化路徑，**不提供本例在量子處理器（QPU，實際執行量子操作的硬體）上準備此態所需的量子閘成本**。不能因為 Python 只寫一行就稱它是固定成本的硬體載入。

q0、q1 是量子位元編號。最低有效位元是二進位數中代表個位的位置；CUDA-Q 的狀態記憶體區塊以 q0 為這個位置。[D3] 本文邏輯 `|q0 q1⟩` 順序是 `[00,01,10,11]`，二量子位元載入前要重排為 `[a00,a10,a01,a11]`。取出時使用具名 `State.amplitude(label)`，轉回邏輯順序。測試逐一載入四個基底狀態，專門檢查這個邊界。

## 4. 路徑 B：一個量子位元的 RY

若 d=2，正規化後為 `[a0,a1]`：

```text
θ = 2 atan2(a1,a0)
RY(θ)|0⟩ = a0|0⟩ + a1|1⟩
```

RY 是繞布洛赫球 y 軸旋轉的量子閘，布洛赫球是單一量子位元狀態的幾何表示。`atan2(y,x)` 依平面上的兩個座標求角度，能辨認正負號所在的方向；象限就是平面依兩條座標軸分成的四個區域。`acos` 是反餘弦函數。

`atan2` 保留符號與象限，比單靠 `acos(a0)` 更適合本日含負數的例子。這條路徑需要一個 RY。

## 5. 兩個量子位元：三個 RY 與兩個 CNOT

對 `[a00,a01,a10,a11]`，先計算兩個分支的長度：

```text
r0 = sqrt(a00²+a01²)
r1 = sqrt(a10²+a11²)
α  = 2 atan2(r1,r0)
β0 = 2 atan2(a01,a00)
β1 = 2 atan2(a11,a10)
```

`sqrt` 表示平方根。第一個分支對應 q0=0 的 `00`、`01`，第二個分支對應 q0=1 的 `10`、`11`。`r0`、`r1` 是各分支長度，`α`、`β0`、`β1` 是要用的旋轉角度。

某分支向量長度為 0 時，其 β 設為 0，因為該分支沒有振幅；這不是對所有輸入都忽略符號。

CNOT 是受控反相閘，控制位元為 1 時翻轉目標位元，為 0 時保持不變。

先用 RY(α) 準備 q0 的兩個分支權重，再讓 q1 在 q0=0 時旋轉 β0、q0=1 時旋轉 β1。實際電路分解為：

```text
q0: ──RY(α)──────────────●────────────────────●──
                        │                    │
q1: ──RY((β0+β1)/2)──────X──RY((β0−β1)/2)─────X──
```

q0=0 時 q1 兩次旋轉相加為 β0；q0=1 時兩次 X 使其中一次旋轉角反號，結果為 β1。這給出一個可逐步檢查的含正負號實數的狀態準備。

```python
ry(angles[0], q[0])
ry(angles[1], q[1])
x.ctrl(q[0], q[1])
ry(angles[2], q[1])
x.ctrl(q[0], q[1])
```

沒有呼叫通用狀態準備函式庫。此範例可產生 `[1,0,0,1]/sqrt(2)` 的貝爾態，貝爾態是不能拆成兩個獨立單位元純態的糾纏態，所以與 Day 12 的純單量子位元旋轉乘積態編碼不同。

## 6. 成本要分層記錄

| 本例維度 | 補零後振幅 | 量子位元數 | 明確狀態準備量子閘 |
|---:|---:|---:|---|
| 2 | 2 | 1 | 1 RY |
| 3 | 4 | 2 | 3 RY＋2 CNOT |
| 4 | 4 | 2 | 3 RY＋2 CNOT |

量測基底轉換是量測前加入操作，以改變區分狀態的方式；路由是配合硬體連接性安排額外操作，原生量子閘是裝置直接支援的基本操作。編譯器會將程式轉成可執行形式，也可能簡化操作。

這是原始電路計數，沒有量測基底轉換、路由、硬體原生量子閘分解。特殊輸入的旋轉操作或 CNOT 可能被編譯器消除，不能把表格當成所有編譯輸出的精確成本。

主控端（host）是安排工作的普通 Python 程式。主控端必須讀取整個特徵向量、驗證、補零與正規化，這些陣列操作隨資料長度增加；量子位元數的對數成長不代表整個載入流程也只有對數成本。NVIDIA 的狀態準備文件也區分狀態向量與將其編譯成電路的工作。[D13] 本文只實作到兩個量子位元，不用這個小例子推估一般高維或特殊結構資料的量子閘數。

硬體每次取樣都需要準備狀態再量測；模擬器則可從已算出的分布抽樣。一次 Z 基底結果只是一個位元字串，無法讀回所有振幅與符號。GPU 計算環境也不改變這個量測限制。

## 7. 可執行範例與實驗

CPU 是一般電腦的中央處理器，GPU 是擅長平行運算的圖形處理器。執行後端（backend）指定使用的模擬器，本章預設使用 CPU，`nvidia` 使用 GPU。`.venv` 是專案獨立保存套件的虛擬環境；`OMP_NUM_THREADS=1` 將 CPU 平行工作的執行緒數設為 1。

沿用 `.venv`，沒有新增套件；版本鏈入口：[requirements-day13.txt](../../requirements-day13.txt)。在專案根目錄執行：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day13/demo.py
OMP_NUM_THREADS=1 python articles/day13/demo.py --values 1 -2 3 -4
OMP_NUM_THREADS=1 python articles/day13/demo.py --values 0 0 -3 4 --backend nvidia

OMP_NUM_THREADS=1 python articles/day13/experiment.py
OMP_NUM_THREADS=1 python articles/day13/experiment.py --backend nvidia
```

示範印出正規化振幅、角度、明確電路圖、機率與保真度。實驗每個執行後端有 12 組案例，包含 2／3／4 維、尺度倍數、整體／相對正負號、不同基底、零分支及貝爾態。

量測計數（counts）記錄每種結果出現幾次，shots 表示量測次數。精確期望值由模擬器保存的狀態直接計算，不做有限次抽樣，但仍可能有浮點誤差。JSON 是以欄位名稱保存資料的文字格式，TXT 保存純文字。

每組以直接載入和明確電路兩條路徑核對目標狀態，另對量子閘計算精確 X 與 1,000 量測次數的 Z 量測計數。結果保存到 `results/day13/<backend>/`：`predictions.json`、`summary.json`、`circuit.txt`。重跑更新該目錄，可用 `--output-dir /tmp/day13-check` 另存。

## 8. 驗證與下一篇

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day13 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY13_TARGET=nvidia python -m unittest discover -s articles/day13 -p 'test_*.py' -v
```

隨機種子（seed）設定隨機序列的起點，方便在相同環境重做實驗。

六個測試涵蓋正規化與輸入不變性、錯誤與極端有限值、尺度／整體相位、相對正負號、四個基底的載入順序、固定隨機種子隨機含正負號的向量與零分支。完整數字見 [實驗結果](../../results/day13/README.md)。

無噪聲表示未加入使操作偏離理想情況的干擾。準確率是分類正確的比例。

本日為無噪聲模擬器驗證，沒有 QPU 實測、分類準確率、載入速度比較或量子優勢宣稱。[Day 14](../day14/README.md) 將回到資料編碼電路與可調電路模板，整理資料表示與可訓練電路的組合。

## 9. 來源

- [D5] [NVIDIA Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：振幅編碼定義、State.from_data。
- [D3] [NVIDIA Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)：狀態向量的 little-endian 順序。
- [D13] [NVIDIA Approximate State Preparation using MPS Sequential Encoding](https://nvidia.github.io/cuda-quantum/0.13.0/applications/python/mps_encoding.html)：狀態準備是將狀態向量轉成電路；MPS 是將完整狀態拆成一串相連矩陣的表示方式，本文未實作此方法。

查閱日期 2026-09-07，實際 CUDA-Q 0.15.1。小型 RY／CNOT 分解由本文公式與測試核對，共用索引見 [REFERENCES.md](../../REFERENCES.md)。

## 延伸研究

[N6] Kevin W. Aoun et al. “Quantum State Preparation via Neural Network Encoding in Quantum Machine Learning.” arXiv:2605.31006v1 (2026)；預印本。[原始來源](https://arxiv.org/abs/2605.31006v1)；[完整書目](../../REFERENCES.md#n6)。

本章區分振幅表示與實際狀態準備；這篇研究提供用神經網路產生電路參數的案例，方便比較載入狀態與建立準備電路的成本。
