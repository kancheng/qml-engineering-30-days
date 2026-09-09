# Day 13｜Amplitude Encoding：正規化不是免費的 State Preparation

## 本章摘要｜初學者學習筆記

### 中文

[Day12](../day12/README.md) 比較角度範圍、旋轉軸與初始態，說明資料轉成旋轉角度後，量子態與量測方式如何影響可區分性。Day13 接著改用振幅編碼，將資料向量轉成量子態的係數，並檢查正規化保留了哪些資訊，以及實際準備這個狀態需要哪些操作。

這一章目標在於理解 **「把資料寫成合法的量子態，與用量子電路準備出這個狀態，是兩個需要分開處理的步驟」**，為評估 QML 資料載入方式建立基礎。本章限定兩到四個有限實數，先將向量補零到適合一或兩個量子位元的長度，再除以向量長度，使振幅的絕對值平方總和為 1。例如 `[3,4,0]` 會變成 `[0.6,0.8,0,0]`；其中 0.6、0.8 是振幅，對應的量測機率是 0.36、0.64。重點之一是正規化會移除原始大小，而相對正負號仍可能影響後續量測，因此只核對 Z 基底機率並不足夠。實作分別使用模擬器直接載入狀態，以及明確的 RY／CNOT 電路準備狀態，核對兩條路徑的結果。讀完本章，應能說明振幅編碼與角度編碼的差別，理解全零向量為何不能直接正規化，並區分量子位元數、前處理成本與狀態準備成本。少量量子位元可以表示多個振幅，仍須計入資料載入的工作，也無法在單次量測中讀回完整向量。

### English

[Day12](../day12/README.md) compared angle ranges, rotation axes, and initial states, explaining how encoded states and measurement choices affect the distinguishability of inputs. Day13 turns to amplitude encoding, converting a data vector into quantum-state coefficients and examining both the information retained after normalization and the operations needed to prepare the state.

This chapter aims to explain **why expressing data as a valid quantum state and preparing that state with a quantum circuit are two separate tasks**, providing a foundation for evaluating QML data-loading methods. The scope is two to four finite real values. Each vector is padded with zeros to a length suitable for one or two qubits, then divided by its norm so that the squared magnitudes of the amplitudes sum to 1. For example, `[3,4,0]` becomes `[0.6,0.8,0,0]`; the amplitudes 0.6 and 0.8 correspond to measurement probabilities of 0.36 and 0.64. Normalization removes the original magnitude, while relative signs can still affect subsequent measurements, so checking Z-basis probabilities alone is insufficient. The implementation compares direct simulator-state loading with explicit RY/CNOT state-preparation circuits. The learning goal is to distinguish amplitude encoding from angle encoding, explain why a zero vector cannot be normalized directly, and separate qubit count from preprocessing and state-preparation costs. Representing multiple amplitudes with a small number of qubits still requires data-loading work, and a single measurement cannot recover the complete vector.

---

Day 12 把每個 feature 映射成 rotation angle。今天改把資料放進**量子態的 amplitudes**，並實作兩條可核對的路徑：simulator state 載入，以及明確的 RY／CNOT 電路。

程式：[amplitude_encoding.py](amplitude_encoding.py)、[demo.py](demo.py)、[experiment.py](experiment.py)。範圍限定 2–4 個有限實數，支援負數與零分支；不支援任意複數或更高維 gate synthesis。

## 1. 從 Feature Vector 到 Normalized State

給定長度 d 的非零向量，先補零至 `N=2^ceil(log2(d))`，再用 L2 norm 正規化：

```text
a = pad(x) / ||pad(x)||₂
|ψ(x)⟩ = Σ_i a_i |i⟩
```

這與 NVIDIA Python API 的 amplitude encoding 定義一致。[D5] 本文保留 Day 11 的 logical basis 順序 `|q0 q1⟩`，因此：

```text
[3,4,0] → [3,4,0,0] → [0.6,0.8,0,0]
|ψ⟩ = 0.6|00⟩ + 0.8|01⟩
P(00)=0.36，P(01)=0.64
```

amplitude 不是 probability，要平方絕對值才得到 Z 基底機率。每筆 feature vector 各自做 normalization，沒有把整個 dataset 合併成一個 state，也不使用 Day 11 的逐欄 min/max scaler。

```python
amplitudes, metadata = normalize([3, 4, 0])
```

實作先除以最大絕對值，再求 norm，避免直接平方很大或很小的有限輸入造成 overflow／underflow。metadata 保存 `norm_scale` 與 `norm_scaled`，原始 L2 norm 在數學上是兩者乘積；極端數值下該乘積未必能用 float 表示，所以不強迫合併成一個浮點數。

全零向量無法正規化；NaN、Inf、錯誤維度與複數輸入也明確拒絕。本日只接受 d=2、3、4，是 gate 實作範圍限制，不是 amplitude encoding 的一般限制。

## 2. 正規化丟掉哪些資訊？

`[3,4]` 和 `[30,40]` 會得到相同 state；`[-3,-4]` 只差 global phase，因此也代表相同物理態。如果任務依賴向量大小，須在 classical pipeline 另外保留 norm，或重新設計表示方式。

但 `[3,-4]` 改的是**相對符號**：它和 `[3,4]` 的 probabilities 相同，state 卻不同。兩態 fidelity 為：

```text
|0.6² − 0.8²|² = 0.0784
```

X expectation 可以看到差異：一個 qubit 的 `[3,±4]` 得到 ±0.96；補成 `[3,±4,0,0]` 時讀取最後一個 logical qubit X1，也得到 ±0.96。實驗同時保存 exact X、counts 與 fidelity，避免只核對 probabilities 而漏掉符號錯誤。

## 3. 路徑 A：直接載入 Simulator State

```python
state = simulator_state(amplitudes)
result = cudaq.get_state(loaded, state)
```

底層使用：

```python
@cudaq.kernel
def loaded(state: cudaq.State):
    q = cudaq.qvector(state)
```

`simulator_state` 依 target precision 建立 complex64 或 complex128 buffer，再呼叫 `cudaq.State.from_data`。這是本地 simulator 的狀態初始化路徑，**不提供本例在 QPU 上準備此態所需的 gate 成本**。不能因為 Python 只寫一行就稱它是 constant-cost hardware loading。

CUDA-Q 的 state buffer 以 q0 為 least-significant bit。[D3] 本文 logical `|q0 q1⟩` 順序是 `[00,01,10,11]`，二 qubit 載入前要重排為 `[a00,a10,a01,a11]`。取出時使用具名 `State.amplitude(label)`，轉回 logical 順序。測試逐一載入四個 basis states，專門檢查這個邊界。

## 4. 路徑 B：一個 Qubit 的 RY

若 d=2，正規化後為 `[a0,a1]`：

```text
θ = 2 atan2(a1,a0)
RY(θ)|0⟩ = a0|0⟩ + a1|1⟩
```

`atan2` 保留符號與象限，比單靠 `acos(a0)` 更適合本日含負數的例子。這條路徑需要一個 RY。

## 5. 兩個 Qubit：三個 RY 與兩個 CNOT

對 `[a00,a01,a10,a11]`，先計算兩個分支的長度：

```text
r0 = sqrt(a00²+a01²)
r1 = sqrt(a10²+a11²)
α  = 2 atan2(r1,r0)
β0 = 2 atan2(a01,a00)
β1 = 2 atan2(a11,a10)
```

某分支 norm 為 0 時，其 β 設為 0，因為該分支沒有 amplitude；這不是對所有輸入都忽略符號。

先用 RY(α) 準備 q0 的兩個分支權重，再讓 q1 在 q0=0 時旋轉 β0、q0=1 時旋轉 β1。實際電路分解為：

```text
q0: ──RY(α)──────────────●────────────────────●──
                        │                    │
q1: ──RY((β0+β1)/2)──────X──RY((β0−β1)/2)─────X──
```

q0=0 時 q1 兩次旋轉相加為 β0；q0=1 時兩次 X 使其中一次旋轉角反號，結果為 β1。這給出一個可逐步檢查的 signed-real state preparation。

```python
ry(angles[0], q[0])
ry(angles[1], q[1])
x.ctrl(q[0], q[1])
ry(angles[2], q[1])
x.ctrl(q[0], q[1])
```

沒有呼叫通用 state-preparation library。此範例可產生 `[1,0,0,1]/sqrt(2)` 的 Bell state，所以與 Day 12 的純單 qubit rotation product encoding 不同。

## 6. 成本要分層記錄

| 本例維度 | 補零後 amplitudes | Qubits | 明確 preparation gates |
|---:|---:|---:|---|
| 2 | 2 | 1 | 1 RY |
| 3 | 4 | 2 | 3 RY＋2 CNOT |
| 4 | 4 | 2 | 3 RY＋2 CNOT |

這是原始電路計數，沒有量測基底轉換、路由、硬體原生 gate 分解。特殊輸入的 rotation 或 CNOT 可能被 compiler 消除，不能把表格當成所有編譯輸出的精確成本。

host 必須讀取整個 feature vector、驗證、補零與正規化，這些陣列操作隨資料長度增加；qubit 數的對數成長不代表整個載入流程也只有對數成本。NVIDIA 的 state preparation 文件也區分 statevector 與將其編譯成電路的工作。[D13] 本文只實作到兩個 qubit，不用這個小例子推估一般高維或特殊結構資料的 gate 數。

每次取樣亦須準備狀態並量測；一次 Z 基底結果只是一個 bit string，無法讀回所有 amplitudes 與符號。GPU 計算環境也不改變這個量測限制。

## 7. 可執行範例與實驗

沿用 `.venv`，沒有新增套件；版本鏈入口：[requirements-day13.txt](../../requirements-day13.txt)。在專案根目錄執行：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day13/demo.py
OMP_NUM_THREADS=1 python articles/day13/demo.py --values 1 -2 3 -4
OMP_NUM_THREADS=1 python articles/day13/demo.py --values 0 0 -3 4 --backend nvidia

OMP_NUM_THREADS=1 python articles/day13/experiment.py
OMP_NUM_THREADS=1 python articles/day13/experiment.py --backend nvidia
```

demo 印出 normalized amplitudes、角度、明確電路圖、probabilities 與 fidelity。experiment 每個 backend 有 12 組案例，包含 2／3／4 維、尺度倍數、global／relative sign、不同 basis、零分支及 Bell state。

每組以 direct loading 和 explicit gates 兩條路徑核對目標 state，另對 gates 計算 exact X 與 1,000 shots 的 Z counts。結果保存到 `results/day13/<backend>/`：`predictions.json`、`summary.json`、`circuit.txt`。重跑更新該目錄，可用 `--output-dir /tmp/day13-check` 另存。

## 8. 驗證與下一篇

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day13 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY13_TARGET=nvidia python -m unittest discover -s articles/day13 -p 'test_*.py' -v
```

六個測試涵蓋正規化與輸入不變性、錯誤與極端有限值、尺度／global phase、relative sign、四個 basis 的載入順序、固定 seed 隨機 signed vectors 與零分支。完整數字見 [實驗結果](../../results/day13/README.md)。

本日為無噪聲 simulator 驗證，沒有 QPU 實測、分類 accuracy、載入速度比較或量子優勢宣稱。[Day 14](../day14/README.md) 將回到 Feature Map＋Ansatz，整理資料表示與可訓練電路的組合。

## 9. 來源

- [D5] [NVIDIA Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：amplitude encoding 定義、State.from_data。
- [D3] [NVIDIA Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)：statevector 的 little-endian 順序。
- [D13] [NVIDIA Approximate State Preparation using MPS Sequential Encoding](https://nvidia.github.io/cuda-quantum/0.13.0/applications/python/mps_encoding.html)：state preparation 是將 statevector 轉成電路；本文未實作 MPS 方法。

查閱日期 2026-09-07，實際 CUDA-Q 0.15.1。小型 RY／CNOT 分解由本文公式與測試核對，共用索引見 [REFERENCES.md](../../REFERENCES.md)。
