# Day 07｜第一個 CUDA-Q 量子核心程式：量子位元、參數與流程控制

[Day6](../day06/README.md) 釐清 CUDA-Q、CUDA 與 cuQuantum 的分工，並用環境診斷與 貝爾態小型測試確認指定後端能正確執行。Day7 接著拆解量子程式的寫法，從單一量子位元擴展到可調整大小的電路，理解參數、迴圈與分支如何決定電路結構。

**同一份電路程式可以接受不同設定，但一般 Python 工作與量子操作需要清楚分工。** 這個分工讓電路容易修改，也讓錯誤容易追查。主控端程式（host）負責檢查輸入、選擇執行後端、呼叫電路與保存結果；以 `@cudaq.kernel` 標記的函式則負責配置量子位元、施加量子閘與量測。本章先使用 `cudaq.qubit()` 配置一個量子位元，再以 `cudaq.qvector(n)` 配置一組量子位元，透過帶有型別的參數調整數量、旋轉角度與電路選項。型別表示數值、整數或真假選項等資料種類，迴圈則用來重複執行操作。核心重點是區分一般條件分支與量子控制閘：`if` 根據傳入的布林值（真或假）決定是否執行一組操作，受控 X 閘則直接作用於量子狀態，不會先量測控制位元。實作藉由迴圈建立不同大小的電路，並以 NumPy 參考結果、最小量子位元數等邊界條件，以及左右交換後不同的量測案例核對正確性。可調整角度讓同一份電路接受不同輸入，但本章尚未用最佳化器，也就是依誤差調整參數的方法，進行模型訓練。

[Day6](../day06/README.md) clarified the roles of CUDA-Q, CUDA, and cuQuantum, using environment diagnostics and a small Bell-state test to verify execution on the requested backend. Day7 examines how quantum programs are written, extending a single-qubit example to circuits of adjustable size and explaining how parameters, loops, and branches determine circuit structure.

**One circuit program can accept different configurations, while ordinary Python tasks and quantum operations have distinct roles.** Keeping those roles clear makes circuits easier to modify and problems easier to trace. The host program validates inputs, selects the execution backend, invokes circuits, and saves results. A function marked with `@cudaq.kernel` allocates qubits, applies gates, and performs measurements. The chapter starts with `cudaq.qubit()` for a single qubit, then uses `cudaq.qvector(n)` for a register, with typed parameters controlling the qubit count, rotation angle, and circuit options. A central distinction is between classical branching and quantum-controlled gates: an `if` statement uses an input Boolean to decide whether to execute operations, whereas controlled-X acts directly on the quantum state without first measuring the control qubit. Loops construct circuits of different sizes, while NumPy references, boundary cases, and asymmetric measurement examples check correctness. An adjustable angle is a program input; no optimizer learns parameters in this chapter.

---

Day 4–6 已經執行過 CUDA-Q 量子核心程式。今天把它拆開來理解，從一個量子位元到可調整大小的量子暫存器，讓同一份量子核心程式根據參數產生不同電路。

本章以三個問題串起程式：**哪些事情留在 Python 主控端，哪些事情寫進 量子核心程式？`if`、`for` 和量子控制閘又有什麼不同？**

## 1. 量子核心程式是主控端與量子程式的分界

主控端（host）是安排工作的普通 Python 程式；量子核心程式（quantum kernel）描述量子操作。

`@cudaq.kernel` 是放在函式前的標記，表示這個函式交由 CUDA-Q 編譯與執行。編譯是將程式描述轉成可執行的運算形式。執行後端（backend）指定實際使用的模擬器或硬體；本章使用模擬器，以一般電腦的數值運算模仿量子電路。量子核心程式沿用部分 Python 語法，但不是任意 Python 程式都可以放進去；官方規格列出允許的型別與控制流程。[D10][D11]

| 放在 Python 主控端 | 放在本日量子核心程式 |
|---|---|
| CLI、讀寫檔案、檢查輸入範圍 | 配置量子位元／qvector |
| 選擇執行後端、設定隨機種子 | 套用 RY、X、受控 X 閘 |
| NumPy 參考計算、評估指標 | 用參數控制 `if` 分支 |
| `cudaq.sample`、`cudaq.get_state` | 用 `for` 重複套用量子閘／量測 |
| CSV／JSON 結果紀錄 | `mz` 量測操作 |

CLI 是命令列介面，讓程式接受終端機輸入的選項；CSV 是表格文字檔，JSON 則用欄位名稱保存結構化資料。NumPy 是 Python 的數值運算套件，用來獨立計算參考答案。`sample` 取得重複量測的計數，`get_state` 則讀取模擬器內部的狀態向量。

本日 `experiment.py` 負責主控端工作，`kernels.py` 只放電路。先在主控端檢查輸入，再呼叫量子核心程式，錯誤就比較容易定位。

## 2. 環境與檔案

虛擬環境 `.venv` 是專案獨立保存 Python 套件的資料夾；`pip check` 檢查套件依賴是否相容，也就是各套件所需的其他套件版本能否搭配使用。沿用 Day 6 的環境，沒有新增套件：

```bash
source .venv/bin/activate
python -m pip check
```

Ubuntu 是本專案主機使用的 Linux 作業系統。全新專案副本可使用 `python3.12 -m venv .venv`，啟用後安裝 `python -m pip install -r requirements-day07.txt`。本日依賴入口沿用已驗證的 CUDA-Q 0.15.1／NumPy 2.2.6 套件組合。

- [kernels.py](kernels.py)：兩個量子核心程式。
- [demo.py](demo.py)：可調參數的命令列示範與電路圖。
- [experiment.py](experiment.py)：NumPy 參考與 36 組設定逐一計算。
- [test_kernels.py](test_kernels.py)：量子位元配置、分支、迴圈、順序及合法輸入條件測試。
- [results/day07](../../results/day07/)：CPU／GPU 各自的 CSV、JSON 與電路文字圖。

## 3. 最小範例：配置一個量子位元

```python
import cudaq

@cudaq.kernel
def single_rotation(theta: float):
    q = cudaq.qubit()
    ry(theta, q)
    mz(q)
```

`cudaq.qubit()` 配置一個初始為 `|0⟩` 的量子位元；`ry(theta, q)` 套用 Day 3 的 RY；`mz(q)` 在 Z 基底量測。這三個步驟都屬於量子核心程式。[D10]

呼叫與結果處理留在主控端：

```python
cudaq.set_target("qpp-cpu")
cudaq.set_random_seed(42)
counts = cudaq.sample(single_rotation, 0.0, shots_count=32)
print(counts)  # theta=0，預期全部為 0
```

`ry` 是繞布洛赫球 y 軸旋轉的量子閘，以角度改變振幅；振幅的絕對值平方才是量測機率。`mz` 在 Z 基底下量測，也就是區分 0 與 1。`π` 弧度代表半圈。`set_target` 選擇執行後端，`qpp-cpu` 是 CPU 模擬器；CPU 是一般電腦的中央處理器。`set_random_seed` 設定隨機種子，方便在相同環境重做抽樣。`shots_count` 是要求的量測次數。

完整函式可在 `kernels.py` 找到，測試會驗證 `theta=0` 全為 0、`theta=π` 全為 1。

`theta: float` 表示參數 `theta` 接受浮點數，也就是電腦以有限位數表示的小數。參數是呼叫函式時傳入的設定，型別則說明設定值屬於哪一類資料。可傳角度不表示模型已訓練：今天的角度來自實驗設定，沒有最佳化器。

## 4. 從量子位元到量子暫存器

量子暫存器（register）是一組可透過編號操作的量子位元。將固定兩個量子位元改成 `n` 個：

```python
@cudaq.kernel
def register_circuit(n: int, theta: float, entangle: bool,
                     flip_last: bool, measure: bool):
    q = cudaq.qvector(n)
    ry(theta, q[0])
    if entangle:
        for i in range(1, n):
            x.ctrl(q[0], q[i])
    if flip_last:
        x(q[n - 1])
    if measure:
        for i in range(n):
            mz(q[i])
```

這就是本日的主要量子核心程式，完整版本位於 [kernels.py](kernels.py)。五個參數各自控制一件事。`int` 是整數，`float` 是浮點數，`bool` 是布林值，只能是 `True`（真）或 `False`（假）：

| 參數 | 型別 | 用途 |
|---|---|---|
| `n` | int | 量子暫存器大小 |
| `theta` | float | q0 的 RY 角度，單位弧度 |
| `entangle` | 布林值 | 是否執行受控 X 閘迴圈 |
| `flip_last` | 布林值 | 是否在最後一個量子位元加 X |
| `measure` | 布林值 | 是否執行電路末端的量測 |

`q[i]` 指向編號為 `i` 的量子位元，供後續量子閘操作使用；它不是量測後得到的 0 或 1。索引就是編號，這裡從 0 開始。索引合法範圍是 `0` 到 `n−1`，所以主控端先要求 `n ≥ 1`。示範程式把 n 限在 1–8，方便小型驗證；這不是 CUDA-Q 的量子位元上限。

## 5. `for`：用一份程式描述不同大小的電路

```python
for i in range(1, n):
    x.ctrl(q[0], q[i])
```

`for` 是重複執行操作的迴圈；`range(1, n)` 依序提供 1 到 `n−1`，不包含 `n`。例如 `n=3` 時，`i` 先取 1，再取 2。

這段依序把 q0 當控制位元，對 q1 到 q(n−1) 套用 X。它是星狀結構，也就是每個受控操作都以 q0 為中心；鏈狀結構才是依序由 q0 控制 q1、q1 控制 q2。受控 X 閘又稱 CNOT，控制位元為 1 時翻轉目標位元，為 0 時保持目標位元。

當 `theta=π/2`、`entangle=True`、`flip_last=False`：

```text
n=2: (|00⟩ + |11⟩)/√2       Bell state
n=3: (|000⟩ + |111⟩)/√2     GHZ state
n=4: (|0000⟩ + |1111⟩)/√2   GHZ state
```

貝爾態（Bell state）是兩量子位元的糾纏態；這裡的 GHZ 態是它的多位元延伸，名稱來自 Greenberger、Horne 與 Zeilinger。糾纏表示這個純態無法拆成各位元獨立純態的乘積；純態則是能用單一向量完整描述的狀態。

這些結果可由 Day 3 的 RY 與 Day 4 的受控 X 閘規則直接推導。`cos` 與 `sin` 是餘弦、正弦函數，決定下式兩項的振幅；`theta` 是旋轉角度。當 `entangle=True` 且 `flip_last=False` 時，一般角度得到：

```text
cos(theta/2)|00…0⟩ + sin(theta/2)|11…1⟩
```

n=1 時 `range(1, 1)` 是空迴圈，只剩單一 RY；測試特別涵蓋這個邊界。`entangle=True` 只是要求受控 X 閘結構，不保證每個輸入都糾纏，例如 `theta=0` 仍得到乘積態，也就是可以拆成各位元獨立狀態的組合。

## 6. `if` 與量子控制閘的差別

`if` 是條件分支：條件為 `True` 時執行縮排內的操作，為 `False` 時略過。

`if entangle` 判斷的是主控端傳入的布林值，決定這次執行是否包含那組量子閘。`x.ctrl(q[0], q[i])` 則是量子控制操作，作用於整個狀態，不會先量測 q0。

這兩件事不能互換：

| 操作 | 條件來源 | 是否先量測控制位元 |
|---|---|---|
| `if entangle:` | 呼叫端傳入的布林值 | 否 |
| `x.ctrl(q[0], q[i])` | 控制位元的量子狀態 | 否 |
| 依中途量測結果分支 | 執行中產生的一般量測結果 | 是 |

量測回饋指先讀取中途量測結果，再依結果決定後續操作。最後一列屬於量測回饋，本日沒有實作。今天的分支全由輸入參數決定，不能拿本例的成功推定每個執行後端都支援相同的動態電路功能。

`if flip_last` 是另一個可觀察的分支。對三量子位元 GHZ 狀態加最後一個 X，可能出現的結果從 `000／111` 變成 `001／110`。

## 7. 直接修改參數觀察結果

`OMP_NUM_THREADS=1` 限制 CPU 使用的 OpenMP 執行緒數；OpenMP 是分配平行工作的工具，執行緒是程序內可分別安排的工作單位。GPU 是擅長平行運算的圖形處理器，RTX 3060 是本次使用的 GPU 型號；`nvidia` 後端使用 GPU 模擬量子電路。

從專案根目錄執行：

```bash
# 三量子位元 GHZ 態，預設 theta=π/2。
OMP_NUM_THREADS=1 python articles/day07/demo.py

# 關閉受控 X 閘迴圈。
OMP_NUM_THREADS=1 python articles/day07/demo.py --no-entangle

# 在最後一個量子位元加 X。
OMP_NUM_THREADS=1 python articles/day07/demo.py --flip-last

# 同樣程式切換至 RTX 3060，量子暫存器改成 4 個量子位元。
OMP_NUM_THREADS=1 python articles/day07/demo.py --backend nvidia --qubits 4
```

示範會印出明確的執行後端、`cudaq.draw` 電路圖、量測計數與驗證結果。預設三量子位元、theta=π/2 時：

| entangle | flip_last | 理想量測結果，各約 50% |
|---|---|---|
| false | false | `000`、`100` |
| false | true | `001`、`101` |
| true | false | `000`、`111` |
| true | true | `001`、`110` |

另外可用 `--theta 0` 得到結果確定的測試；未啟用 `--flip-last` 時全部是 0，啟用後只有最後一位是 1。

## 8. 位元順序不能只用 GHZ 驗證

`000` 與 `111` 反過來還是一樣，因此 GHZ 量測計數不能證明程式的位元順序正確。本日採 `explicit_measurements=True`，量子核心程式依 q0、q1、…順序呼叫 mz，結果字串也按該順序解讀。[D5]

測試使用 theta=π、n=3 的非對稱案例：

```text
entangle=False, flip_last=False → 100
entangle=False, flip_last=True  → 101
entangle=True,  flip_last=False → 111
entangle=True,  flip_last=True  → 110
```

NumPy 參考把 q0 放在索引的最高位，也就是位元字串最左側，例如 `100` 對應整數索引 4。CUDA-Q 精確狀態核對使用 `state.amplitude(bitstring)` 明確查詢各基底標籤（例如 `000` 或 `100`），避免直接假定底層記憶體緩衝區的排列。[D5] 這個程式介面（API）與量測計數的標籤對照也由上述非對稱案例驗證。

## 9. 實驗設計與保存

```bash
OMP_NUM_THREADS=1 python articles/day07/experiment.py --backend qpp-cpu
OMP_NUM_THREADS=1 python articles/day07/experiment.py --backend nvidia
```

每個執行後端掃描：

| 欄位 | 設定 |
|---|---|
| n | 2、3、4 |
| theta | 0、π/2、π |
| entangle | false、true |
| flip_last | false、true |
| 量測次數／隨機種子 | 每組 1,000／42 |
| 雜訊 | 無 |
| 組合數 | 3 × 3 × 2 × 2 = 36 |

NumPy 參考計算重用 Day 3 的 RY；受控 X 閘與最後一位 X 以基底索引置換實作，也就是把各振幅移到操作後應在的位置。每組比較狀態正規化、與參考計算的保真度、精確機率、量測位元字串與量測次數總數，另記錄抽樣分布的 TVD。

正規化表示振幅絕對值平方的總和為 1。保真度（fidelity）衡量兩個狀態的接近程度，本例使用正規化向量內積的絕對值平方，1 表示相同物理狀態。內積是對應分量相乘後加總，複數向量的第一個向量需先取複共軛，也就是將虛部反號。總變異距離（total variation distance，TVD）將各結果的抽樣機率與理論機率之差取絕對值，加總後除以 2；0 表示分布相同。

數值精度描述電腦用多少位數保存與計算數值；有限精度可能造成微小誤差。

JSON 保存 Python、NumPy、CUDA-Q 的版本與後端數值精度、隨機種子、最大保真度誤差與是否全數通過。CSV 保存每組參數、量測計數、檢查和時間；`counts`／`checks` 欄位是 JSON 字串，可用 `json.loads` 讀回。`circuit.txt` 是 n=3、theta=π/2、開啟 entangle 的實際電路圖輸出。

每個執行後端各自寫入 `results/day07/<backend>/`；重跑會更新該目錄，可加 `--output-dir /tmp/day07-check` 另存。

這是逐組正確性檢查，固定一個隨機種子方便逐組追溯，不是多隨機種子的統計比較或訓練實驗。TVD 只用來描述有限次量測的波動；精確狀態檢查才是電路數學正確性的主要依據。執行時間含即時編譯（JIT，執行時將程式轉成可用的運算形式）與初始化，沒有 CPU／GPU 效能結論。

## 10. 驗證與常見誤解

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day07 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY07_TARGET=nvidia python -m unittest discover -s articles/day07 -p 'test_*.py' -v
```

同一套 6 個測試會分別驗證 CPU 與 GPU。實測結果與環境限制見 [結果紀錄](../../results/day07/README.md)。

- `cudaq.qubit()` 與 `cudaq.qvector(n)` 配置的是量子資源，不是一般 NumPy 陣列。
- 量子核心程式的輸入參數需要適當型別；CLI 先轉換與驗證，再傳入量子核心程式。
- `range(1, n)` 不含 n，q[n] 超出範圍。
- Python 布林值分支不等於 量子控制閘。
- `measure=False` 留給模擬器的精確狀態驗證，正式抽樣使用 `measure=True`。
- n=1、theta 端點與非對稱基底狀態都值得測；只看一張 GHZ 次數分布圖不足以確認細節。
- 主控端的檔案讀寫、NumPy 參考計算與評估指標都放在量子核心程式外。

## 11. 本日來源與下一篇

- [D10] [NVIDIA Building Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html)：量子位元配置、參數與電路建構。
- [D11] [NVIDIA Quantum Kernels Specification](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/kernels.html)：主控端／量子核心程式邊界、型別與控制流程。
- [D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：明確量測順序、振幅查詢與電路繪圖。

查閱日期：2026-09-06。實際驗證使用 CUDA-Q 0.15.1；最新文件與固定版本的行為以測試交叉核對。共用索引見 [REFERENCES.md](../../REFERENCES.md)。

[Day 08](../day08/README.md) 將用相同狀態準備比較 `sample`、`run`、`observe`，釐清量測計數、單次一般回傳值與期望值的差別；期望值是依各結果的機率計算出的平均值。
