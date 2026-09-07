# Day 07｜第一個 CUDA-Q Quantum Kernel：Allocation、參數與 Control Flow

Day 4–6 已經執行過 CUDA-Q kernel。今天把它拆開來理解，從一個 Qubit 到可調整大小的 register，讓同一份 kernel 根據參數產生不同電路。

今天要回答：**哪些事情留在 Python host，哪些事情寫進 quantum kernel？`if`、`for` 和 controlled gate 又有什麼不同？**

## 1. Kernel 是 Host 與 Quantum Code 的分界

`@cudaq.kernel` 標記由 CUDA-Q 編譯與執行的函式。它沿用部分 Python 語法，但不是任意 Python 程式都可以放進去；官方規格列出允許的型別與控制流程。[D10][D11]

| 放在 Python host | 放在本日 kernel |
|---|---|
| CLI、讀寫檔案、檢查輸入範圍 | 配置 qubit／qvector |
| 選擇 target、設定 random seed | 套用 RY、X、controlled-X |
| NumPy 參考計算、metrics | 用參數控制 `if` 分支 |
| `cudaq.sample`、`cudaq.get_state` | 用 `for` 重複套用 gates／量測 |
| CSV／JSON 結果紀錄 | `mz` 量測操作 |

本日 `experiment.py` 負責 host 工作，`kernels.py` 只放電路。先在 host 檢查輸入，再呼叫 kernel，錯誤就比較容易定位。

## 2. 環境與檔案

沿用 Day 6 的 `.venv`，沒有新增套件：

```bash
source .venv/bin/activate
python -m pip check
```

全新 Ubuntu checkout 可使用 `python3.12 -m venv .venv`，啟用後安裝 `python -m pip install -r requirements-day07.txt`。本日依賴入口沿用已驗證的 CUDA-Q 0.15.1／NumPy 2.2.6 stack。

- [kernels.py](kernels.py)：兩個量子 kernel。
- [demo.py](demo.py)：可調參數的命令列示範與電路圖。
- [experiment.py](experiment.py)：NumPy 參考與 36 組設定掃描。
- [test_kernels.py](test_kernels.py)：allocation、分支、迴圈、順序及輸入契約測試。
- [results/day07](../../results/day07/)：CPU／GPU 各自的 CSV、JSON 與電路文字圖。

## 3. 最小範例：配置一個 Qubit

```python
import cudaq

@cudaq.kernel
def single_rotation(theta: float):
    q = cudaq.qubit()
    ry(theta, q)
    mz(q)
```

`cudaq.qubit()` 配置一個初始為 `|0⟩` 的 Qubit；`ry(theta, q)` 套用 Day 3 的 RY；`mz(q)` 在 Z 基底量測。這三個步驟都屬於 kernel。[D10]

呼叫與結果處理留在 host：

```python
cudaq.set_target("qpp-cpu")
cudaq.set_random_seed(42)
counts = cudaq.sample(single_rotation, 0.0, shots_count=32)
print(counts)  # theta=0，預期全部為 0
```

完整函式可在 `kernels.py` 找到，測試會驗證 `theta=0` 全為 0、`theta=π` 全為 1。

`theta: float` 是 kernel signature 的型別資訊。可傳角度不表示模型已訓練：今天的角度來自實驗設定，沒有 optimizer。

## 4. 從 Qubit 到 Register

將固定兩個 Qubit 改成 `n` 個：

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

這就是本日的主要 kernel，完整版本位於 [kernels.py](kernels.py)。五個參數各自控制一件事：

| 參數 | 型別 | 用途 |
|---|---|---|
| `n` | int | register 大小 |
| `theta` | float | q0 的 RY 角度，單位 radians |
| `entangle` | bool | 是否執行 controlled-X 迴圈 |
| `flip_last` | bool | 是否在最後一個 Qubit 加 X |
| `measure` | bool | 是否執行 terminal measurements |

`q[i]` 是量子位元的 handle，不是已知的 classical 0／1。索引合法範圍是 `0` 到 `n−1`，所以 host 先要求 `n ≥ 1`。示範程式把 n 限在 1–8，方便小型驗證；這不是 CUDA-Q 的 qubit 上限。

## 5. `for`：用一份程式描述不同大小的電路

```python
for i in range(1, n):
    x.ctrl(q[0], q[i])
```

這段依序把 q0 當 control，對 q1 到 q(n−1) 套用 X。它是 star 結構，不是 q0→q1→q2 的 chain。

當 `theta=π/2`、`entangle=True`、`flip_last=False`：

```text
n=2: (|00⟩ + |11⟩)/√2       Bell state
n=3: (|000⟩ + |111⟩)/√2     GHZ state
n=4: (|0000⟩ + |1111⟩)/√2   GHZ state
```

這些結果可由 Day 3 的 RY 與 Day 4 的 controlled-X 規則直接推導。一般角度得到：

```text
cos(theta/2)|00…0⟩ + sin(theta/2)|11…1⟩
```

n=1 時 `range(1, 1)` 是空迴圈，只剩單一 RY；測試特別涵蓋這個邊界。`entangle=True` 只是要求 controlled-X 結構，不保證每個輸入都糾纏，例如 theta=0 仍得到 product state。

## 6. `if` 與 Controlled Gate 的差別

`if entangle` 判斷的是 host 傳入的 bool，決定這次執行是否包含那組 gates。`x.ctrl(q[0], q[i])` 則是量子 controlled operation，作用於整個 state，不會先量測 q0。

這兩件事不能互換：

| 操作 | 條件來源 | 是否先量測 control |
|---|---|---|
| `if entangle:` | 呼叫端傳入的 bool | 否 |
| `x.ctrl(q[0], q[i])` | control qubit 的量子狀態 | 否 |
| 依中途 measurement 結果分支 | 執行中產生的 classical measurement result | 是 |

最後一列屬於 measurement-based feedback，本日沒有實作。今天的分支全由輸入參數決定，不能拿本例的成功推定每個 backend 都支援相同的動態電路功能。

`if flip_last` 是另一個可觀察的分支。對三 Qubit GHZ state 加最後一個 X，支援集合從 `000／111` 變成 `001／110`。

## 7. 直接修改參數觀察結果

從 repository 根目錄執行：

```bash
# 三 Qubit GHZ，預設 theta=π/2。
OMP_NUM_THREADS=1 python articles/day07/demo.py

# 關閉 controlled-X 迴圈。
OMP_NUM_THREADS=1 python articles/day07/demo.py --no-entangle

# 在最後一個 Qubit 加 X。
OMP_NUM_THREADS=1 python articles/day07/demo.py --flip-last

# 同樣程式切換至 RTX 3060，register 改成 4 個 Qubit。
OMP_NUM_THREADS=1 python articles/day07/demo.py --backend nvidia --qubits 4
```

示範會印出明確的 target、`cudaq.draw` 電路圖、counts 與驗證結果。預設三 Qubit、theta=π/2 時：

| entangle | flip_last | 理想量測結果，各約 50% |
|---|---|---|
| false | false | `000`、`100` |
| false | true | `001`、`101` |
| true | false | `000`、`111` |
| true | true | `001`、`110` |

另外可用 `--theta 0` 得到 deterministic 測試；無 flip 時全部是零字串，有 flip 時只有最後一位是 1。

## 8. 位元順序不能只用 GHZ 驗證

`000` 與 `111` 反過來還是一樣，因此 GHZ counts 不能證明程式的位元順序正確。本日採 `explicit_measurements=True`，kernel 依 q0、q1、…順序呼叫 mz，結果字串也按該順序解讀。[D5]

測試使用 theta=π、n=3 的非對稱案例：

```text
entangle=False, flip_last=False → 100
entangle=False, flip_last=True  → 101
entangle=True,  flip_last=False → 111
entangle=True,  flip_last=True  → 110
```

NumPy 參考把 q0 放在 index 的最高位。CUDA-Q exact-state 核對使用 `state.amplitude(bitstring)` 明確查詢各 basis label，避免直接假定底層 buffer 的排列。[D5] 這個 API 與 counts 的標籤對照也由上述非對稱案例驗證。

## 9. 實驗設計與保存

```bash
OMP_NUM_THREADS=1 python articles/day07/experiment.py --backend qpp-cpu
OMP_NUM_THREADS=1 python articles/day07/experiment.py --backend nvidia
```

每個 backend 掃描：

| 欄位 | 設定 |
|---|---|
| n | 2、3、4 |
| theta | 0、π/2、π |
| entangle | false、true |
| flip_last | false、true |
| shots／seed | 每組 1,000／42 |
| noise | 無 |
| 組合數 | 3 × 3 × 2 × 2 = 36 |

NumPy reference 重用 Day 3 的 RY；controlled-X 與最後一位 X 以 basis index permutation 實作。每組比較 state normalization、與 reference 的 fidelity、exact probabilities、量測 bitstrings 與 shots 總數，另記錄 sampled distribution 的 TVD。

JSON 保存 Python／NumPy／CUDA-Q／target precision、seed、最大 fidelity error 與是否全數通過。CSV 保存每組參數、counts、檢查和時間；`counts`／`checks` 欄位是 JSON 字串，可用 `json.loads` 讀回。`circuit.txt` 是 n=3、theta=π/2、開啟 entangle 的實際 draw 結果。

每個 backend 各自寫入 `results/day07/<backend>/`；重跑會更新該目錄，可加 `--output-dir /tmp/day07-check` 另存。

這是 correctness sweep，固定一個 seed 方便逐組追溯，不是多 seed 的統計比較或訓練實驗。TVD 只用來描述 finite-shot 波動；exact state checks 才是電路數學正確性的主要依據。執行時間含 JIT／初始化，沒有 CPU／GPU 效能結論。

## 10. 驗證與容易踩到的坑

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day07 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY07_TARGET=nvidia python -m unittest discover -s articles/day07 -p 'test_*.py' -v
```

同一套 6 個測試會分別驗證 CPU 與 GPU。實測結果與環境限制見 [結果紀錄](../../results/day07/README.md)。

- `cudaq.qubit()` 與 `cudaq.qvector(n)` 配置的是量子資源，不是一般 NumPy array。
- kernel 的 input parameters 需要適當型別；CLI 先轉換與驗證，再傳入 kernel。
- `range(1, n)` 不含 n，q[n] 超出範圍。
- Python bool 分支不等於 quantum controlled gate。
- `measure=False` 留給 simulator 的 exact-state 驗證，正式 sampling 使用 `measure=True`。
- n=1、theta endpoints 與非對稱 basis states 都值得測；只看一張 GHZ histogram 不足以確認細節。
- Host 的檔案 I/O、NumPy reference 與 metrics 都放在 kernel 外。

## 11. 本日來源與下一篇

- [D10] [NVIDIA Building Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html)：allocation、參數與電路建構。
- [D11] [NVIDIA Quantum Kernels Specification](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/kernels.html)：host／kernel 邊界、型別與控制流程。
- [D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：explicit measurements、State.amplitude 與 draw。

查閱日期：2026-09-06。實際驗證使用 CUDA-Q 0.15.1；最新文件與固定版本的行為以測試交叉核對。共用索引見 [REFERENCES.md](../../REFERENCES.md)。

[Day 08](../day08/README.md) 將用相同 state preparation 比較 `sample`、`run`、`observe`，釐清 counts、單次 classical return values 與 expectation values 的差別。
