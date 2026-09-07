# Day 04｜Superposition、Entanglement、Measurement：建立第一個 Bell State

Day 3 用矩陣改變一個 Qubit；今天把狀態擴展成兩個 Qubit，先用 NumPy 看清楚每一步，再以 CUDA-Q 在 CPU 與 RTX 3060 上執行相同概念。

今天的核心問題是：**兩個量子位元各自都像隨機結果，為什麼合在一起卻有固定的關係？**

## 1. 今天的交付物

- [bell_state.py](bell_state.py)：tensor product、CNOT、Bell state 與經典混合態的 NumPy 參考實作。
- [cudaq_bell.py](cudaq_bell.py)：CUDA-Q 電路、Z／X 基底量測、CPU／GPU 示範。
- [experiment.py](experiment.py)：3 種狀態 × 2 種基底 × 3 種 shots × 5 個 seeds，共 90 筆／backend。
- [test_bell_state.py](test_bell_state.py)、[test_cudaq_bell.py](test_cudaq_bell.py)：理論性質與框架整合測試。
- [results/day04](../../results/day04/)：分 backend 保存 CSV 與 JSON。

這是量子基礎與模擬器驗證實驗。經典混合態是概念對照，NumPy 是數值參考；今天尚未建立 ML 模型或比較分類效能。

## 2. 延續專案獨立環境

沿用 Day 3 的 Python 3.12 `.venv`，增加 CUDA 12 版 CUDA-Q 0.15.1，保留 NumPy 2.2.6。在 repository 根目錄執行：

```bash
source .venv/bin/activate
python -m pip install -r requirements-day04.txt
python -m pip check
```

若是全新 checkout，先執行 `python3.12 -m venv .venv`。實際驗證使用的完整依賴清單見 [requirements-day04-lock.txt](../../requirements-day04-lock.txt)，需要相同依賴版本時改用此檔安裝。

本專案明確選用 `cuda-quantum-cu12` 發行套件，不另外疊裝 `cudaq` 或其他 CUDA-Q 二進位發行套件。官方提醒不同發行套件可能衝突。[D1]

Day 4 先完成必要的安裝與 Bell state 驗證；Day 6 再系統整理 CUDA-Q、CUDA、cuQuantum 與環境設定。這樣 Day 4 的程式可立即執行。

## 3. 兩個 Qubit：用 Tensor Product 組合狀態

本日 NumPy 向量順序固定為 `|q0 q1⟩`，也就是 `[00, 01, 10, 11]`：

```text
|00⟩ = |0⟩ ⊗ |0⟩ = [1, 0, 0, 0]ᵀ
|01⟩ = |0⟩ ⊗ |1⟩ = [0, 1, 0, 0]ᵀ
|10⟩ = |1⟩ ⊗ |0⟩ = [0, 0, 1, 0]ᵀ
|11⟩ = |1⟩ ⊗ |1⟩ = [0, 0, 0, 1]ᵀ
```

`⊗` 是 tensor product，在 NumPy 使用 `np.kron`。若只對 q0 施加 H，整個系統的操作為 `H ⊗ I`：

```text
(H ⊗ I)|00⟩ = (|00⟩ + |10⟩)/√2 = |+⟩ ⊗ |0⟩
```

這個狀態已經有 superposition，但仍能拆成兩個單量子位元狀態的乘積。對兩個 Qubit 都施加 H，得到的 `|+⟩ ⊗ |+⟩` 也仍是 product state。

## 4. CNOT：讓 q0 控制 q1 的翻轉

今天使用 q0 為 control、q1 為 target：

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

CNOT 不會量測 control，它對整個疊加態做線性、unitary 轉換。把它接在 `H ⊗ I` 後面：

```text
|00⟩ → (|00⟩ + |10⟩)/√2 → (|00⟩ + |11⟩)/√2 = |Φ+⟩
```

這就是本日的 Bell state。CNOT 是否產生糾纏取決於輸入；例如 `CNOT|00⟩` 仍是 `|00⟩`。

## 5. NumPy 範例：看見 Bell State 的 Amplitudes

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

預期 amplitudes 約為 `[0.7071, 0, 0, 0.7071]`，Z 基底機率為 `[0.5, 0, 0, 0.5]`。完整示範：

```bash
python articles/day04/bell_state.py
```

對純態，能否拆成 `|a⟩ ⊗ |b⟩` 是判斷兩部分有無糾纏的方式。本日將 Bell amplitudes reshape 成 2×2 矩陣，其 rank 為 2，不能拆成兩個向量的外積；測試也驗證單一 Qubit 的 reduced state 是 `I/2`。混合態需要更一般的 separability 定義，不能直接套用這個純態 rank 檢查。[F2]

## 6. CUDA-Q 範例：把矩陣換成 Quantum Kernel

最小 Bell state kernel 如下，對應 NVIDIA 官方入門範例的 H + controlled-X 結構。[D1]

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

Repository 的 [cudaq_bell.py](cudaq_bell.py) 另支援 product state、混合態準備與 X 基底。直接執行：

```bash
python articles/day04/cudaq_bell.py --backend qpp-cpu
python articles/day04/cudaq_bell.py --backend nvidia
```

本次兩個 backend 的 Bell Z 示範都得到 `00: 493`、`11: 507`，`01`／`10` 為 0。這是指定環境、seed 的結果，不保證不同版本與 backend 都產生相同 counts。

```text
q0: ──H──●──MZ
         │
q1: ─────X──MZ
```

`h`、`x.ctrl`、`mz` 是 kernel 內由 CUDA-Q 編譯器識別的操作，不能當成普通 Python 函式在 kernel 外呼叫。本日採 `explicit_measurements=True`，按 q0、q1 的量測次序串接輸出，並以只翻轉 q0 的非對稱案例測試得到 `10`。[D5]

## 7. Measurement 與 Shots：一個結果不是機率分布

Bell state 在 Z 基底下每次得到 `00` 或 `11`。單看任一 Qubit，0／1 各半；合看兩個結果，則永遠相同。條件在 q0 已量到 0 時，q1 為 0；但我們不能選擇 q0 的隨機結果來傳送訊息。

真實硬體必須重新準備並量測以累積 shots；模擬器可以先建立狀態，再從其分布抽樣，不必真的做相同次數的完整狀態演化。[D3]

估計某個結果的機率 `p`，獨立抽樣的標準誤為 `sqrt(p(1-p)/N)`。對 `p=0.5`：

| Shots N | 機率估計的理論標準誤 |
|---:|---:|
| 100 | 0.0500 |
| 1,000 | 約 0.0158 |
| 10,000 | 0.0050 |

Shots 增加通常能縮小抽樣波動，但某一個 seed 的誤差不保證單調下降。因此本日保留五組 seeds，而不只挑最好看的結果。

## 8. 為什麼只看到 00／11，還不能認定 Entanglement？

想像每次先丟一枚經典公平硬幣，選擇準備 `|00⟩` 或 `|11⟩`。它也會產生相同的 Z 基底統計，但這是 separable mixture。[F2]

為了用程式描述這個對照，今天只引入最少的 density matrix：純態使用 `ρ = |ψ⟩⟨ψ|`；經典混合則把每種準備的 density matrix 依機率加權。

```text
ρ_bell = 1/2 [1 0 0 1]       ρ_mix = 1/2 [1 0 0 0]
             [0 0 0 0]                   [0 0 0 0]
             [0 0 0 0]                   [0 0 0 0]
             [1 0 0 1]                   [0 0 0 1]
```

兩者對角線相同，Z 基底機率自然相同；Bell state 還有連結 `00` 與 `11` 的 off-diagonal coherence。

把兩個 Qubit 都先施加 H，再做 Z 量測，就相當於量 X 基底。下面的分布可由本日矩陣實作直接計算，順序皆為 `[00, 01, 10, 11]`：

| 狀態 | Z 基底分布 | X 基底分布 |
|---|---|---|
| Product `|++⟩` | `[0.25, 0.25, 0.25, 0.25]` | `[1, 0, 0, 0]` |
| Bell `|Φ+⟩` | `[0.5, 0, 0, 0.5]` | `[0.5, 0, 0, 0.5]` |
| 經典 mixture | `[0.5, 0, 0, 0.5]` | `[0.25, 0.25, 0.25, 0.25]` |

X 基底的 0 代表 `+`，1 代表 `−`。這組比較區分了本日三個已知準備；它不是 device-independent Bell inequality 實驗，也不由此宣稱量子優勢。

CUDA-Q 版 mixture 用 NumPy 的 binomial 抽樣決定本批 shots 中準備 `00`／`11` 的數量，再分批交給 CUDA-Q 量測。這等價於各 shot 獨立丟硬幣的聚合 counts，並未把 mixture 假裝成 Bell 純態。

## 9. 實驗設計與執行

| 欄位 | 設定 |
|---|---|
| Qubits | 2 |
| States | product、bell、mixture |
| Bell preparation | H(q0) → CNOT(q0, q1)，準備深度 2 |
| Measurement | Z、X；X 額外一層 H⊗H |
| Shots | 100、1,000、10,000 |
| Seeds | 42–46 |
| Noise | 無 |
| Backends | numpy、qpp-cpu、nvidia |
| Metrics | counts、相同結果比例、correlation、total variation distance |

```bash
python articles/day04/experiment.py --backend numpy
python articles/day04/experiment.py --backend qpp-cpu
python articles/day04/experiment.py --backend nvidia
```

每個 backend 獨立寫入 `results/day04/<backend>/`，不會互相覆蓋。重跑同一 backend 則更新其結果；可用 `--output-dir /tmp/day04-check` 指定另存位置。

`correlation = P(00) + P(11) − P(01) − P(10)`，對應本次基底的兩個 ±1 量測結果乘積平均；`TVD = 1/2 Σ|p_sample − p_exact|` 衡量整體分布誤差。JSON 保存各設定五次執行的平均 TVD、樣本標準差與平均 correlation。`p_equal` 是相同結果的比例，不是 entanglement score。

執行時間包含初次 JIT／runtime 初始化。只有兩個 Qubit，資料不能用來判定 GPU 加速優勢。

## 10. 驗證與結果紀錄

```bash
python -m unittest discover -s articles/day04 -p 'test_*.py' -v
DAY04_TARGET=nvidia python -m unittest discover -s articles/day04 -p 'test_cudaq_bell.py' -v
```

第一個指令包含 NumPy 與 CUDA-Q CPU 測試；第二個明確切換 GPU。若只有 Day 3 的 NumPy 環境，可先執行 `-p 'test_bell_state.py'` 與 NumPy 實驗。

測試核對 CNOT truth table、Bell amplitudes、reduced state、density matrix 合法性、Z／X 分布、shots 總數、固定 seed 重跑與量測位元順序。CUDA-Q 的 `get_state` 僅用於模擬器的數值驗證，不代表 QPU 能直接讀出完整 state vector。[D3]

本次環境與結果摘要見 [ENVIRONMENT.md](ENVIRONMENT.md)；完整數據見 [results/day04](../../results/day04/)。

## 11. 今天容易混淆的地方

- Superposition 不必然是 entanglement，例如 `|++⟩`。
- CNOT 不是量測 control 後執行一般 Python `if`。
- `00`／`11` 相關性也可能由經典混合產生。
- 單一 Qubit 的 50%／50% 分布不能描述完整兩 Qubit 狀態。
- 不同框架可能有不同的向量與位元順序，Bell state 本身對稱，必須加非對稱案例驗證。
- Seed 用來追溯重跑，不是要求不同抽樣器輸出逐筆相同。
- GPU quantum simulation 仍是經典硬體上的模擬。

## 12. 本日來源

- [D1] [NVIDIA CUDA-Q Quick Start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)：安裝與 H + controlled-X Bell 範例。
- [D3] [NVIDIA Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)：`sample`、`get_state` 與模擬器／硬體執行差異。
- [D5] [NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)：`explicit_measurements` 的量測順序語意。
- [F2] [IBM Quantum Learning — Multiple systems and reduced states](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/multiple-systems)：separable mixture、entanglement 與 reduced state。

查閱日期：2026-09-06。官方教學用於基礎概念與 API；表格的數值由本日代碼與測試交叉驗證。共用索引見 [REFERENCES.md](../../REFERENCES.md)。

## 13. 下一篇

[Day 05](../day05/README.md) 會把 state preparation、gate、measurement 與 classical post-processing 串成完整 pipeline，整理 Notebook 與架構圖，完成第一個 Quantum Fundamentals milestone。
