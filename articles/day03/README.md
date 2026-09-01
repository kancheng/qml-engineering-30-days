# Day 03｜Quantum Gate：量子世界的 Layer？

Day 2 把 Qubit 寫成 normalized complex vector：

```text
      [α]
|ψ⟩ = [ ]
      [β]
```

今天開始真正動手。只用 Surface Pro 7、Python 和 NumPy，把 X、Y、Z、Hadamard、RX、RY、RZ 寫成矩陣，實際計算 gate 如何改變 state vector。

標題把 Quantum Gate 類比成 neural network layer，是為了提供入口，不代表兩者等價。Quantum Gate 受到 unitary constraint；一般 neural network layer 不必 unitary，也可以改變維度或丟失資訊。

## 1. 今天要解決什麼問題？

完成今天內容後，應該能回答：

1. Quantum Gate 為什麼用矩陣表示？
2. Unitary 到底限制了什麼？
3. X、Y、Z、H 分別如何改變 amplitude 與 phase？
4. RX、RY、RZ 的參數 `θ` 為什麼會成為 QML 的 trainable parameter？
5. 為什麼只看 Z-basis probability 可能看不出 gate 做了什麼？
6. NumPy state-vector simulation 和真正 QPU 有何不同？

今天的交付物包括程式、自動測試，以及旋轉閘掃描產生的 CSV／JSON 實驗資料。

## 2. Quantum Gate 是受限制的線性轉換

對 state `|ψ⟩` 施加 gate `U`：

```text
|ψ′⟩ = U|ψ⟩
```

合法的 closed-system quantum gate 必須是 unitary matrix：

```text
U†U = UU† = I
```

`U†` 是 conjugate transpose。Unitary 帶來三個重要結果：

- 保持 state norm，所以 probabilities 加總仍為 1。
- 保持 inner product，因此也保持兩個 pure states 間的幾何關係。
- 存在 inverse，而且 `U⁻¹ = U†`。

這裡刻意限定「closed-system gate」。Measurement、noise channel 和 reset 不能直接全部當成相同形式的 unitary single-system evolution；後續會分開處理。

## 3. Pauli X：Quantum NOT，但不只是一個布林 NOT

```text
    [0  1]
X = [    ]
    [1  0]
```

作用在 computational basis：

```text
X|0⟩ = |1⟩
X|1⟩ = |0⟩
```

所以 X 常被稱為 quantum NOT。更完整地說，它會交換任意 state 的兩個 amplitudes：

```text
  [α]   [β]
X [ ] = [ ]
  [β]   [α]
```

在 Bloch sphere 上，X 對應繞 x axis 旋轉 `π`，但可能伴隨不影響物理預測的 global phase，取決於我們拿它和 `RX(π)` 如何比較。

## 4. Pauli Y：交換 basis，同時加入 complex phase

```text
    [0  -i]
Y = [     ]
    [i   0]
```

```text
Y|0⟩ = i|1⟩
Y|1⟩ = -i|0⟩
```

若只量測 `Y|0⟩` 的 Z-basis probability，結果和 `X|0⟩` 一樣都是 100% 得到 1；但兩個 state vector 不完全相同。這再次提醒我們：probability distribution 不足以描述所有 phase 資訊。

## 5. Pauli Z：不改 Z-basis probability，也可能改變未來結果

```text
    [1   0]
Z = [     ]
    [0  -1]
```

```text
Z|0⟩ = |0⟩
Z|1⟩ = -|1⟩
```

如果輸入是 `|1⟩`，負號只是整個 state 的 global phase；但若輸入是 superposition：

```text
          |0⟩ + |1⟩             |0⟩ - |1⟩
Z|+⟩ = Z ─────────── = |−⟩ = ───────────
               √2                    √2
```

這時 Z 改變的是兩個 basis components 的 relative phase。立刻量 Z basis 仍是 50%／50%，但再經 Hadamard 後，`|+⟩` 與 `|−⟩` 會分別成為 `|0⟩` 與 `|1⟩`。

## 6. Hadamard：建立與解除特定 Superposition

```text
        [1   1]
H = 1/√2 [     ]
        [1  -1]
```

```text
H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
H|1⟩ = |−⟩ = (|0⟩ - |1⟩)/√2
```

Hadamard 不只是「製造隨機」。它是 deterministic unitary transformation；50%／50% 出現在對輸出做 Z-basis measurement 時。再施加一次 H：

```text
H² = I
H|+⟩ = |0⟩
H|−⟩ = |1⟩
```

這個例子同時展示 interference：amplitudes 可以相加或抵消。

## 7. Rotation Gates：QML 可訓練參數的入口

Pauli rotation 定義為：

```text
RX(θ) = exp(-iθX/2) = cos(θ/2)I - i sin(θ/2)X
RY(θ) = exp(-iθY/2) = cos(θ/2)I - i sin(θ/2)Y
RZ(θ) = exp(-iθZ/2) = cos(θ/2)I - i sin(θ/2)Z
```

展開成矩陣：

```text
        [ cos(θ/2)   -i sin(θ/2)]
RX(θ) = [                         ]
        [-i sin(θ/2)   cos(θ/2)  ]

        [cos(θ/2)  -sin(θ/2)]
RY(θ) = [                      ]
        [sin(θ/2)   cos(θ/2)]

        [e^(-iθ/2)      0     ]
RZ(θ) = [                      ]
        [    0       e^(iθ/2) ]
```

角度使用 `θ/2`，是因為 SU(2) 對三維 rotation 的表示具有 half-angle 結構。工程上先記住官方框架的 rotation gate 通常也採這個 convention；不要自行假設 `cos(θ)`。

Mitarai et al. 的 Quantum Circuit Learning 使用參數化量子電路，讓 classical computer 迭代調整 circuit parameters，構成 hybrid learning framework。[P2] 今天還不做 optimizer，但 `RX(θ)`、`RY(θ)`、`RZ(θ)` 已經是後面 Ansatz 的基本積木。

## 8. 實作：只用 NumPy 建立 Gate Simulator

核心程式位於 [quantum_gates.py](quantum_gates.py)。以 X gate 為例：

```python
import numpy as np

ket_zero = np.array([1, 0], dtype=np.complex128)
x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
ket_one = x_gate @ ket_zero
```

正式版本另外檢查：

- gate shape 是否為 `(2, 2)`；
- `U†U` 是否接近 identity；
- input state shape 是否為 `(2,)`；
- state 是否 normalized；
- expectation value 對 Hermitian observable 是否為實數。

執行完整實驗：

```bash
python articles/day03/experiment.py
```

執行 correctness tests：

```bash
python -m unittest discover -s articles/day03 -p "test_*.py" -v
```

不需要 Ubuntu、CUDA-Q 或 GPU；Surface Pro 7 即可。

## 9. Experiment Design

### 研究問題

`RX(θ)`、`RY(θ)`、`RZ(θ)` 作用於 `|0⟩` 時，Z-basis probabilities 與 Bloch components 如何變化？

### 設定

| 欄位 | 設定 |
|---|---|
| Runtime | Python + NumPy，CPU state-vector simulation |
| Input | `|0⟩` |
| Gates | RX、RY、RZ |
| Angles | `0, π/4, π/2, 3π/4, π` |
| Measurement | exact state-vector probabilities，沒有 finite shots |
| Observables | X、Y、Z expectation values |
| Random seed | 不需要；本實驗沒有隨機抽樣 |
| Output | `results/day03/gate_sweep.csv`、`summary.json` |

這不是 QPU experiment，也不是 CUDA-Q benchmark。它是 deterministic linear algebra experiment，目的是驗證定義與程式實作一致。

## 10. Result

完整資料在 [gate_sweep.csv](../../results/day03/gate_sweep.csv)，摘要在 [summary.json](../../results/day03/summary.json)。關鍵結果：

| Gate / Input | `P(0)` | `P(1)` | 解讀 |
|---|---:|---:|---|
| `X|0⟩` | 0 | 1 | basis state 翻轉 |
| `H|0⟩` | 0.5 | 0.5 | 產生 `|+⟩` |
| `RY(π/2)|0⟩` | 0.5 | 0.5 | 轉到 Bloch sphere 赤道 |
| `RY(π)|0⟩` | 0 | 1 | 到達 `|1⟩`，允許 global phase 差異 |
| `RZ(θ)|0⟩` | 1 | 0 | `|0⟩` 是 Z eigenstate，Z probability 不變 |

所有 15 筆 rotation records 的 norm error 都只在 floating-point tolerance 內；自動測試也驗證 X／Y／Z 的 basis action、`H² = I`、rotation endpoints 和 non-unitary rejection。

## 11. 結果怎麼解讀？

對 `|0⟩` 而言：

```text
RX(θ): P(1) = sin²(θ/2)，同時產生 imaginary relative phase
RY(θ): P(1) = sin²(θ/2)，amplitudes 可保持為 real
RZ(θ): P(1) = 0，只改變 |0⟩ 的 phase
```

RX 和 RY 在這個特定 input、特定 measurement basis 下得到相同 probabilities，不代表兩者是同一個 gate。CSV 中的 X／Y／Z expectation values 能看出它們在 Bloch sphere 上沿不同方向移動。

RZ 對 `|0⟩` 的 Z probability 完全不變，也不代表 RZ 沒有作用。若輸入換成 superposition，RZ 改變的 relative phase 可以被後續 gate 轉成可量測差異。QML circuit 的效果取決於 **input state、gate sequence、observable**，不能只孤立看某個 gate。

## 12. Quantum Gate 是 Neural Network Layer 嗎？

這個類比只在「一連串可組合 transformation」的層次有幫助：

```text
Neural Network: x → Layer₁ → activation → Layer₂ → output
Quantum Circuit: |ψ⟩ → U₁(θ₁) → U₂(θ₂) → measurement
```

根本差異包括：

- Gate 必須 unitary；一般 layer 不必。
- Gate 對 closed state evolution 可逆；ReLU、pooling 等通常不可逆。
- Quantum state 不能直接完整讀出；輸出需透過 measurement。
- 沒有 measurement／reset 時，gate 不會任意丟棄資訊。
- QML 的 nonlinearity 通常不是在 state evolution 中插入 ReLU，而是由 encoding、measurement、loss 與 classical optimization 的整體模型產生。

所以「Quantum Gate 是量子版 Layer」適合作為提問，不適合作為正式定義。

## 13. 今天踩到的坑

- 把 `U.T` 當成 `U†`；有 complex entries 時必須 conjugate transpose。
- rotation matrix 使用 `θ` 而不是 `θ/2`。
- 看見 `RX(π)|0⟩ = -i|1⟩` 就判定錯誤；它和 `|1⟩` 只差 global phase。
- 認為 Hadamard 本身產生 classical randomness；隨機結果出現在 measurement。
- 只比較 Z-basis probabilities，忽略 relative phase 與其他 observables。
- 把 NumPy state-vector simulation 說成量子硬體實驗。
- 以為所有 quantum operations 都是 unitary gate，忽略 measurement 與 noise channel。

## 14. 內容與來源核對

本日矩陣定義、unitarity、single-qubit gate 與 circuit model 以 Nielsen–Chuang 標準教材為基礎。[F1] Cambridge 的目錄也明確包含 quantum circuits、single-qubit operations、controlled operations、measurement 與 universal gates。

Parameterized circuit 與 QML 的連接引用 Mitarai et al. 2018 的原始論文，而非二手教學；APS 的 version of record 列出題名、四位作者、*Physical Review A* 98、032309、發表日 2018-09-10 與 DOI。[P2]

程式正確性不只靠引用：本專案以 unit tests 驗證矩陣性質，再輸出 machine-readable experiment data。文獻說明定義與研究脈絡；測試驗證這份程式是否符合那些定義。

## 15. 本日文獻與資源

- [F1] Michael A. Nielsen and Isaac L. Chuang, *Quantum Computation and Quantum Information: 10th Anniversary Edition*, Cambridge University Press (2010), [publisher page](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE).
- [P2] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, “Quantum circuit learning,” *Physical Review A* 98, 032309 (2018), [DOI](https://doi.org/10.1103/PhysRevA.98.032309).
- [文獻與資源索引](../../REFERENCES.md)：集中記錄版本狀態、核對內容與引用原則。

## 16. 今日 GitHub Commit

```text
feat: implement and verify single-qubit gates with NumPy
```

## 17. 下一篇

Day 04 將進入兩個 Qubit：用 tensor product 表示 `|00⟩`，以 H + CNOT 建立 Bell state，重複 measurement 觀察 `00`／`11` correlation。是否直接使用 CUDA-Q，會先依 Ubuntu 筆電的實際環境檢查結果決定；即使環境尚未完成，也保留 NumPy fallback，確保文章可重現。
