# Day 02｜從 Bit 到 Qubit：工程師需要懂多少量子力學？

## 本章摘要｜初學者學習筆記

### 中文

[Day1](../day01/README.md) 說明量子計算在機器學習流程中的位置，以及為什麼需要透過實驗與傳統方法比較。Day2 接著建立描述量子資料的基本語言，從熟悉的 0／1 出發，理解 qubit（量子位元）的狀態與量測結果。

這一章目標在於看懂 **「量子狀態如何表示，以及如何從狀態算出量測機率」**，為後續量子閘運算與資料編碼打下基礎。本章先以單一量子位元的純態為範圍，把 `|ψ⟩ = α|0⟩ + β|1⟩` 對應成兩個複數組成的向量：`α`、`β` 稱為機率振幅，取絕對值平方後，才是在 0／1 基底下量到各結果的機率；兩個機率相加必須等於 1。另一個重點是相對相位，也就是兩個振幅之間的相位差：即使兩個狀態量到 0／1 的機率相同，經過後續量子閘運算，仍可能產生不同結果。Bloch Sphere（布洛赫球）提供這些單一量子位元純態的幾何圖像，NumPy 小實驗則用來核對向量長度、理論機率與重複抽樣的結果。讀完本章，應能辨認振幅與機率的差別，理解一次量測只得到一個結果，以及為什麼需要重新準備相同狀態、重複量測，才能估計機率分布。

### English

[Day1](../day01/README.md) explained where quantum computation fits into a machine learning workflow and why experiments need comparisons with classical methods. Day2 builds the basic language for describing quantum data, starting from familiar bits to explain qubit states and measurement outcomes.

This chapter aims to explain **how a quantum state is represented and how measurement probabilities are calculated from that state**, laying the foundation for quantum gates and data encoding. The scope is a single qubit in a pure state. The expression `|ψ⟩ = α|0⟩ + β|1⟩` corresponds to a vector containing two complex numbers. These numbers, called probability amplitudes, give the probabilities of measuring 0 and 1 in the computational basis through their squared magnitudes; the two probabilities must sum to 1. Another key concept is relative phase—the phase difference between the amplitudes. Two states with identical probabilities for 0 and 1 can still produce different outcomes after further quantum gates. The Bloch sphere provides a geometric picture of these single-qubit pure states, while a small NumPy experiment checks the vector norm, theoretical probabilities, and repeated sampling results. The learning goal is to distinguish amplitudes from probabilities, understand that a single measurement yields one outcome, and explain why estimating a probability distribution requires repeated preparation and measurement of the same state.

---

看到 `|ψ⟩ = α|0⟩ + β|1⟩`，很多工程師的第一反應不是好奇，而是先被符號勸退。

其實今天需要的數學不多。先把 Qubit 當成一個受到特殊規則約束的二維向量，就能讀懂後續 Quantum Gate、Circuit、Measurement 與 QML Encoding 的基本表示。

## 1. 今天要解決什麼問題？

今天只建立六個概念：Classical Bit、Qubit、State Vector、Dirac Notation、Probability Amplitude 與 Bloch Sphere。完成後，至少要能回答：

- `|0⟩` 和數字 0 是否相同？
- `α`、`β` 是機率，還是別的東西？
- 為什麼 `|α|² + |β|² = 1`？
- Qubit 在 Measurement 前後有什麼差異？
- Bloch Sphere 描述的是什麼？

## 2. Classical Bit：只能讀到 0 或 1

Classical Bit 的狀態可以寫成：

```text
b ∈ {0, 1}
```

在電腦裡，Bit 可能由電壓、電荷或磁性實作，但在程式層，通常只關心數值是 0 還是 1。多個 Bit 組合後可表示更多狀態，例如兩個 Bit 有 `00`、`01`、`10`、`11` 四種可能。

如果系統此刻是 `01`，它就不是同時處於其他三個狀態。這是接下來和 Qubit 最重要的差別之一。

## 3. Qubit：不是「可以同時讀出 0 和 1」

一個 Qubit 的一般狀態寫成：

```text
|ψ⟩ = α|0⟩ + β|1⟩
```

其中：

- `|ψ⟩`（讀作 ket psi）表示目前的 Quantum State。
- `|0⟩`、`|1⟩` 是 computational basis states。
- `α`、`β` 是 probability amplitudes，通常可以是複數。

這個表示叫做 Superposition，但不要把它簡化成「Measurement 時可以一次拿到 0 和 1」。在 computational basis 量測一個 Qubit，一次仍只得到 0 或 1；Superposition 描述的是量測前的狀態及其可能產生的統計結果。

## 4. Dirac Notation 與 State Vector

Dirac notation 看似陌生，其實可以直接對應線性代數：

```text
      [1]             [0]
|0⟩ = [ ]       |1⟩ = [ ]
      [0]             [1]
```

因此：

```text
|ψ⟩ = α|0⟩ + β|1⟩

      [α]
    = [ ]
      [β]
```

Ket `|ψ⟩` 是 column vector；對應的 Bra `⟨ψ|` 是 conjugate transpose：

```text
⟨ψ| = [α*  β*]
```

星號代表 complex conjugate。兩者相乘得到 inner product：

```text
⟨ψ|ψ⟩ = |α|² + |β|²
```

合法 Quantum State 必須 normalized，所以 `⟨ψ|ψ⟩ = 1`。

### 只補今天需要的複數

複數可以寫成 `a + bi`，其中 `i² = -1`。它的 complex conjugate 是 `a - bi`，絕對值平方為：

```text
|a + bi|² = (a + bi)(a - bi) = a² + b²
```

所以當 amplitude 是 complex number，`|α|²` 仍是非負實數，能成為 probability。今天不需要完整複分析，但必須知道「取平方」和「取絕對值平方」不同。

## 5. Probability Amplitude 不是 Probability

假設：

```text
          1       1
|ψ⟩ = ─────|0⟩ + ─────|1⟩
         √2      √2
```

`1/√2` 是 amplitude，不是量測機率。依 Born rule，在 computational basis 量測時：

```text
P(0) = |α|² = 1/2
P(1) = |β|² = 1/2
```

兩個機率相加為 1，正是 normalization 的原因。

再看一個帶有 complex phase 的狀態：

```text
          1       i
|ψ⟩ = ─────|0⟩ + ─────|1⟩
         √2      √2
```

在 computational basis 量測，0 與 1 仍各有 50% 機率。這不代表 phase 沒有作用：後續 Quantum Gate 會讓 amplitude 互相 interference，使相對 phase 影響可觀察結果。只看單次 basis measurement，會遺失大量 state 資訊。

## 6. Measurement：從 State 到 Classical Result

對 `|ψ⟩ = α|0⟩ + β|1⟩` 做 computational basis measurement：

```text
Quantum State ── measurement ──→ 0  with probability |α|²
                              └→ 1  with probability |β|²
```

得到 0 後，狀態會對應到 `|0⟩`；得到 1 後則對應到 `|1⟩`。若要估計機率分布，需要重新準備同一狀態並重複量測許多次。這些重複執行次數就是後面會遇到的 shots。

因此 Quantum Program 常見的輸出不是一個完全確定的答案，而是一組 counts 或由樣本估計的 expectation value。

## 7. Bloch Sphere：把一個 Qubit 畫成球面上的點

忽略無法觀察的 global phase 後，任一 pure single-qubit state 都能寫成：

```text
|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩
```

`θ` 和 `φ` 對應 Bloch Sphere 上的位置：

```text
              |0⟩
               ↑ z
               │
       |−⟩ ←── • ──→ |+⟩      x
              /│
             / │
            y  ↓
              |1⟩
```

- 北極是 `|0⟩`，南極是 `|1⟩`。
- 赤道包含 `( |0⟩ + e^(iφ)|1⟩ ) / √2` 這類等機率狀態。
- Quantum Gate 可視為對狀態向量做 rotation，但這個直覺只直接適用於單一 pure Qubit。

Bloch Sphere 的重點不是背球面座標，而是看懂兩件事：Qubit 除了 0/1 的量測機率，還有 relative phase；因此相同的 computational-basis probabilities 不代表相同 Quantum State。

### Global phase 與 relative phase

`|ψ⟩` 與 `e^(iγ)|ψ⟩` 相差 global phase，會給出相同的物理預測；但 `α|0⟩ + β|1⟩` 中兩項的 relative phase 可以在後續 interference 中影響量測。

例如 `|+⟩ = (|0⟩ + |1⟩)/√2` 和 `|−⟩ = (|0⟩ − |1⟩)/√2` 在 Z basis 量測時都各有 50% 的 0／1，卻是不同狀態。對兩者再施加 Hadamard gate，前者會變成 `|0⟩`，後者會變成 `|1⟩`。Day 3 將用矩陣乘法直接驗證。

## 8. 和 AI Engineering 有什麼關係？

從 ML 視角，可以暫時建立以下對照：

| AI / Linear Algebra | Quantum Computing |
|---|---|
| Feature vector | State vector（但必須 normalized） |
| Linear transform | Quantum gate（還必須是 unitary） |
| Model output | Measurement-derived classical value |
| Sampling variance | 有限 shots 帶來的估計誤差 |

這只是學習橋梁，不是完全等價。尤其 Quantum State 不能被任意讀出或複製；後續也會看到多 Qubit state 的維度隨 qubit 數指數成長。

## 9. 小實驗：用向量算量測機率

先不用 CUDA-Q，只以 NumPy 驗證 normalization 與 Born rule：

```python
import numpy as np

alpha = 1 / np.sqrt(2)
beta = 1j / np.sqrt(2)
state = np.array([alpha, beta], dtype=np.complex128)

norm = np.vdot(state, state).real
probabilities = np.abs(state) ** 2

print(f"norm = {norm:.1f}")
print(f"P(0) = {probabilities[0]:.1f}")
print(f"P(1) = {probabilities[1]:.1f}")
```

預期輸出：

```text
norm = 1.0
P(0) = 0.5
P(1) = 0.5
```

若把 state 改成 `[1, 1]`，機率總和會變成 2；它不是合法的 normalized Quantum State。除以向量長度後才可作為 state vector。

Repository 已提供可直接執行的 [state_vector.py](state_vector.py)：

```bash
python articles/day02/state_vector.py
```

除了 norm 和 probability，它也固定 random seed 模擬有限 shots，讓理論機率與抽樣估計可以被分開觀察。這個抽樣是 classical simulation，用來理解 measurement statistics，並不是在 QPU 上執行。

## 10. 結果怎麼解讀？

今天最重要的不是「Qubit 同時是 0 和 1」，而是以下較精確的說法：

1. Single Qubit 是二維複數 Hilbert space 中的 normalized state vector。
2. `α`、`β` 是 amplitudes，絕對值平方才是 computational-basis measurement probabilities。
3. Relative phase 會影響 interference，不能只靠 0/1 機率描述完整狀態。
4. Measurement 把 Quantum State 轉成 Classical Result；估計分布需要重複準備與量測。
5. Bloch Sphere 是 single pure Qubit 的幾何表示，不是多 Qubit 系統的完整地圖。

## 11. 今天踩到的坑

- 把 amplitude 直接當成 probability。
- 說 Qubit 可以「一次讀出 0 和 1」。
- 只檢查 `α + β = 1`，正確條件是 `|α|² + |β|² = 1`。
- 以為相同 measurement probabilities 就是相同 state，忽略 relative phase。
- 把 Bloch Sphere 當成所有 Quantum System 都能直接使用的視覺化。
- 把 global phase 和 relative phase 混在一起。
- 看見 state vector simulator 可以讀出完整向量，就誤以為真實 QPU 也能在單次 measurement 中吐出所有 amplitudes。

## 12. 從一個 Qubit 到後續 QML

今天的二維 state vector 會直接延伸到後續內容：

```text
normalized vector
      ↓
unitary gate transformation
      ↓
multi-qubit tensor product
      ↓
parameterized circuit
      ↓
measurement expectation
      ↓
QML model output
```

對 `n` 個 Qubit，pure state vector 有 `2^n` 個 complex amplitudes。這解釋了 state-vector simulation 的記憶體成本為何會快速成長，也預告 Day 27 的 CPU／GPU benchmark。但「state space 很大」本身仍不等於可用的 quantum advantage：資料載入、circuit depth、measurement 與 classical comparison 都必須計入。

## 13. 延伸資源與來源核對

- [F1] Nielsen and Chuang, *Quantum Computation and Quantum Information*, 10th Anniversary Edition, Cambridge University Press (2010), [publisher page](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE)。用於 state、Dirac notation 與 measurement 的標準背景。
- [R4] Schuld and Petruccione, *Supervised Learning with Quantum Computers*, Springer (2018), [DOI](https://doi.org/10.1007/978-3-319-96424-9)。用於把 quantum information 基礎連接到 supervised QML。
- [D1] [NVIDIA CUDA-Q Quick Start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)。官方範例以 Bell／GHZ state 和 sampling 驗證安裝，會在 Day 4 實作。

上述出版資訊已在 [文獻與資源索引](../../REFERENCES.md) 核對。Day 2 的物理敘述以標準教材為主，不用一般部落格充當定義來源。

## 14. 今日 GitHub Commit

```text
docs: explain qubits with state vectors and amplitudes
```

## 15. 下一篇

Day 03 將把 Quantum Gate 當成作用在 State Vector 上的受限線性轉換，依序認識 X、Y、Z、H、RX、RY、RZ，並開始建立第一批單量子位元 Quantum Circuit。CNOT 與兩量子位元 Bell state 留到 Day 04。
