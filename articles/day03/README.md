# Day 03｜量子閘如何改變量子狀態？

[Day2](../day02/README.md) 介紹如何用向量表示單一量子位元的純態，並區分機率振幅、量測機率與相對相位。Day3 接著探討如何改變量子狀態，透過 NumPy 矩陣運算，觀察量子閘對狀態與量測機率的影響。

**量子閘是改變量子狀態的基本操作，串接起來就形成量子電路。** 理解每一步如何改變振幅與量測機率，才能進一步建立量子機器學習（Quantum Machine Learning，QML）模型。本章以單一量子位元為範圍，把 X、Y、Z、Hadamard 與旋轉閘寫成矩陣，再用「矩陣乘上狀態向量」計算結果。核心概念是么正性（unitarity，也稱酉性）：本章討論的理想量子閘會保持向量長度，使量測機率總和維持為 1，且運算可以反轉。觀察時也要留意相對相位；例如 Z 閘作用於疊加態後，當下量到 0／1 的機率可能不變，但接上 Hadamard 閘後就能看出差異。RX、RY、RZ 則加入可調整的旋轉角度，成為後續資料編碼與模型訓練的基本元件。本章先掃描角度、核對矩陣性質與保存結果，尚未進行訓練。輸入狀態、閘的順序與量測方式需要一起考慮；同一個閘放在不同位置，可能產生不同的預測結果。

[Day2](../day02/README.md) introduced vector representations of single-qubit pure states and distinguished probability amplitudes, measurement probabilities, and relative phase. Day3 uses NumPy matrix operations to examine how quantum gates change those states.

**Quantum gates are basic operations that change quantum states; connecting them forms a quantum circuit.** A matrix multiplied by a state vector describes each operation. The ideal gates covered here preserve vector length, so measurement probabilities continue to sum to one, and their operations can be reversed. X, Y, Z, and Hadamard (H) illustrate how amplitudes and relative phase change. A Z gate can leave immediate probabilities for 0 and 1 unchanged, while a subsequent H gate reveals the difference. Rotation gates RX, RY, and RZ add adjustable angles that later support data encoding and model training. This chapter evaluates several angles and checks the results rather than training a model. The input state, gate order, and measurement basis must be considered together to understand the output.

---

Day 2 把量子位元寫成已正規化的複數向量，也就是長度為 1、分量可含虛數的向量：

```text
      [α]
|ψ⟩ = [ ]
      [β]
```

今天開始真正動手。在 Ubuntu 的專案獨立 Python 環境中，只用 NumPy，把 X、Y、Z、Hadamard、RX、RY、RZ 寫成矩陣，實際計算閘如何改變狀態向量。相同程式也可在 Surface Pro 7 的 CPU 上執行。

## 1. 從向量到矩陣運算

NumPy 是 Python 的數值運算套件。本章以矩陣，也就是按列與欄排列的數值表，計算量子閘如何作用於狀態向量。單一量子位元的狀態有兩個振幅，因此這裡的量子閘使用 2 × 2 矩陣。

以矩陣乘上向量時，每一列會分別乘上對應的向量分量，再將結果相加，形成新的分量。振幅是計算機率的複數係數；取絕對值平方，才得到對應量測結果的機率。

本章會保存程式、自動測試與角度掃描結果。「掃描」是依序代入多個角度觀察變化，並不是根據預測誤差自動調整參數的訓練。

## 2. 量子閘是受限制的線性轉換

對狀態 `|ψ⟩` 施加閘 `U`：

```text
|ψ′⟩ = U|ψ⟩
```

封閉系統指理想上沒有與外界交換資訊或受到干擾的系統。這裡的理想量子閘用么正矩陣表示：

```text
U†U = UU† = I
```

`U†` 是共軛轉置：先交換矩陣的列與欄，再將每個複數的虛部反號。`I` 是單位矩陣，乘上它不會改變向量。么正矩陣（unitary matrix）滿足上式，表示先執行 `U`，再執行 `U†`，可以還原輸入。么正性帶來三個重要結果：

- 保持狀態向量長度，所以機率加總仍為 1。
- 保持內積，因此也保持兩個純態間的幾何關係。
- 存在反向運算，而且 `U⁻¹ = U†`。

內積是將對應分量相乘再相加的運算，對複數向量需先將第一個向量取共軛；它能描述兩個狀態的重疊關係。

雜訊通道描述干擾如何改變狀態，重設則是將量子位元準備回指定狀態。量測、雜訊通道和重設不能直接全部當成相同形式的么正演化；後續會分開處理。

## 3. Pauli X：交換兩個振幅

```text
    [0  1]
X = [    ]
    [1  0]
```

X、Y、Z 合稱 Pauli 閘，是以物理學家 Pauli 命名的三個基本操作。計算基底是用 `|0⟩` 與 `|1⟩` 描述狀態的方式，也稱 Z 基底。X 作用在這兩個狀態時：

```text
X|0⟩ = |1⟩
X|1⟩ = |0⟩
```

X 會將 0 與 1 對調，類似一般邏輯中的反相操作（NOT）。更完整地說，它會交換任意狀態的兩個振幅：

```text
  [α]   [β]
X [ ] = [ ]
  [β]   [α]
```

在布洛赫球上，X 對應繞 x 軸旋轉 `π`；與 `RX(π)` 比較時，兩者只差不影響物理預測的整體相位。

## 4. Pauli Y：交換基底，同時加入複數相位

```text
    [0  -i]
Y = [     ]
    [i   0]
```

```text
Y|0⟩ = i|1⟩
Y|1⟩ = -i|0⟩
```

若只量測 `Y|0⟩` 的 Z 基底機率，結果和 `X|0⟩` 一樣都是 100% 得到 1；兩個狀態向量相差整體相位，因此代表相同的物理狀態。相對相位造成的可觀測差異，則在下一節以 Z 作用於疊加態的例子說明。

## 5. Pauli Z：不改 Z 基底機率，也可能改變未來結果

```text
    [1   0]
Z = [     ]
    [0  -1]
```

```text
Z|0⟩ = |0⟩
Z|1⟩ = -|1⟩
```

如果輸入是 `|1⟩`，負號只是整個狀態的整體相位；但若輸入是疊加態：

```text
          |0⟩ + |1⟩             |0⟩ - |1⟩
Z|+⟩ = Z ─────────── = |−⟩ = ───────────
               √2                    √2
```

這時 Z 改變的是兩個基底分量的相對相位。立刻量 Z 基底仍是 50%／50%，但再經 Hadamard 後，`|+⟩` 與 `|−⟩` 會分別成為 `|0⟩` 與 `|1⟩`。

## 6. Hadamard：建立與解除特定疊加態

```text
        [1   1]
H = 1/√2 [     ]
        [1  -1]
```

```text
H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
H|1⟩ = |−⟩ = (|0⟩ - |1⟩)/√2
```

Hadamard 不只是「製造隨機」。它是確定的么正轉換；50%／50% 出現在對輸出做 Z 基底量測時。再施加一次 H：

```text
H² = I
H|+⟩ = |0⟩
H|−⟩ = |1⟩
```

這個例子同時展示干涉：振幅可以相加或抵消。

## 7. 旋轉閘：QML 可訓練參數的入口

Pauli 旋轉閘定義為：

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

`θ` 是希臘字母 theta，表示旋轉角度，單位為弧度：`π` 是半圈，`2π` 是一圈。`sin`、`cos` 是三角函數，`exp` 表示指數運算。以上公式使用半角 `θ/2`：例如 `RY(π)` 會將 `|0⟩` 轉成 `|1⟩`，代入矩陣時使用的是 `sin(π/2)` 與 `cos(π/2)`。布洛赫球上的旋轉角度與振幅公式中的角度並不相同。

Mitarai 等人的研究使用參數化量子電路，也就是包含可調整參數的電路，讓一般電腦迭代調整電路參數，構成量子與經典混合學習流程。[P2] 最佳化器是依誤差更新參數的方法，本章尚未進行這個步驟； `RX(θ)`、`RY(θ)`、`RZ(θ)` 已經是後續電路模板（預先安排的閘組合）的基本元件。

## 8. 實作：只用 NumPy 建立量子閘模擬器

### 建立獨立示範環境

以下指令在專案根目錄執行。使用 Python 3.12 建立 `.venv`，即使目前終端顯示 Conda `(base)`，套件也會安裝到專案的虛擬環境。`.venv` 不納入 Git。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-day03.txt
python -c "import sys, numpy; print(sys.executable); print(numpy.__version__)"
```

最後應顯示專案內 `.venv/bin/python` 與 `2.2.6`。之後開新終端只需重新 `source .venv/bin/activate`；離開時使用 `deactivate`。程式編輯器的 Python 直譯器也選取此 `.venv/bin/python`。

虛擬環境是替專案分開保存 Python 與套件的資料夾，避免不同專案的版本互相影響。Git 是記錄檔案修改歷史的工具；套件安裝資料夾通常不放入版本紀錄。CPU 是一般電腦的中央處理器，這裡以它執行數值模擬；CUDA-Q 則是後續使用的量子程式開發工具。

本日依賴固定在 [requirements-day03.txt](../../requirements-day03.txt)。設備資訊另見 [環境說明](ENVIRONMENT.md)；本日使用 CPU，不需安裝 CUDA-Q。

### 範例一：直接觀察閘輸出

```bash
python articles/day03/demo.py
```

可執行的完整代碼在 [demo.py](demo.py)，預期輸出：

```text
X|0>           P(0)=0.000000 P(1)=1.000000
H|0>           P(0)=0.500000 P(1)=0.500000
HH|0>          P(0)=1.000000 P(1)=0.000000
HZH|0>         P(0)=0.000000 P(1)=1.000000
RY(pi/2)|0>    P(0)=0.500000 P(1)=0.500000
```

電路由右往左作用：`HZH|0⟩` 先 H，再 Z，最後 H。中間的 Z 不改變當下的 Z 基底機率，卻改變相對相位，因此最後得到 1；直接做 `HH|0⟩` 則得到 0。

### 範例二：矩陣乘法

核心程式位於 [quantum_gates.py](quantum_gates.py)。以 X 閘為例：

```python
import numpy as np

ket_zero = np.array([1, 0], dtype=np.complex128)
x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
ket_one = x_gate @ ket_zero
```

程式中的 `@` 是矩陣乘法，`np.complex128` 用實部與虛部各 64 位元儲存複數。`(2, 2)` 表示兩列兩欄，`(2,)` 表示含有兩個分量的一維陣列。

可觀測量（observable）是量測所關心的物理量，在此以矩陣表示。厄米矩陣（Hermitian matrix）等於自己的共軛轉置，能確保期望值為實數。期望值是依機率計算的平均值，例如 Z 的期望值是 `P(0) − P(1)`。X、Y、Z 的期望值也正是布洛赫球的三個座標分量。

正式版本另外檢查：

- 閘矩陣的形狀是否為 `(2, 2)`；
- `U†U` 是否接近單位矩陣；
- 輸入狀態的形狀是否為 `(2,)`；
- 狀態是否已正規化；
- 期望值對厄米可觀測量是否為實數。

### 範例三：掃描旋轉角度並保存實驗

完整代碼在 [experiment.py](experiment.py)，核心迴圈如下（使用本日 `quantum_gates.py`）：

```python
import numpy as np
from quantum_gates import KET_ZERO, apply_gate, probabilities, rx, ry, rz

for name, gate in (("RX", rx), ("RY", ry), ("RZ", rz)):
    for theta in np.linspace(0, np.pi, 5):
        state = apply_gate(gate(float(theta)), KET_ZERO)
        print(name, theta, probabilities(state))
```

從專案根目錄執行完整實驗：

```bash
python articles/day03/experiment.py
```

執行正確性測試：

```bash
python -m unittest discover -s articles/day03 -p "test_*.py" -v
```

執行實驗會更新 `results/day03/gate_sweep.csv` 與 `summary.json`；摘要同時保存實際 Python、NumPy、作業系統、架構與執行後端。CSV 是以列與欄保存數值的文字格式，JSON 則以欄位名稱保存結構化資料。執行後端指實際負責運算的程式或硬體。浮點數是電腦以有限位數儲存小數的方式，容差是比較數值時允許的微小誤差。不同環境可能有微小浮點差異，比較數值時使用容差，不要求檔案逐字相同。

## 9. 實驗設計

### 研究問題

`RX(θ)`、`RY(θ)`、`RZ(θ)` 作用於 `|0⟩` 時，Z 基底機率與布洛赫球座標分量如何變化？

### 設定

| 欄位 | 設定 |
|---|---|
| 執行環境 | Python + NumPy，CPU 狀態向量模擬 |
| 輸入 | `|0⟩` |
| 量子閘 | RX、RY、RZ |
| 角度 | `0, π/4, π/2, 3π/4, π` |
| 量測 | 由狀態向量直接計算的機率，沒有有限次量測抽樣 |
| 可觀測量 | X、Y、Z 期望值 |
| 隨機種子 | 不需要；本實驗沒有隨機抽樣 |
| 輸出 | `results/day03/gate_sweep.csv`、`summary.json` |

這不是量子硬體實驗，也不是 CUDA-Q 效能評測。它是沒有隨機抽樣的矩陣運算實驗，目的是驗證定義與程式實作一致。

## 10. 結果

完整資料在 [gate_sweep.csv](../../results/day03/gate_sweep.csv)，摘要在 [summary.json](../../results/day03/summary.json)。關鍵結果：

| 量子閘／輸入 | `P(0)` | `P(1)` | 解讀 |
|---|---:|---:|---|
| `X|0⟩` | 0 | 1 | 基底狀態翻轉 |
| `H|0⟩` | 0.5 | 0.5 | 產生 `|+⟩` |
| `RY(π/2)|0⟩` | 0.5 | 0.5 | 轉到布洛赫球赤道 |
| `RY(π)|0⟩` | 0 | 1 | 到達 `|1⟩`，允許整體相位差異 |
| `RZ(θ)|0⟩` | 1 | 0 | `|0⟩` 是 Z 的本徵態（作用後只是原向量乘上一個數），Z 機率不變 |

所有 15 筆旋轉閘紀錄的向量長度誤差都只在浮點數容差內；自動測試也驗證 X／Y／Z 的基底作用、`H² = I`、旋轉角度端點和拒絕非么正矩陣。

## 11. 結果怎麼解讀？

對 `|0⟩` 而言：

```text
RX(θ): P(1) = sin²(θ/2)，振幅可含虛數，相對相位取決於角度
RY(θ): P(1) = sin²(θ/2)，振幅可保持為實數
RZ(θ): P(1) = 0，只改變 |0⟩ 的整體相位
```

RX 和 RY 在這個特定輸入、特定量測基底下得到相同機率，不代表兩者是同一個閘。CSV 中的 X／Y／Z 期望值能看出它們在布洛赫球上沿不同方向移動。

RZ 對 `|0⟩` 的 Z 機率完全不變，也不代表 RZ 沒有作用。若輸入換成疊加態，RZ 改變的相對相位可以被後續閘轉成可量測差異。QML 電路的效果取決於 **輸入狀態、閘的順序、可觀測量**，不能只孤立看某個閘。

## 12. 量子閘和神經網路層有什麼不同？

神經網路層（neural network layer）是一般機器學習模型中的一段計算，將輸入轉換後交給下一層。量子閘也能依序串接，但理想量子閘保持向量長度與內積，一般神經網路層沒有這項限制。

例如，常見的 ReLU 函數會把負數改成 0，保留正數。不同負數經過它都變成 0，因此無法單靠輸出還原原值。理想量子閘則可以用反向操作還原狀態。

量子閘對狀態向量的運算是線性的，也就是保留向量加法與倍數關係。不過，將資料轉成角度、取振幅絕對值平方計算機率，再結合一般電腦的計算後，整個模型的輸入與預測不必是線性關係。

因此，串接量子閘與串接神經網路層有流程上的相似之處，但不能直接視為相同的運算元件。

## 13. 常見誤解

- 把 `U.T` 當成 `U†`；有複數元素時必須共軛轉置。
- 旋轉矩陣使用 `θ` 而不是 `θ/2`。
- 看見 `RX(π)|0⟩ = -i|1⟩` 就判定錯誤；它和 `|1⟩` 只差整體相位。
- 認為 Hadamard 本身產生一般隨機性；隨機結果出現在量測。
- 只比較 Z 基底機率，忽略相對相位與其他可觀測量。
- 把 NumPy 狀態向量模擬說成量子硬體實驗。
- 以為所有量子操作都是么正閘，忽略量測與雜訊通道。

## 14. 定義與程式如何核對？

矩陣定義與單量子位元運算以 Nielsen 與 Chuang 的教材為依據。[F1] 參數化電路與 QML 的連接則參考 Mitarai 等人的研究：由一般電腦反覆調整電路參數，讓量子電路參與學習流程。[P2]

自動測試檢查矩陣是否符合定義，保存的實驗資料則記錄指定輸入下的結果。兩者用途不同，也都不等於真實量子硬體的執行證據。

## 15. 本日文獻與資源

- [F1] Michael A. Nielsen and Isaac L. Chuang, *Quantum Computation and Quantum Information: 10th Anniversary Edition*, Cambridge University Press (2010), [publisher page](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE).
- [P2] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, “Quantum circuit learning,” *Physical Review A* 98, 032309 (2018), [DOI](https://doi.org/10.1103/PhysRevA.98.032309).
- [文獻與資源索引](../../REFERENCES.md)：集中記錄版本狀態、核對內容與引用原則。

## 16. 下一篇

[Day 04](../day04/README.md) 將進入兩個量子位元：用張量積，也就是組合兩個狀態向量的運算，表示 `|00⟩`，再用 H 閘與受控反相閘（CNOT）建立貝爾態。CNOT 依第一個位元決定是否翻轉第二個位元；貝爾態則是一種無法拆成兩個獨立純態的聯合狀態。後續會透過不同量測方式，觀察這種關係與一般機率混合的差異。
