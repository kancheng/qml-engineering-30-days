# Day 02｜從 Bit 到 Qubit：量子狀態與量測機率

[Day1](../day01/README.md) 介紹量子計算如何參與機器學習，以及比較模型時需要保留哪些證據。Day2 從一般電腦的 0 與 1 出發，建立描述量子位元與計算量測機率的基本方法。

量子位元（qubit）的狀態可以用兩個數描述，但這兩個數並不是量到 0 或 1 的機率，而是用來計算機率的「振幅」。本章以單一量子位元為例，逐步拆解狀態符號、複數與機率之間的關係，再說明為什麼 **兩個狀態即使量到 0、1 的機率相同，經過後續操作仍可能得到不同結果**。差異來自振幅之間的相對相位，可以先理解成複數在平面上指向的方向差。最後用球面圖像整理這些關係，並以 NumPy 數值運算核對理論機率與重複抽樣的差異。這些概念是後續理解量子電路如何處理資料的基礎。

[Day1](../day01/README.md) introduced the role of quantum computation in machine learning and the evidence needed to compare models. Day2 starts from ordinary bits and develops the basic tools for describing a qubit and calculating measurement probabilities.

A qubit's state can be described by two numbers, but these numbers are amplitudes used to calculate probabilities, rather than the probabilities themselves. This chapter explains state notation, complex numbers, and their connection to measurement, then examines why **two states with identical probabilities for 0 and 1 can behave differently after further operations**. The difference comes from relative phase, which can be pictured as the difference between the directions of complex numbers in a plane. A geometric picture and a small NumPy calculation connect these ideas to theoretical probabilities and repeated sampling, providing a foundation for understanding how quantum circuits process data.

---

## 1. 從只能取 0 或 1 的位元開始

一般電腦的位元（bit）只有 0、1 兩種可能值。兩個位元組合後，可以表示 `00`、`01`、`10`、`11`；某個確定的組合，例如 `01`，就是這四種可能之一。

量子位元的量測結果也可以是 0 或 1，但量測前的狀態需要更多資訊才能描述。本章先討論純態（pure state）：能由一個狀態向量完整描述的量子狀態。向量可以先理解成按順序排列的一組數值。受到雜訊影響或以機率混合不同純態的情況，可能需要更一般的表示方式，會留到後續章節。

## 2. 拆開量子狀態的符號

單一量子位元的純態寫成：

```text
|ψ⟩ = α|0⟩ + β|1⟩
```

- `ψ` 是希臘字母 psi，用來替狀態命名。
- `|ψ⟩` 表示一個狀態向量，外面的 `| ⟩` 稱為 ket 符號。
- `|0⟩` 表示在 0／1 量測方式下必定得到 0 的狀態；`|1⟩` 則必定得到 1。
- `α`、`β` 是希臘字母 alpha、beta，代表兩個機率振幅（probability amplitude），也就是組合這兩個基本狀態的係數。

`|0⟩` 和 `|1⟩` 合稱計算基底（computational basis）。基底就像建立座標所用的基本方向，任何單一量子位元的純態都能用這兩個方向組合表示。

兩項振幅都不為零時，這個狀態相對於計算基底具有疊加（superposition）。疊加會影響後續運算，但在這個基底下量測一次，仍只得到 0 或 1。

## 3. 狀態符號其實可以寫成向量

狄拉克符號（Dirac notation）就是使用 ket 等符號表示量子狀態與運算的寫法。轉成直向排列的欄向量（column vector）後：

```text
|0⟩ = [1, 0]ᵀ
|1⟩ = [0, 1]ᵀ
|ψ⟩ = [α, β]ᵀ
```

右上角的 `ᵀ` 表示轉置（transpose），也就是把橫向排列改成直向排列。因此，`|0⟩` 是一個向量的名稱，不是數字 0；它的第一個分量為 1，第二個分量為 0。

計算量測機率時，還需要了解振幅可以使用的複數。

## 4. 複數、振幅與機率

複數（complex number）寫成 `a + bi`，其中 `a`、`b` 是實數，而虛數單位 `i` 滿足 `i² = -1`。可以把複數畫在平面上：橫座標是 `a`，縱座標是 `b`。

複共軛（complex conjugate）將虛部的正負號反轉：`a + bi` 的共軛是 `a - bi`。複數乘上自己的共軛，得到絕對值的平方，也就是平面上到原點距離的平方：

```text
|a + bi|² = (a + bi)(a - bi) = a² + b²
```

依波恩規則（Born rule），在計算基底下量測的機率為：

```text
P(0) = |α|²
P(1) = |β|²
```

`P(0)` 表示得到 0 的機率。兩個結果涵蓋所有可能，因此 `|α|² + |β|² = 1`。這個條件稱為正規化（normalization），也等同狀態向量的長度為 1。

例如 `α = 1/√2`、`β = i/√2`，兩個振幅的絕對值平方都是 `1/2`，所以 0 與 1 各有 50% 機率。這裡必須取「絕對值平方」；直接算 `β²` 會得到 `-1/2`，不能作為機率。

### 用內積檢查正規化

將 ket 轉成橫向排列，再對每個分量取複共軛，得到 bra，記作 `⟨ψ|`。這個步驟稱為共軛轉置（conjugate transpose）：

```text
|ψ⟩ = [α, β]ᵀ
⟨ψ| = [α*, β*]
⟨ψ|ψ⟩ = α*α + β*β = |α|² + |β|² = 1
```

星號 `*` 表示複共軛。最後一行是向量的內積（inner product）：將對應分量相乘後相加。向量與自己的內積就是長度的平方，因此這個算式可以檢查狀態是否已正規化。

## 5. 相同機率，為什麼還可能是不同狀態？

複數除了到原點的距離，還有指向的角度，這個角度稱為相位（phase）。兩個振幅的角度差稱為相對相位（relative phase）。例如，正的實數指向右方，負的實數指向左方，兩者相差半圈。

考慮兩個狀態：

```text
|+⟩ = (|0⟩ + |1⟩) / √2
|−⟩ = (|0⟩ − |1⟩) / √2
```

兩者量到 0、1 的機率都是 50%，但第二項的正負號不同，因此相對相位不同。量子閘（quantum gate）是改變量子狀態的基本操作；施加稱為 Hadamard 閘的 H 操作後：

```text
H|+⟩ = |0⟩
H|−⟩ = |1⟩
```

此時再量測，就能得到不同的確定結果。H 閘會將原本的振幅重新組合，讓某些項相加、某些項抵消，這種作用稱為干涉（interference）。Day 03 會以矩陣，也就是按列與欄排列的數值表，直接計算這個過程。

### 整體相位與相對相位

若所有振幅都乘上同一個長度為 1 的複數，只會改變整體相位（global phase）。例如，將整個狀態乘上 `-1`，得到 `-|ψ⟩`，不會改變物理預測。

相對相位則比較同一狀態內不同振幅的方向差。前面的 `|+⟩` 與 `|−⟩` 只改變其中一項的符號，因而能在後續操作中產生差異。

## 6. 量測如何產生結果？

在理想的計算基底量測中：

```text
α|0⟩ + β|1⟩ → 以 |α|² 的機率得到 0，量測後為 |0⟩
             → 以 |β|² 的機率得到 1，量測後為 |1⟩
```

若得到 0，立刻以相同方式再次量測，而且中間沒有其他操作或干擾，就仍會得到 0。要估計原本狀態的機率分布，需要每次重新準備原本的狀態，再執行量測。

這些重複執行次數稱為 shots。量測計數（counts）記錄各個結果出現幾次，例如 1,000 次中有 493 次得到 0、507 次得到 1。除以總次數後，便得到機率的抽樣估計；有限次數的結果不必剛好等於理論機率。

期望值（expectation value）是依機率計算的平均值。例如將結果 0 記為 `+1`、結果 1 記為 `-1`，期望值就是 `P(0) − P(1)`。實驗則以對應的樣本平均估計它。

## 7. 布洛赫球：單一量子位元的幾何圖像

布洛赫球（Bloch sphere）將單一量子位元的純態畫成球面上的點。忽略整體相位後，狀態可以寫成：

```text
|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩
```

`θ` 是從北極方向量起的角度，`φ` 是繞著南北軸的角度。`sin`、`cos` 是三角函數；`e^(iφ) = cosφ + i sinφ` 則是長度為 1、方向為 `φ` 的複數，負責表示相對相位。

- 北極是 `|0⟩`，量測必定得到 0。
- 南極是 `|1⟩`，量測必定得到 1。
- 赤道上的狀態量到 0、1 的機率各為 50%，但不同位置具有不同的相對相位。
- `|+⟩` 與 `|−⟩` 位於赤道的相反兩側。

對單一量子位元的理想量子閘，可以用球面上的旋轉理解。球面上的點代表狀態，並不是粒子在空間中的實際位置；多量子位元的完整狀態也不能直接用同一顆球表示。

## 8. 這和機器學習中的向量有什麼關係？

機器學習（Machine Learning，ML）常把一筆資料的特徵排列成向量，例如花瓣長度與寬度。量子狀態也用向量表示，但用途與限制不同：

| 一般數值運算 | 量子計算中的對應與限制 |
|---|---|
| 用特徵向量保存輸入 | 狀態向量保存振幅，必須滿足正規化條件 |
| 用矩陣轉換向量 | 理想量子閘使用么正矩陣（unitary matrix），也就是保持向量長度與內積的矩陣 |
| 直接讀取程式中的數值 | 真實裝置透過量測取得結果，單次量測不能讀出所有振幅 |
| 重複抽樣估計平均值 | 有限 shots 也會產生抽樣波動，需要與理論值分開解讀 |

模擬器是以一般電腦數值運算模仿量子系統的程式，可以保存並檢查完整狀態向量。量子處理器（Quantum Processing Unit，QPU）則是真正執行量子操作的硬體，取得資訊的方式受到量測規則限制。

## 9. 小實驗：用向量算量測機率

NumPy 是 Python 的數值運算套件。以下程式建立兩個複數分量，計算向量與自己的內積，再求每個振幅的絕對值平方：

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

程式中的 `1j` 是 Python 表示虛數單位的方式；`np.complex128` 指定使用 128 位元儲存一個複數，其中實部與虛部各占 64 位元。`np.vdot` 會對第一個向量取複共軛後計算內積，`.real` 取出實部，`np.abs` 則計算絕對值。

變數 `norm` 在這段程式中實際保存的是向量長度的平方；正規化後，長度與長度平方都等於 1。若改用 `[1, 1]`，長度平方會是 2，需要將整個向量除以 `√2` 才能正規化。

專案已提供可直接執行的 [state_vector.py](state_vector.py)：

```bash
python articles/day02/state_vector.py
```

程式還會固定隨機種子（seed），也就是產生隨機序列的起始設定，再模擬有限 shots。相同設定方便重現抽樣結果；抽樣是在一般電腦上進行，並未使用 QPU。

## 10. 從結果檢查觀念

閱讀輸出時，可以依序核對：

1. 振幅的絕對值平方相加是否為 1，而不是檢查振幅本身相加是否為 1。
2. 理論機率與有限次抽樣比例是否被分開記錄；50% 的理論機率不要求每批樣本恰好各占一半。
3. 比較狀態時，是否也考慮相對相位；只比較 0／1 的機率不能完整辨識狀態。
4. 完整向量是否來自模擬器內部資料，而非誤認成硬體單次量測的輸出。

## 11. 從單一量子位元走向 QML

量子機器學習（Quantum Machine Learning，QML）會將資料轉成量子狀態，再用含有可調參數的電路處理，最後由量測產生預測所需的數值。本章的振幅、正規化與量測機率，正是這條流程的起點。

對 `n` 個量子位元，純態向量有 `2^n` 個複數振幅：1 個量子位元需要 2 個，2 個需要 4 個，3 個需要 8 個。每增加一個量子位元，儲存完整向量所需的數值就加倍，因此模擬的記憶體成本會快速成長。

較大的狀態空間並不自動代表量子優勢，也就是在明確任務與成本條件下優於適當的一般計算方法。資料如何放入電路、需要多少操作與量測，以及比較方法是否公平，都會影響結論。

## 12. 延伸資源

- [F1] Nielsen and Chuang, *Quantum Computation and Quantum Information*, 10th Anniversary Edition, Cambridge University Press (2010), [publisher page](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE)。用於 量子狀態、狄拉克符號與量測 的標準背景。
- [R4] Schuld and Petruccione, *Supervised Learning with Quantum Computers*, Springer (2018), [DOI](https://doi.org/10.1007/978-3-319-96424-9)。用於把 量子資訊基礎連接到監督式 QML，也就是使用附有正確答案的資料訓練模型。
- [D1] [NVIDIA CUDA-Q Quick Start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)。提供電路與取樣的入門範例；後續章節會使用 CUDA-Q 這套量子程式開發工具實作電路。

完整書目與使用目的見 [文獻與資源索引](../../REFERENCES.md)。

## 13. 下一篇

[Day 03](../day03/README.md) 將以矩陣運算改變狀態向量，逐一介紹單量子位元的基本量子閘，並驗證 H 閘如何讓相對相位影響量測結果。兩個量子位元之間的操作則留到 Day 04。

## 延伸研究

[N2] Theodore McKeever and Ahsan Nazir. “An Introduction to the Foundations and Interpretations of Quantum Mechanics.” arXiv:2603.09818v2 (2026)；預印本講義。[原始來源](https://arxiv.org/abs/2603.09818v2)；[完整書目](../../REFERENCES.md#n2)。

本章從位元走向量子位元；這份講義可延伸理解狀態與量測規則，避免把疊加態當成一次就能讀出多個答案。
