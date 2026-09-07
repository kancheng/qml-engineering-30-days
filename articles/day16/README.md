# Day 16｜QNN 到底是不是 Neural Network？

**在本系列，QNN 指以參數化量子電路提供可訓練輸出的模型；它能接入 ML pipeline，但不等同把 MLP 的 neuron 換成 qubit。** QNN 也不是所有文獻中只有一種架構的名稱。本文聚焦 Day 14–15 使用的 variational circuit classifier。

今天以相同二維輸入與 scalar probability 介面，比較一個四參數 PQC 和九參數 MLP。程式：[models.py](models.py)、[demo.py](demo.py)、[experiment.py](experiment.py)。兩者都重新初始化，沒有載入 Day 15 訓練 weights，沒有 accuracy 排名。

## 1. 相似的是訓練介面

```text
features → model(weights) → prediction → loss → classical optimizer
```

Day 15 的量子模型與 MLP 都可以提供 scalar loss 給 optimizer。但相同的介面不代表相同的內部計算、參數意義或成本。NVIDIA 的 Hybrid QNN 教學示範將 classical layers 與量子 expectation 組合，提供這種整合的實例。[D15] 本文不安裝 PyTorch，也不移植該版本教學的梯度實作。

| 項目 | 本日 PQC／QNN | 本日 MLP |
|---|---|---|
| 輸入 | 兩個已縮放 features | 同一組 features |
| 內部 | RY feature map＋一層 RY／CNOT Ansatz | 2→2→1 dense layers |
| 參數 | 四個 rotation angles | dense weights 與 biases，共九個 |
| 非線性表現 | 角度編碼及 expectation 對輸入／參數的函數 | tanh hidden activation＋sigmoid output |
| scalar 輸出 | p1=(1−ZZ)/2 | sigmoid(logit) |
| 本日執行 | qpp-cpu 或 nvidia simulator | NumPy host CPU |

GPU backend 指 QNN 的 simulator；本日 MLP 在兩次執行中都由 CPU NumPy 計算。沒有做兩者速度比較。

## 2. 九個 MLP 參數怎麼數？

```text
h = tanh(x W1 + b1)
z = h W2 + b2
p1 = sigmoid(z)
```

| 部分 | Shape | 參數數量 |
|---|---|---:|
| W1 | 2×2 | 4 |
| b1 | 2 | 2 |
| W2 | 2 | 2 |
| b2 | scalar | 1 |
| 合計 | | 9 |

`mlp_forward` 的 weights 展平成長度 9：前四個 reshape 為 W1，第 4–5 個為 b1，第 6–7 個為 W2，最後一個為 b2。索引從 0 起算。

```python
hidden = np.tanh(x @ w[:4].reshape(2, 2) + w[4:6])
logits = hidden @ w[6:8] + w[8]
p1 = 0.5 * (1 + np.tanh(logits / 2))
```

最後一行等價於 sigmoid，避免直接計算巨大 exp。測試另外用逐個 hidden neuron 的 scalar 公式核對 batch matrix 計算，並驗證零 weights 輸出 0.5、只有 output bias 時的解析結果。

這是刻意選擇的最小 MLP，不是為了匹配 QNN 容量而搜尋到的架構。

## 3. 四個 QNN 參數不是四個 Amplitudes

Day 14 一層 Ansatz 有四個 RY angles；兩個 qubit 的 pure state 恰好有四個 complex amplitudes，但兩個「四」沒有一對一關係。

weights 決定 gates，gates 共同改變 state。amplitudes 是運算中間態，受 normalization 與 global phase 等條件約束，不是四個獨立可訓練 neuron。四個 rotation 參數也不表示能自由指定任意四個 complex amplitudes；本例 RY-only 從實數態出發。

CNOT 是兩個 qubit 之間的 unitary 操作，沒有新增一個 floating-point weight。增加層數會增加 rotation 參數和電路成本，不能直接對應成 MLP hidden width。

## 4. Gate 線性，為什麼輸出可以非線性？

固定 unitary U，對任意向量 v、w 與係數 a、b：

```text
U(av+bw) = aUv+bUw
```

實驗用 Day 3 的 RY matrix 核對此式。線性代數測試的向量組合不要求 normalized；它檢查矩陣作用的性質，不是宣稱所有係數組合都直接是合法量子態。

另一方面，從 |0⟩ 做 RY(θ) 再讀 Z：

```text
f(θ) = cos(θ)
f(π/4) − [f(0)+f(π/2)]/2 ≈ 0.207107
```

因此 θ→f(θ) 並非線性。θ 依賴輸入 x 時，模型也能對 classical x 呈現非線性，無需在 qubit amplitude 上插入 tanh。

要區分三種敘述：

- U 對 state vector 的作用是線性。
- expectation 對 pure-state amplitudes 是二次形式 `ψ†Oψ`，對 density matrix 則是線性 `Tr(ρO)`。
- classical features 經角度編碼、參數化 gates 到 expectation，可以形成非線性函數。

不能把這些結論簡化為「量測就是 ReLU」或「量子 gate 是 nonlinear neuron」。本例的 exact expectation 是 deterministic simulator 計算，仍然會呈現上述非線性，不需要抽樣噪聲才產生它。

## 5. 參數幾何也不同

單個 RY angle 增加 2π，該 gate 只差 global phase，所以本例的預測不變。測試逐一對四個 QNN weights 加 2π 核對。MLP 的一般 dense weights 沒有同樣的 2π 週期規則。

tanh 或 sigmoid 可能使某些參數改動只造成很小的輸出變化；QNN 也可能因選定 state、gate 或 observable 而不敏感。本日只做固定點的有限參數改動，**不是梯度、trainability 或 barren plateau 證據**。

實驗固定 features `[0.25,-0.4]`、seed 42，對每個參數分別加 −0.5、0、+0.5：QNN 共 4×3=12 組，MLP 共 9×3=27 組，合計 39 組。兩種模型的參數單位與用途不同，不能把相同 delta 當成公平容量比較。

## 6. 同介面不代表同樣的輸出語意

QNN 的 p1 是 ZZ 奇數 parity 的 probability；MLP 的 p1 是 sigmoid score。都可接 binary label loss，但都沒有因為輸出在 [0,1] 就自動得到機率校準。

Day 15 已展示其中一個 QNN 可完成小型 XOR 任務。本日使用**未訓練**模型，畫出固定 x1=0.25、x0∈[−1,1] 的輸出切片。曲線較彎、較平或靠近 0.5，不代表模型較好或較差。

![Untrained forward comparison](../../results/day16/qpp-cpu/forward_comparison.png)

兩個 seeds 都使用 normal(0,0.2) 初始化，但參數數量、意義與資料流不同；相同 seed 不是 matched initialization。圖中沒有 labels、loss 或 accuracy，不能用來評估泛化。

## 7. 可執行示範

沿用 `.venv` 和 Day 15 繪圖套件，本日無新增依賴；版本鏈：[requirements-day16.txt](../../requirements-day16.txt)。專案根目錄執行：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day16/demo.py
OMP_NUM_THREADS=1 python articles/day16/demo.py --features -0.5 0.25 --backend nvidia

OMP_NUM_THREADS=1 python articles/day16/experiment.py
OMP_NUM_THREADS=1 python articles/day16/experiment.py --backend nvidia
OMP_NUM_THREADS=1 python articles/day16/plot_results.py
OMP_NUM_THREADS=1 python articles/day16/plot_results.py --backend nvidia
```

每個 backend 保存 2 seeds × 21 個輸入，共 42 組 QNN／MLP forward 比較，以及上述 39 組參數反應。QNN 的 CUDA-Q 結果與 NumPy reference 核對；MLP 則由 scalar unit tests 核對 matrix 實作，未把同一函式再執行一次當作獨立 reference。

輸出為 `results/day16/<backend>/predictions.json`、`parameter_response.json`、`summary.json`、`forward_comparison.png`。前兩者保存完整 inputs 與 weights。experiment 重跑更新目錄，可加 `--output-dir /tmp/day16-check` 另存；繪圖工具讀取預設 backend 目錄。

## 8. 驗證與下一步

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day16 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY16_TARGET=nvidia python -m unittest discover -s articles/day16 -p 'test_*.py' -v
```

六個測試核對 MLP bias／scalar reference／輸入契約、初始化與不變性、QNN 週期性與參數反應，以及 state 線性／角度輸出非線性的區別。完整結果見 [實驗紀錄](../../results/day16/README.md)。

[Day 17](../day17/README.md) 再處理梯度如何取得，Day 18 才進行公平 Classical ML vs QML benchmark；Day 19 延伸 classical layer＋quantum layer 的 hybrid model。本日未執行梯度訓練、QPU、速度評比或量子優勢驗證。

## 9. 來源

[D15] [NVIDIA Hybrid Quantum Neural Networks（CUDA-Q 0.8.0 教學）](https://nvidia.github.io/cuda-quantum/0.8.0/examples/python/tutorials/hybrid_qnns.html)：作為 classical layers 與 quantum expectation 整合的歷史範例，不作本日 API／gradient 正確性的依據。查閱日期 2026-09-07，實際環境 CUDA-Q 0.15.1。

本日 MLP 與線性代數公式均明示並測試，QNN 重用 Day 14；來源索引見 [REFERENCES.md](../../REFERENCES.md)。
