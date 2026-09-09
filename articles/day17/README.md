# Day 17｜QML 怎麼 Backprop？Gradient 從哪裡來？

## 本章摘要｜初學者學習筆記

### 中文

[Day16](../day16/README.md) 比較 QNN 與 MLP 的參數、運算與讀出，釐清量子閘對狀態的線性作用，如何與模型輸出的非線性並存。Day17 接著探討如何計算參數變動對輸出與損失的影響，讓最佳化器能利用梯度調整權重。

這一章目標在於理解 **「量子電路的梯度從哪裡來，以及如何把電路輸出的導數接到完整的訓練損失」**，為使用梯度的 QML 訓練建立基礎。Gradient（梯度）描述各個權重微小改變時，目標數值如何變化，可用來決定更新方向。本章先用 central finite difference（中央有限差分）比較權重增加與減少一小段距離後的輸出，理解近似步長與數值誤差的取捨；再實作 parameter-shift（參數位移法），對本章每個只控制一次 RY 的獨立權重，利用正負 π/2 位移後的兩次電路輸出求導。重點是位移公式有適用條件，不能直接套到任意共享權重或整個 loss；電路期望值的導數仍須透過 chain rule（鏈式法則），接上機率轉換與 Brier loss。本章以獨立矩陣導數核對結果，執行短程梯度下降，並觀察有限 shots 如何使梯度估計產生波動。讀完本章，應能區分輸出導數與損失梯度，說明有限差分與參數位移法的差別，並理解取得梯度需要額外電路評估，解析公式成立也不代表有限抽樣沒有誤差。

### English

[Day16](../day16/README.md) compared QNN and MLP parameters, computation, and readout, clarifying how linear gate action on states can coexist with nonlinear model outputs. Day17 examines how parameter changes affect outputs and loss, enabling an optimizer to use gradients for weight updates.

This chapter aims to explain **where quantum-circuit gradients come from and how output derivatives connect to the full training loss**, establishing a foundation for gradient-based QML training. A gradient describes how a target quantity changes under small changes to each weight and can guide the update direction. The chapter first uses central finite differences to compare outputs at slightly increased and decreased weights, examining the tradeoff between approximation step size and numerical error. It then implements parameter-shift: for each independent weight controlling a single RY gate in this model, two circuit evaluations at shifts of plus and minus π/2 provide the output derivative. The shift rule has specific conditions and cannot be applied directly to arbitrary shared weights or the entire loss. Expectation derivatives must still pass through the chain rule for the probability conversion and Brier loss. Independent matrix derivatives check the results, a short gradient-descent run tests integration, and finite-shot experiments illustrate fluctuations in gradient estimates. The learning goal is to distinguish output derivatives from loss gradients, explain the difference between finite differences and parameter-shift, and recognize both the additional circuit evaluations required and the sampling error that remains even when an analytic shift identity holds.

---

Day 15 用不需要梯度的座標搜尋，Day 16 比較 QNN 與 MLP。今天實作 **parameter-shift、central finite difference、獨立矩陣導數與 loss chain rule**，再跑四次 gradient descent 更新。

程式：[gradients.py](gradients.py)、[experiment.py](experiment.py)、[demo.py](demo.py)。沿用 Day 14 的獨立 RY weights；沒有 PyTorch autograd 或 QPU 執行。

## 1. Backprop 的外層還是 Chain Rule

```text
weights → PQC → f=⟨ZZ⟩ → p=(1−f)/2 → Brier loss
```

classical loss 可用一般微積分求導；量子部分需要提供 `df/dw`。parameter-shift 是取得電路 expectation 導數的方法，不等於整個 ML computation graph 的 reverse-mode backprop。

本日對每個參數逐一執行 shifted circuits。沒有儲存每個量子 gate 的 activation 後倒傳，也沒有使用 simulator adjoint differentiation。這些方法的執行需求不同，不能只用「都是 backprop」略過成本。

## 2. Central Finite Difference：步長是近似參數

```text
df/dw_j ≈ [f(w+h e_j) − f(w−h e_j)] / (2h)
```

對平滑函數，central difference 的截斷誤差隨小 h 下降，但 h 太小會放大浮點相減誤差；finite shots 時還會放大抽樣波動。因此不是 h 越小越好。

本日比較 h=`1e-1,1e-3,1e-5,1e-7`，兩個 backend 的精度分別為 CPU fp64、GPU fp32。有限差分是**診斷結果**，不要求所有 h 都通過同一誤差門檻；故意保留小步長在 fp32 上可能失準的數字。

## 3. Parameter-shift：符合條件時的解析等式

對只出現一次的獨立 RY 參數，`RY(w)=exp(−iwY/2)`：

```text
df/dw_j = [f(w + π/2 e_j) − f(w − π/2 e_j)] / 2
```

這不是把 finite difference 的 h 隨便設成 π/2：分母是 **2，不是 π**。公式來自該 gate generator 的頻譜結構；Schuld 等人的研究推導了這類解析 shifted-circuit 梯度。[P3]

本日實作共用 central-difference helper，再乘上 π/2，把分母轉回 2。可讀成以下等價程式：

```python
plus = weights.copy()
minus = weights.copy()
plus[j] += np.pi / 2
minus[j] -= np.pi / 2
gradient[j] = 0.5 * (predict(x, plus) - predict(x, minus))
```

Day 9／14 的每個 weight 只控制一個 RY，符合本例條件；layer 1 有四個、layer 2 有八個獨立 weights。這不自動適用於任意 gate、不同 generator、縮放角度或共享 weight。

## 4. 共享參數的反例

若同一 θ 出現在連續兩個 RY：`RY(θ)RY(θ)=RY(2θ)`，讀 Z 得 `f(θ)=cos(2θ)`。此時：

```text
真實導數 = −2 sin(2θ)
直接把共享 θ 同時 ±π/2：差值為 0
```

必須處理各 gate occurrence 的導數並加總，或使用符合該頻率／generator 的規則。若實際角度為 `a*w+b`，還要乘上 classical chain factor a。本日測試保存共享參數的反例，沒有把標準兩點公式當通用黑盒。

## 5. 正確接到 Brier Loss

`p=(1−f)/2`、`L=mean((p−y)²)`，因此：

```text
dL/dw = mean[2(p−y) × (−1/2) × df/dw]
       = −mean[(p−y) df/dw]
```

```python
values = np.array([predict(x, weights, config) for x in features])
p = (1 - values) / 2
jacobian = np.array([parameter_shift(x, weights, config) for x in features])
gradient = np.mean(-(p - labels)[:, None] * jacobian, axis=0)
```

不能直接計算 `[L(w+π/2)−L(w−π/2)]/2` 就宣稱得到 loss gradient：平方等 classical processing 會改變函數形式。本日用單 qubit Brier 反例驗證兩者不一致。

這裡明確對未 clipping 的解析 `p=(1−f)/2` 求導；exact simulator 輸出在數值誤差內符合機率範圍。若在計算圖額外加 clipping 或 threshold，必須另處理其導數，不能默認套用本式。分類的 hard threshold 不在這條 loss 計算路徑中。

## 6. 不用 Parameter-shift 驗證自己

`matrix_gradient` 使用 Day 3 的矩陣：

```text
dRY(w)/dw = (−iY/2) RY(w)
df/dw_j = 2 Re[⟨∂ψ/∂w_j|ZZ|ψ⟩]
```

逐 gate 傳遞 state 及每個參數的 derivative state，遇到對應 rotation 時加入該 gate 的導數。這個 reference 不使用 shifted evaluations，也不用 finite difference，因此能獨立檢查 shift 的符號、倍率與層數。

loss gradient 再與 NumPy 完整 Brier loss 的 central difference（h=1e-5）比較，核對外層 chain rule。測試也涵蓋 amplitude feature map 的兩層 Ansatz，但主要 sweep 固定 angle encoding。

## 7. 梯度成本與短更新

P 個獨立 weights、B 筆資料：

```text
expectation Jacobian：2PB 次 shifted forward
loss + gradient：B(1+2P) 次 forward（含原始 p）
```

本日 B=2、P=4，每次 loss＋gradient 為 18 次 observe。四次更新 `w ← w−0.4*gradient`，另外在最終 weights 評估一次，共五次 gradient evaluations、90 次 observe。未把 NumPy reference、主要 sweep 與 sampling 額外成本算進這個數字。

兩筆 features=`[[0.25,-0.4],[-0.6,0.2]]`，labels=`[1,1]`；這是梯度更新整合檢查，不是重訓 Day 15 的分類器，也沒有 holdout 或 accuracy。固定 learning rate 的 gradient descent 一般不保證每步 loss 都下降；本次結果如實保存，不透過 test 調整步長。

## 8. Finite Shots 仍有梯度不確定性

固定 seed42 weights、features=`[0.25,-0.4]` 的 w1 導數，用 100／1,000 shots **每個 shift**，各十對 sampling seeds。每筆梯度是兩個 ZZ 估計值差的一半，保存兩份 counts 與 seeds。

解析公式成立，不代表有限次抽樣的梯度恰好等於真值。若兩個 shift 的抽樣獨立：

```text
Var(g_hat) = [Var(f_plus_hat) + Var(f_minus_hat)] / 4
```

對 ±1 observable、每側 S shots，其 variance 上界為 `1/(2S)`。本日這個公式由獨立樣本平均推導，十次重複只作小型展示，不保證實測誤差隨 shots 嚴格單調。沒有把 noisy p 與 noisy Jacobian 相乘來宣稱 loss gradient 的無偏性。

## 9. 執行與圖表

沿用 `.venv`，無新增依賴；[requirements-day17.txt](../../requirements-day17.txt)。在專案根目錄執行：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day17/demo.py
OMP_NUM_THREADS=1 python articles/day17/experiment.py
OMP_NUM_THREADS=1 python articles/day17/experiment.py --backend nvidia
OMP_NUM_THREADS=1 python articles/day17/plot_results.py
OMP_NUM_THREADS=1 python articles/day17/plot_results.py --backend nvidia
```

每個 backend：L=1／2 × 三筆 features，共六個 gradient cases；每個 case 四種 finite-difference h，保存 24 筆向量比較。另有五筆短更新紀錄與 20 筆 finite-shot gradient estimates。

![GPU gradient error](../../results/day17/nvidia/gradient_error.png)

圖示每個 h 在所有 case／parameters 上的最大誤差，虛線是 parameter-shift 最大誤差。輸出存 `results/day17/<backend>/`，可用 `--output-dir /tmp/day17-check` 另存 experiment；繪圖讀取預設 backend 目錄。完整數字見 [結果紀錄](../../results/day17/README.md)。

## 10. 測試與下一篇

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day17 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY17_TARGET=nvidia python -m unittest discover -s articles/day17 -p 'test_*.py' -v
```

五個測試涵蓋差分與輸入驗證、angle／amplitude 的 shift／matrix／NumPy FD、loss chain rule、loss 直接 shift 反例，以及共享參數反例。

下一篇 [Day 18](../day18/README.md) 進行 Classical ML vs QML 的公平 benchmark。本日仍是無噪聲 simulator 數值核對與梯度示範，沒有 QPU 或量子優勢宣稱。

## 11. 來源

[P3] Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, Nathan Killoran. “Evaluating analytic gradients on quantum hardware.” *Physical Review A* **99**, 032331 (2019). [出版商／DOI](https://doi.org/10.1103/PhysRevA.99.032331)，[arXiv:1811.11184](https://arxiv.org/abs/1811.11184)。2018 為 preprint 年份，正式出版為 2019。

2026-09-07 核對作者、題名、年份、DOI 與兩次 shifted circuit 的主張；本日 RY 公式、chain rule 與反例均由獨立矩陣導數／解析式核對。共用索引：[REFERENCES.md](../../REFERENCES.md)。
