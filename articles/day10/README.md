# Day 10｜第一個 CUDA-Q 訓練迴圈：依預測誤差更新權重

Day 9 的權重還是人指定的。今天補上訓練真正缺的兩步：**怎麼衡量「預測差多少」**，以及**要不要接受下一組權重**。

想像調收音機：每次微微轉一個旋鈕，聽雜訊有沒有變小；變小就留下，沒變好就轉回去，再試下一個旋鈕。量子電路負責依目前旋鈕算出預測；一般 Python 程式負責算誤差、決定要不要接受新旋鈕。兩者分工，才組成一輪訓練。

本章沿用 Day 9 的兩位元、四個 RY 權重與 ZZ 讀出，不改 [模型程式](../day09/pqc.py)。資料是可手算答案的小型回歸題，方便檢查流程有沒有跑通。完整程式：[train.py](train.py)。

Day10 adds a loss function and a coordinate-search optimizer around Day9's parameterized circuit. Exact simulator expectations drive training on a small synthetic regression task; holdout checks and a final finite-shot readout are recorded separately. Lower training error verifies the hybrid loop, not practical generalization or quantum advantage.

---

## 1. 一輪訓練在做什麼？

```mermaid
flowchart LR
    W[候選權重] --> Q[CUDA-Q：逐筆跑電路]
    Q --> P[observe：得到 ZZ 期望值]
    P --> L[CPU：算訓練集 MSE]
    L --> O[CPU：座標搜尋決定是否更新]
    O --> W
```

讀圖：量子模擬器算「這組權重下，每筆輸入的預測」；CPU 把預測與答案比對成一個損失數字；最佳化器再決定下一個候選權重。

幾個會反覆出現的詞：

- **參數化量子電路（PQC）**：含可調角度的電路；今天這些角度叫**權重**
- **ZZ（Z0Z1）**：兩個量測結果相同記 +1、不同記 −1，再取平均；這個平均叫**期望值**
- **執行後端（backend）**：指定用哪個模擬器；本章的 CPU／GPU 後端都在一般電腦上模擬，沒有真實量子處理器

`observe` 的期望值可以成為一般程式裡目標函數的一部分。[D12] 本章自己寫座標搜尋，讓「何時接受更新」與「評估成本」都看得見，沒有呼叫 `cudaq.optimizers`。

## 2. 先出一道能核對答案的小題

**回歸**預測連續數值（例如溫度）；**分類**則判斷類別。**特徵**是輸入數字，**標籤**是正確答案。

本章固定兩個特徵，標籤由可直接計算的公式產生：

```python
y = np.cos(np.pi * features[:, 1] + 0.6)
```

`features[:, 1]` 取出每筆的第二個特徵 `x1`（編號從 0 起）。`np.cos` 是餘弦；`0.6` 是固定角度位移。

訓練資料是九個格子點：`x0 ∈ {-0.6, 0, 0.6}` 與 `x1 ∈ {-0.5, 0, 0.5}` 的組合。另留四筆**保留資料（holdout）**，全程不參與更新：`[-0.8,-0.3]`、`[-0.2,0.7]`、`[0.3,-0.7]`、`[0.8,0.2]`。`dataset()` 寫死這些值，不從網路下載。

題目刻意讓標籤只依賴 `x1`。由 Day 9 的解析關係可知，`weights=[0, 0.6, 0, 0]` 就能表示這個目標；測試也會逐筆核對。這是用來檢查「訓練流程能不能把誤差壓下去」的合成題，**不是**真實資料集上的成效證明。

最佳化器拿不到這組已知答案。它從種子 42 或 43 的小角度隨機權重出發。**初始化**指定起點；**隨機種子（seed）**固定隨機序列，方便重做實驗。

## 3. 損失在一般程式裡算

**主控端（host）**安排電路執行與整理結果。**損失函數**把「差多少」收成一個數字；本章用**平均平方誤差（MSE）**：每筆預測減答案、平方，再對訓練集平均。例如某筆差 0.1，該筆平方誤差是 0.01。

```text
L(w) = (1/N) Σ_i (f(x_i,w) − y_i)²
f(x,w) = ⟨Z0 Z1⟩
```

`N` 是訓練筆數；`f` 是固定權重下的電路輸出。程式核心：

```python
def objective(weights):
    predictions = [predict(x, weights, 1) for x in train_features]
    return mse(predictions, train_labels)
```

**目標函數（objective）**是最佳化器用來比較候選設定的規則；本例就是整批訓練 MSE。**前向計算**只在固定權重下算預測，本身不改權重。

每次目標函數包含九次精算 `observe`：量子模擬器算預測，Python／NumPy 算損失。不要把 `observe` 當成最佳化器，也不要把這個實數輸出當成分類機率。

## 4. 座標搜尋：一次只轉一個旋鈕

**座標搜尋**把每個權重當成一個方向，每次只試改其中一個。它直接比較候選結果的損失，**不需要梯度**；梯度描述參數微微改變時，損失往哪邊、改多少。

每一輪依序處理 w0 到 w3。對目前第 `j` 個權重，試 `w[j] − step` 與 `w[j] + step`，選損失較小者；**只有嚴格變好才接受**。下一個座標沿用已接受的權重。

```python
candidate = weights.copy()
candidate[j] += direction * step
candidate_loss = objective(candidate)
# 每個座標比較正、負兩個候選，再決定是否更新。
```

`weights.copy()` 避免試候選時改壞原資料；`direction` 為 +1 或 −1。**步長（step）**是每次加減的量，單位是弧度（π 為半圈）。

規則：初始步長 0.4；若一整輪都沒更新，步長減半；步長小於 `1e-4` 或達到 40 輪就停。這是離散搜尋，**不保證**找到所有權重裡誤差最低的全域最佳解。停止原因寫成 `max_sweeps`，代表預算用完，不能直接解讀成「已經收斂」；收斂通常要事先定義更明確的穩定條件。

四個參數每輪要八次目標函數。跑完 `S` 輪後：

```text
objective evaluations = 1 + 8S
training observe calls = 9 × (1 + 8S)
```

第一項是初始損失。最終核對、保留資料與抽樣另計，不計入 `training_observe_calls`。`history.json` 第 0 筆是起點；之後每筆是一輪結束後的權重、損失、該輪步長與累積評估次數。

## 5. 怎麼執行？

沿用既有 `.venv`；固定依賴見 [requirements-day10.txt](../../requirements-day10.txt)。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day10/train.py
OMP_NUM_THREADS=1 python articles/day10/train.py --seed 43
OMP_NUM_THREADS=1 python articles/day10/train.py --backend nvidia
OMP_NUM_THREADS=1 python articles/day10/train.py --backend nvidia --seed 43
```

預設寫入 `results/day10/<backend>/seed<seed>/`，重跑會更新該目錄。可用 `--output-dir /tmp/day10-check` 另存；`--sweeps 3` 可做短流程示範。`OMP_NUM_THREADS=1` 固定 CPU 執行緒數，方便控制小型電路的額外平行成本，**不是** GPU 加速證據。

## 6. 結果檔怎麼讀？

- `summary.json`：環境、種子、初始／最終權重、停止原因、評估成本、訓練與保留 MSE
- `history.json`、`training_curve.csv`：每輪曲線與權重，可開試算表
- `predictions.json`：輸入、標籤、初始／最終預測、NumPy 參考，以及最終 1,000 次量測的計數

**比較基準（baseline）**提供「什麼都不學」的參考：對所有輸入都預測**訓練標籤平均值**，並同時看訓練集與保留集。保留資料不參與損失、步長選擇或停止條件。種子 42 與 43 的結果都保留，不挑好看的那個當代表。

四筆保留資料只是這個示範的額外核對，**不能**估計實務上的泛化能力（對未參與訓練的新資料表現如何）。最終預測與 Day 9 的 NumPy 矩陣比對，是在檢查前向計算是否算對，不是另一個學習模型的效能對照。

實際數字見 [Day 10 結果](../../results/day10/README.md)。四次實驗（CPU／GPU × 兩顆種子）訓練 MSE 都下降；常數基準的訓練 MSE 約 0.36。種子 43 的最終誤差高於 42，說明起點與有限搜尋預算會影響結果。

## 7. 精算訓練 ≠ 抽樣式訓練

訓練使用 `shots_count=-1`：在無雜訊模擬器上由完整狀態算期望值，不做有限次抽樣。「精確」仍可能有浮點誤差（電腦用有限位數存小數）。

訓練結束後，另用 1,000 次量測讀一次結果，保存種子與各結果出現次數（counts）。這些抽樣**沒有**回傳給最佳化器。

因此紀錄裡損失不會變差，是因為實作只接受「確定變好」的候選。若改用有限次抽樣來比較候選，抽樣波動可能讓「看起來較好」的選擇不穩；本日沒有驗證那種訓練方式。CPU 與 GPU 數值精度不同，非常接近的候選排序也可能不同，故不要求兩邊最終權重逐位一致。

## 8. 測試與銜接

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day10 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY10_TARGET=nvidia python -m unittest discover -s articles/day10 -p 'test_*.py' -v
```

測試涵蓋：MSE 輸入檢查、已知二次函數上的搜尋行為、評估次數與停止原因、原始權重不被就地修改、標籤可表示性、資料分離，以及真正 CUDA-Q 更新後與 NumPy 的損失核對。

至此，Day 6–10 串起了：可執行環境 → 參數化電路 → 前向介面 → **依誤差更新權重的混合訓練迴圈**。本日驗證的是合成可表示任務上的流程：量子前向與經典最佳化分開、評估成本與停止原因可查、holdout 與種子對照保留。它不是真實資料泛化報告，也不是量子優勢或分類準確率宣稱。

下一篇 [Day 11](../day11/README.md) 回到資料進入電路之前：原始數值如何縮放與編碼。正式分類任務依學習路線留待 Day 15。

## 9. 來源

[D12] [NVIDIA Quantum Algorithmic Primitives](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/algorithmic_primitives.html)：期望值與目標函數最佳化的角色。查閱日期：2026-09-06，實際執行 CUDA-Q 0.15.1。座標搜尋與合成回歸是教學實作；共用索引見 [REFERENCES.md](../../REFERENCES.md)。

## 延伸研究

[N5] Brian Coyle et al. “Training-efficient density quantum machine learning.” npj Quantum Information 11, 172 (2025)；研究論文。[原始來源](https://doi.org/10.1038/s41534-025-01099-6)；[完整書目](../../REFERENCES.md#n5)。

本章完成參數更新迴圈；這篇研究可延伸思考每次更新需要多少電路求值。本章的座標搜尋沒有實作該論文方法。
