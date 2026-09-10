# Day 15｜第一個 CUDA-Q 量子分類器：訓練、評分與模型保存

[Day14](../day14/README.md) 將資料編碼、可訓練電路、讀出與最佳化器組合成共用模型，並透過短訓練確認元件能一起運作。Day15 接著加入明確的分類標籤、資料切分、模型保存與評分，完成可以重新載入並產生分類結果的 XOR 示範。

**量子電路的數值輸出，需要明確的機率轉換與分類規則，才能成為模型預測。** 加上資料切分、評分與保存，才能檢查訓練效果，並在重新載入後繼續使用模型。本章使用兩個座標的符號判斷 XOR 類別：符號相異為 class 1，同號為 class 0。電路的 `Z0Z1` 期望值透過 `p1=(1−f)/2` 轉成兩個量測位元不同的機率，再以固定的 0.5 門檻決定預測類別。訓練以 Brier loss（預測機率與標籤的平均平方誤差）更新權重；train 決定縮放規則與訓練更新，validation 選擇兩個初始化結果中的最終模型，test 則在選定後用於評分。重點除了 loss 是否下降，也包括分類錯在哪裡、機率輸出與標籤的差距，以及重新載入模型後能否重現預測。因此，本章同時保存 checkpoint（模型檢查點）、訓練曲線與決策邊界，並以常數預測作為基本對照。訓練使用模擬器直接計算的期望值，最後才另外做有限次量測評估；兩種結果需要分開解讀，也不能由小型合成資料的表現推論實際任務的成效。

[Day14](../day14/README.md) combined data encoding, a trainable circuit, readout, and optimization into a shared model, using short training runs to check that the components work together. Day15 adds explicit class labels, data splits, model persistence, and evaluation to complete an XOR example that can be reloaded and used for classification.

**Numerical circuit outputs need an explicit probability conversion and classification rule to become predictions.** Data splits, evaluation, and model saving make training outcomes checkable and the selected model reusable. The task assigns class 1 to coordinate pairs with opposite signs and class 0 to pairs with matching signs. The circuit's `Z0Z1` expectation is converted through `p1=(1−f)/2` into the probability of differing measurement bits, then a fixed threshold of 0.5 determines the predicted class. Training updates weights using Brier loss, the mean squared difference between predicted probabilities and labels. Training data determines scaling rules and weight updates, validation data selects between the final models from two initializations, and test data is used for scoring after selection. Beyond loss reduction, the chapter examines classification errors, probability errors, and whether reloading the model reproduces predictions. Checkpoints, training curves, and decision boundaries are saved, with a constant predictor providing a basic comparison. Training uses exact simulator expectations, followed by a separate finite-shot evaluation. These results have different meanings, and performance on this small synthetic dataset does not establish effectiveness on real tasks.

---

Day 14 已經能把特徵、可調電路模板、輸出讀取與最佳化器接起來。本章將這些元件接成分類流程：**有資料切分、可載入模型、評分、訓練曲線與決策邊界的 XOR 分類器**。

程式入口：[classifier.py](classifier.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[plot_results.py](plot_results.py)。本章以 XOR 作為分類任務。

## 1. 明確定義任務與資料

XOR 是互斥或（exclusive OR）：兩個條件不同時結果為真，相同時為假。本章將座標正負號視為兩個條件，分類器的工作就是判斷輸入屬於類別 0 還是 1。

給定兩個原始座標，符號相異為類別 1，同號為類別 0：

```python
y = (x0 * x1 < 0).astype(int)
```

程式用乘積是否小於 0 判斷異號，`.astype(int)` 將真假結果轉成整數 1 或 0。象限是座標平面由兩條軸分出的四個區域。隨機種子（seed）控制隨機序列，方便在相同環境重做資料產生與初始化。

`dataset(seed=2026)` 在四個象限分別抽取三筆座標，座標絕對值在 [0.25,0.95]。對訓練資料／驗證資料／測試資料各生成 12 筆，共 36 筆；每個資料分組各六筆類別 0／1，且沒有重複樣本。固定隨機種子和實際樣本都保存。

這是離座標軸有間隔的小型合成資料，測試成績只能描述這份資料分組。它沒有涵蓋所有接近 XOR 邊界的樣本，也不是現實資料集的泛化估計。

| 資料分組 | 用途 | 是否影響權重／選擇 |
|---|---|---|
| 訓練資料 | 建立縮放規則、計算最佳化器損失 | 是 |
| 驗證資料 | 比較兩個隨機種子的最終模型 | 只選隨機種子，不更新權重 |
| 測試資料 | 選定隨機種子後評分 | 否 |

縮放器（scaler）保存每欄訓練資料的最小值與最大值，把不同尺度的輸入轉到指定範圍。截斷（clipping）把越界值改成邊界值，旗標則記錄哪些數值被改動。

沿用 Day 11 的僅使用訓練資料的最小值／最大值縮放器；驗證資料／測試資料只套用轉換。越界會截斷並記錄旗標。先切分再建立規則，能避免資料洩漏，也就是驗證或測試資料提前影響模型與前處理設定。[D14]

## 2. 把 ZZ 讀出轉成分類機率

特徵是描述輸入的數值，可調電路模板（ansatz）是含有權重的操作組合；權重由訓練調整，資料本身不因此改寫。`positive_half` 將已縮放的 `x` 用 `π(x+1)/2` 轉成 0 到 π 的旋轉角度。

固定 Day 14 `Config('angle',1)`，positive_half 角度映射、兩個量子位元、四個可訓練權重與 ZZ 可觀測量。

```text
f(x,w) = ⟨Z0 Z1⟩
p(class 1 | x,w) = (1 − f(x,w))/2
predicted class = 1 if p >= 0.5 else 0
```

可觀測量指定要量測的量；ZZ 是 `Z0Z1` 的簡寫，Z0、Z1 分別對應兩個量子位元。這個量在 `00`／`11` 的結果為 +1，在 `01`／`10` 為 −1，這些可能的量測值稱為本徵值。期望值 `f` 是依機率計算的平均值。因此這裡 p1 是**奇數同位性的量測機率，也就是兩個位元中恰有一個 1 的機率**，不是任意把期望值稱作機率。類別 1 的語意由本章指定的 XOR 標籤定義；機率校準是檢查預測機率與實際發生比例是否吻合，例如預測 80% 的樣本是否約有 80% 屬於該類別；本章並未驗證這件事。

`probabilities` 允許 1e-5 以內的模擬器浮點越界後截斷到合法機率，明顯超過範圍則拒絕。p=0.5 的平手固定歸類別 1，比較基準與模型共用此規則。

## 3. 損失與最佳化器

使用 Brier 損失，也就是二元分類機率與標籤的平均平方誤差：

```text
L(w) = mean((p1(x,w) − y)²)
```

式中的 `p1` 是類別 1 的預測機率，`y` 是 0 或 1 的正確標籤，`mean` 表示對所有樣本取平均。例如 `p1=0.8`、`y=1`，該筆平方誤差是 0.04。主控端（host）是安排電路執行與處理結果的一般 Python 程式。座標搜尋每次嘗試增減一個權重，只有誤差降低時才更新。

它和 Day 10 的期望值回歸的平均平方誤差目標不同，但同樣可由主控端計算，再交給座標搜尋：

```python
def objective(weights):
    p = probabilities(predict_batch(train_scaled, weights, CONFIG))
    return metrics(train_labels, p)['brier']

weights, history, reason = coordinate_search(
    objective, initialize(1, seed), sweeps=16)
```

初始化是訓練開始前指定權重；步長是每次嘗試增減的量。目標函數是回傳整批訓練誤差的函式，每次呼叫都要重新計算預測。

預先固定隨機種子 42／43、每個隨機種子最多 16 輪、初始步長=0.4、步長停止門檻=1e-4。沒有根據測試資料改電路、預算或分類門檻。每輪四個權重的正負候選共八次目標函數；完整 16 輪為 129 次目標函數、1,548 次訓練 observe。初始化、驗證資料、測試資料與圖表的額外評估須分開理解。

訓練採無噪聲精確期望值，`shots_count=-1`；不把抽樣波動放進候選比較。紀錄保存所有權重、損失、步長與累積評估次數。

## 4. 驗證資料選擇與測試資料評分

兩個隨機種子都跑完後，取**最終驗證資料 Brier 最低者**，相同時取隨機種子較小者。不從紀錄裡挑最佳訓練輪次。驗證資料曲線是每輪權重的 NumPy 參考值評估，只用於圖示；選隨機種子的最終驗證資料值使用 CUDA-Q。

選定模型保存模型檢查點，重新載入後才計算各資料分組的預測與測試資料指標：

- 準確率：依固定 0.5 分類門檻分類。
- Brier：機率輸出的平方誤差。
- 混淆矩陣：列是真實類別 0／1，欄是預測類別 0／1。
- 常數比較基準：所有樣本都輸出**訓練資料標籤平均值**，本例為 0.5。

比較基準（baseline）提供同一任務的參考方法。混淆矩陣將各種正確與錯誤分類分開計數；例如真實類別 0、預測類別 1 的格子，記錄被誤判為 1 的數量。

此比較基準是最低限度的合理性對照，不是充分的一般機器學習方法的效能對照。線性、非線性與 QML 的公平比較依學習路線留待後續章節。

本日的程式通過條件檢查數值正確性、兩個隨機種子的訓練資料損失下降與量測次數總數，沒有用測試資料準確率當測試成功的門檻，避免為了通過測試而調整測試資料成績。

## 5. 模型檢查點保存模型語意

模型檢查點（checkpoint）是保存後可重新載入的模型紀錄。只有權重不足以還原預測，還需要知道資料如何轉換、欄位如何排列，以及輸出如何判成類別。檔案格式版本讓載入程式辨認這份紀錄使用哪套欄位規則。

`checkpoint.json` 包含檔案格式版本、編碼／層數、positive_half 映射方式、輸出讀取、分類門檻、欄位順序、標籤規則、縮放器、權重與被選中的隨機種子。

```bash
OMP_NUM_THREADS=1 python articles/day15/demo.py --features -0.7 0.7
```

示範讀取原始座標，套用保存的縮放器，印出 p1、類別與截斷旗標。預設載入 CPU 模型檢查點；執行後端與模型檢查點路徑可分別指定：

```bash
OMP_NUM_THREADS=1 python articles/day15/demo.py \
  --checkpoint results/day15/nvidia/checkpoint.json \
  --backend nvidia --features 0.7 0.7
```

載入程式拒絕不支援的檔案格式、設定、映射方式、輸出讀取、分類門檻及錯誤權重，避免拿別種模型的參數直接推論。

## 6. 訓練曲線與決策邊界

訓練曲線呈現誤差如何隨輪次變化；決策邊界分隔模型判為不同類別的區域。Matplotlib 是 Python 繪圖套件，PNG 是圖片格式。

[plot_results.py](plot_results.py) 使用 Matplotlib 匯出兩張 PNG：

- 訓練曲線：兩個隨機種子的訓練資料 Brier 與驗證資料參考值 Brier，包含初始化第 0 輪。
- 決策邊界：保存的模型在原始座標 [−1,1]²、81×81 網格上的 p1，黑線為 p1=0.5，散點標示三個資料分組與真實標籤。

NumPy 是 Python 數值運算套件；網格是在平面上均勻排列的取樣點，用來畫出預測分布，NPZ 是保存 NumPy 陣列的檔案格式。

網格採 Day 14 的 NumPy 矩陣參考值評估，避免為純展示另外跑 6,561 次 CUDA-Q 電路；**不是 GPU 決策邊界計算的效能評測**。CUDA-Q 與參考值的逐筆預測已在資料資料分組上核對，網格原始值另外保存成 `boundary_grid.npz`。網格也使用保存的縮放器和截斷政策，超過訓練資料範圍的區域可能變平。

![CPU training curve](../../results/day15/qpp-cpu/training_curve.png)

![CPU decision boundary](../../results/day15/qpp-cpu/decision_boundary.png)

圖表只展示選定結果，沒有看到測試資料圖後再調參。本日不宣稱決策邊界 在整個連續平面都符合 XOR。

## 7. 獨立環境與重跑

CPU 是中央處理器，GPU 是擅長平行運算的圖形處理器；執行後端（backend）指定實際使用的模擬器，本章預設使用 CPU，`nvidia` 使用 GPU。`.venv` 是專案獨立保存套件的虛擬環境，`OMP_NUM_THREADS=1` 將 CPU 平行工作的執行緒數設為 1。JSON 用欄位名稱保存資料，CSV 是表格文字檔。

沿用 `.venv`，本日增加固定版本 Matplotlib 與繪圖依賴；不更新 CUDA-Q／NumPy 套件組合。

```bash
source .venv/bin/activate
python -m pip install -r requirements-day15.txt

OMP_NUM_THREADS=1 python articles/day15/experiment.py
OMP_NUM_THREADS=1 python articles/day15/experiment.py --backend nvidia

OMP_NUM_THREADS=1 python articles/day15/plot_results.py
OMP_NUM_THREADS=1 python articles/day15/plot_results.py --backend nvidia
```

實驗預設寫入 `results/day15/<backend>/`，重跑會更新；可用 `--output-dir /tmp/day15-check` 另存。繪圖工具目前讀取預設執行後端目錄。

每個執行後端保存資料集、兩個候選模型的訓練紀錄、訓練 CSV、模型檢查點、各資料分組預測／評估指標、測試資料抽樣與摘要，另有兩張 PNG 和決策邊界網格。完整實測數字見 [結果紀錄](../../results/day15/README.md)。

## 8. 有限次量測與測試

精確期望值由模擬器保存的狀態直接計算，仍有浮點誤差；浮點數是電腦以有限位數表示的小數。量測計數（counts）是各結果出現的次數。選定模型後，測試資料每筆另抽樣 1,000 次，從 01／10 量測計數算 p1，保存隨機種子與抽樣測試指標。這是精確訓練後的有限次量測評估，不是有限次量測訓練；靠近 0.5 的樣本可能因抽樣變動而翻轉分類。

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day15 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY15_TARGET=nvidia python -m unittest discover -s articles/day15 -p 'test_*.py' -v
```

四個測試涵蓋資料分組不重疊、標籤與固定隨機種子、機率轉換／分類門檻／混淆矩陣、浮點容許範圍，以及模型保存後重新載入的 CUDA-Q／NumPy 預測核對。

Day 11–15 至此交付第一個量子分類器、訓練曲線、決策邊界。下一篇 [Day 16](../day16/README.md) 討論量子神經網路（QNN，含有可訓練量子電路的模型）與一般神經網路的關係。這些結果仍是無噪聲模擬器上的小型 XOR 示範，沒有真實量子處理器（QPU）實測、速度優勢或真實任務泛化結論；泛化是模型處理未參與訓練的新資料的能力。

## 9. 來源

[D14] [scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：資料切分、前處理與資料洩漏；本日以 NumPy 實作，沒有新增 scikit-learn 依賴。查閱日期 2026-09-07。

電路、縮放器與最佳化器分別沿用 Day 14、11、10。同位性機率與 Brier 定義在本文明示；來源索引見 [REFERENCES.md](../../REFERENCES.md)。
