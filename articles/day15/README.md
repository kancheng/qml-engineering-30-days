# Day 15｜第一個 CUDA-Q Quantum Classifier

## 本章摘要｜初學者學習筆記

### 中文

[Day14](../day14/README.md) 將資料編碼、可訓練電路、讀出與最佳化器組合成共用模型，並透過短訓練確認元件能一起運作。Day15 接著加入明確的分類標籤、資料切分、模型保存與評分，完成可以重新載入並產生分類結果的 XOR 示範。

這一章目標在於理解 **「量子電路的數值輸出如何成為分類結果，以及如何評估與保存一個訓練完成的模型」**，把前幾章的模型骨架延伸成完整的機器學習流程。本章使用兩個座標的符號判斷 XOR 類別：符號相異為 class 1，同號為 class 0。電路的 `Z0Z1` 期望值透過 `p1=(1−f)/2` 轉成兩個量測位元不同的機率，再以固定的 0.5 門檻決定預測類別。訓練以 Brier loss（預測機率與標籤的平均平方誤差）更新權重；train 決定縮放規則與訓練更新，validation 選擇兩個初始化結果中的最終模型，test 則在選定後用於評分。重點除了 loss 是否下降，也包括分類錯在哪裡、機率輸出與標籤的差距，以及重新載入模型後能否重現預測。因此，本章同時保存 checkpoint（模型檢查點）、訓練曲線與決策邊界，並以常數預測作為基本對照。讀完本章，應能說明分類機率、門檻、資料切分與模型保存的用途，理解精確期望值訓練與最終有限 shots 評估的差別，並掌握小型合成 XOR 結果所能支持的結論範圍。

### English

[Day14](../day14/README.md) combined data encoding, a trainable circuit, readout, and optimization into a shared model, using short training runs to check that the components work together. Day15 adds explicit class labels, data splits, model persistence, and evaluation to complete an XOR example that can be reloaded and used for classification.

This chapter aims to explain **how numerical circuit outputs become class predictions, and how a trained model is evaluated and saved**, extending the earlier model structure into a complete machine learning workflow. The task assigns class 1 to coordinate pairs with opposite signs and class 0 to pairs with matching signs. The circuit's `Z0Z1` expectation is converted through `p1=(1−f)/2` into the probability of differing measurement bits, then a fixed threshold of 0.5 determines the predicted class. Training updates weights using Brier loss, the mean squared difference between predicted probabilities and labels. Training data determines scaling rules and weight updates, validation data selects between the final models from two initializations, and test data is used for scoring after selection. Beyond loss reduction, the chapter examines classification errors, probability errors, and whether reloading the model reproduces predictions. Checkpoints, training curves, and decision boundaries are saved, with a constant predictor providing a basic comparison. The learning goal is to explain the roles of probabilities, thresholds, data splits, and model persistence; distinguish exact-expectation training from final finite-shot evaluation; and understand the limited conclusions supported by a small synthetic XOR task.

---

Day 14 已經能把 features、Ansatz、readout 與 optimizer 接起來。今天完成第三階段交付：**有資料切分、可載入模型、評分、Training Curve 與 Decision Boundary 的 XOR classifier**。

程式入口：[classifier.py](classifier.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[plot_results.py](plot_results.py)。本日採用 XOR，沒有另外訓練 Moons。

## 1. 明確定義任務與資料

給定兩個原始座標，符號相異為 class 1，同號為 class 0：

```python
y = (x0 * x1 < 0).astype(int)
```

`dataset(seed=2026)` 在四個象限分別抽取三筆座標，座標絕對值在 [0.25,0.95]。對 train／validation／test 各生成 12 筆，共 36 筆；每個 split 各六筆 class 0／1，且沒有重複樣本。固定 seed 和實際樣本都保存。

這是離座標軸有間隔的小型合成資料，測試成績只能描述這份 split。它沒有涵蓋所有接近 XOR 邊界的樣本，也不是現實資料集的泛化估計。

| Split | 用途 | 是否影響 weights／選擇 |
|---|---|---|
| train | fit scaler、計算 optimizer loss | 是 |
| validation | 比較兩個 seed 的最終模型 | 只選 seed，不更新 weights |
| test | 選定 seed 後評分 | 否 |

沿用 Day 11 的 train-only min/max scaler；validation／test 只 transform。越界會 clipping 並記錄 flags。資料先分開再 fit preprocessing，是避免資料洩漏的基本邊界。[D14]

## 2. 把 ZZ 讀出轉成分類機率

固定 Day 14 `Config('angle',1)`，positive_half angle map、兩個 qubit、四個可訓練 weights 與 ZZ observable。

```text
f(x,w) = ⟨Z0 Z1⟩
p(class 1 | x,w) = (1 − f(x,w))/2
predicted class = 1 if p >= 0.5 else 0
```

ZZ 在 00／11 的 eigenvalue 為 +1，在 01／10 為 −1。因此這裡 p1 是**奇數 parity 的量測機率**，不是任意把 expectation 稱作 probability。class 1 的語意由本章指定的 XOR label 定義；這個映射不保證已完成機率校準。

`probabilities` 允許 1e-5 以內的 simulator 浮點越界後 clipping 到合法機率，明顯超過範圍則拒絕。p=0.5 的平手固定歸 class 1，baseline 與模型共用此規則。

## 3. Loss 與 Optimizer

使用 Brier loss，也就是 binary probability 與 label 的平均平方誤差：

```text
L(w) = mean((p1(x,w) − y)²)
```

它和 Day 10 的 expectation regression MSE 目標不同，但同樣可由 host 計算，再交給座標搜尋：

```python
def objective(weights):
    p = probabilities(predict_batch(train_scaled, weights, CONFIG))
    return metrics(train_labels, p)['brier']

weights, history, reason = coordinate_search(
    objective, initialize(1, seed), sweeps=16)
```

預先固定 seeds 42／43、每個 seed 最多 16 輪、初始 step=0.4、step tolerance=1e-4。沒有根據 test 改電路、預算或 threshold。每輪四個 weights 的正負候選共八次 objective；完整 16 輪為 129 次 objective、1,548 次訓練 observe。初始化、validation、test 與圖表的額外評估須分開理解。

訓練採無噪聲 exact expectation，`shots_count=-1`；不把抽樣波動放進候選比較。history 保存所有 weights、loss、step 與累積 evaluations。

## 4. Validation 選擇與 Test 評分

兩個 seeds 都跑完後，取**最終 validation Brier 最低者**，相同時取 seed 較小者。不從 history 裡挑最佳 epoch。validation 曲線是每輪 weights 的 NumPy reference 評估，只用於圖示；選 seed 的最終 validation 值使用 CUDA-Q。

選定模型保存 checkpoint，重新載入後才計算各 split 的預測與 test 指標：

- Accuracy：依固定 0.5 threshold 分類。
- Brier：機率輸出的平方誤差。
- Confusion matrix：row 是真實 class 0／1，column 是預測 class 0／1。
- Constant baseline：所有樣本都輸出**train label 平均值**，本例為 0.5。

此 baseline 是最低限度的合理性對照，不是充分的 classical ML benchmark。線性、非線性與 QML 的公平比較依 Roadmap 留待後續章節。

本日的程式通過條件檢查數值正確性、兩個 seeds 的 train loss 下降與 shots 總數，沒有用 test accuracy 當測試成功的門檻，避免為了通過測試而調整 test 成績。

## 5. Checkpoint 保存模型語意

`checkpoint.json` 包含 schema version、encoding／layers、positive_half mapping、readout、threshold、欄位順序、label 規則、scaler、weights 與被選中的 seed。

```bash
OMP_NUM_THREADS=1 python articles/day15/demo.py --features -0.7 0.7
```

demo 讀取原始座標，套用保存的 scaler，印出 p1、class 與 clipping flags。預設載入 CPU checkpoint；backend 與 checkpoint 路徑可分別指定：

```bash
OMP_NUM_THREADS=1 python articles/day15/demo.py \
  --checkpoint results/day15/nvidia/checkpoint.json \
  --backend nvidia --features 0.7 0.7
```

loader 拒絕不支援的 schema、config、mapping、readout、threshold 及錯誤 weights，避免拿別種模型的參數直接推論。

## 6. Training Curve 與 Decision Boundary

[plot_results.py](plot_results.py) 使用 Matplotlib 匯出兩張 PNG：

- Training Curve：兩個 seeds 的 train Brier 與 validation reference Brier，包含初始化第 0 輪。
- Decision Boundary：保存的模型在 raw [−1,1]²、81×81 網格上的 p1，黑線為 p1=0.5，散點標示三個 split 與真實 labels。

網格採 Day 14 的 NumPy 矩陣 reference 評估，避免為純展示另外跑 6,561 次 CUDA-Q 電路；**不是 GPU boundary benchmark**。CUDA-Q 與 reference 的逐筆預測已在資料 split 上核對，grid 原始值另外保存成 `boundary_grid.npz`。網格也使用保存的 scaler 和 clipping 政策，超過 train 範圍的區域可能變平。

![CPU training curve](../../results/day15/qpp-cpu/training_curve.png)

![CPU decision boundary](../../results/day15/qpp-cpu/decision_boundary.png)

圖表只展示選定結果，沒有看到 test 圖後再調參。本日不宣稱 boundary 在整個連續平面都符合 XOR。

## 7. 獨立環境與重跑

沿用 `.venv`，本日增加固定版本 Matplotlib 與繪圖依賴；不更新 CUDA-Q／NumPy stack。

```bash
source .venv/bin/activate
python -m pip install -r requirements-day15.txt

OMP_NUM_THREADS=1 python articles/day15/experiment.py
OMP_NUM_THREADS=1 python articles/day15/experiment.py --backend nvidia

OMP_NUM_THREADS=1 python articles/day15/plot_results.py
OMP_NUM_THREADS=1 python articles/day15/plot_results.py --backend nvidia
```

experiment 預設寫入 `results/day15/<backend>/`，重跑會更新；可用 `--output-dir /tmp/day15-check` 另存。繪圖工具目前讀取預設 backend 目錄。

每個 backend 保存 dataset、兩個 candidate histories、training CSV、checkpoint、各 split predictions／metrics、test sampling 與 summary，另有兩張 PNG 和 boundary grid。完整實測數字見 [結果紀錄](../../results/day15/README.md)。

## 8. Finite-shot 與測試

選定模型後，test 每筆另抽 1,000 shots，從 01／10 counts 算 p1，保存 seed 與 sampled test metrics。這是 exact 訓練後的有限次量測評估，不是 finite-shot 訓練；靠近 0.5 的樣本可能因抽樣變動而翻轉分類。

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day15 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY15_TARGET=nvidia python -m unittest discover -s articles/day15 -p 'test_*.py' -v
```

四個測試涵蓋 split 不重疊、label 與固定 seed、機率轉換／threshold／混淆矩陣、浮點容許範圍，以及 checkpoint roundtrip 的 CUDA-Q／NumPy 預測核對。

Day 11–15 至此交付第一個 Quantum Classifier、Training Curve、Decision Boundary。下一篇 [Day 16](../day16/README.md) 討論 QNN 與 neural network 的關係。這些結果仍是無噪聲模擬器上的小型 XOR 示範，沒有 QPU、速度優勢或真實任務泛化結論。

## 9. 來源

[D14] [scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：資料切分、preprocessing 與資料洩漏；本日以 NumPy 實作，沒有新增 scikit-learn 依賴。查閱日期 2026-09-07。

電路、scaler 與 optimizer 分別沿用 Day 14、11、10。parity probability 與 Brier 定義在本文明示；來源索引見 [REFERENCES.md](../../REFERENCES.md)。
