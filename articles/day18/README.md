# Day 18｜一般機器學習與量子模型：固定條件下的比較

[Day17](../day17/README.md) 實作有限差分、參數位移法與損失函數的鏈式法則，核對梯度並記錄額外電路評估的成本。Day18 接著把重點轉向模型比較，透過預先固定的資料、評分與訓練預算，檢查 Classical ML 與 QML 在同一任務下的表現。

**模型比較需要先固定資料、評分方法與訓練預算，才能解讀差異來自哪些條件。** 即使規則相同，有限的實驗仍只能回答特定設定下的問題。本章以新的小型 XOR 資料切分，比較 logistic 模型、MLP（多層感知器）與 VQC（變分量子分類器），另加入固定輸出訓練標籤平均值的常數對照。三種可訓練模型共用只由 train 擬合的縮放規則、Brier loss 與不需梯度的座標搜尋，因此本章沒有使用 Day17 的參數位移法。每次訓練固定為 109 次 objective evaluations（完整訓練損失評估），避免相同迴圈次數卻給予不同模型不同的候選比較預算；但相同評估次數仍不代表相同執行時間或充分訓練。Validation 只負責在各模型的兩個初始化結果中選擇最終版本，test 則在選定後評分。準確率、機率誤差與運算成本需要一起保存。線性模型在 XOR 表現較差，或某個模型在這次實驗較好，都不足以判定整個模型家族的優劣或量子優勢。

[Day17](../day17/README.md) implemented finite differences, parameter-shift, and the loss chain rule, checking gradients and recording the cost of additional circuit evaluations. Day18 turns to model comparison, using predefined data, evaluation rules, and training budgets to examine classical ML and QML on the same task.

**Model comparisons need predefined data, scoring rules, and training budgets so that differences can be interpreted in context.** Even shared rules only establish results under the tested conditions. A new, small XOR split is used to compare a logistic model, a multilayer perceptron (MLP), and a variational quantum classifier (VQC), alongside a constant predictor based on the mean training label. All three trainable models share training-only scaling, Brier loss, and derivative-free coordinate search, so Day17's parameter-shift method is not used here. Each fit receives 109 objective evaluations—evaluations of the full training loss—to avoid allocating different candidate-comparison budgets through equal loop counts. Equal evaluation counts still do not imply equal execution time or sufficient training. Validation selects between the final models from two initializations within each model family; test scoring follows selection. Accuracy, probability error, and cost need to be recorded together. A linear model struggling with XOR, or one model performing better here, does not establish model-family superiority or quantum advantage.

---

今天把 Day 16 的 MLP 與 Day 14 的 VQC 放在同一個比較協定下，加入 logistic 比較基準，實際訓練後再評分。**公平在本文指比較條件公開且一致，不代表一個最佳化器、單一資料切分就能判定模型家族優劣。**

程式：[benchmark.py](benchmark.py)、[plot_results.py](plot_results.py)。本日採用共同 Brier 目標函數的控制實驗，所有差異與成本都保留。

## 1. 先固定比較規則

QML 是量子機器學習，VQC 是變分量子分類器，也就是反覆調整電路參數來預測類別的模型；MLP 是多層感知器，由多層一般神經網路計算組成。XOR 是互斥或，本例依兩個座標是否異號分成兩類。

每次比較實驗在訓練開始前寫入 `protocol.json`、`dataset.json`、`scaler.json`：

| 項目 | 固定設定 |
|---|---|
| 資料集 | Day 15 XOR 產生規則，新隨機種子=2027 |
| 資料切分 | 訓練資料／驗證資料／測試資料各 12 筆，各類別 6 筆 |
| 前處理 | 同一份僅由訓練資料計算的最小值／最大值縮放；越界截斷 |
| 損失 | Brier：mean((p1−y)²) |
| 最佳化器 | 同一份不需梯度的座標搜尋 |
| 每次訓練預算 | 109 次目標函數評估 |
| 初始化 | normal(0,0.2)，每模型隨機種子 42／43 |
| 選模 | 每個模型選最終驗證 Brier 較低的隨機種子 |
| 分類門檻 | 固定 p1≥0.5 為類別 1 |
| 測試資料 | 各模型選定隨機種子後才評分 |

訓練資料用來更新權重，驗證資料用來選擇設定，測試資料留到模型選定後才評分。縮放器將數值轉到指定範圍，截斷則把超出範圍的值改成邊界值。隨機種子（seed）控制隨機序列，方便重現資料與起始權重。`normal(0,0.2)` 從平均值 0、標準差 0.2 的常態分布抽樣；標準差描述數值的分散程度。

SHA-256 是根據檔案內容計算的摘要，用來檢查兩次實驗是否使用相同資料。資料 SHA-256 保存於比較規則，CPU／GPU 兩次執行必須一致。使用新資料切分是為了不把先前已檢視的 Day 15 測試資料當成新的盲測；它仍是同一個小型合成 XOR 分布，不是獨立真實資料集。

前處理僅在訓練資料訓練、測試資料不參與模型選擇，沿用官方模型評估建議。[D14] 看完本次測試資料不再調整預算或權重。

## 2. 三個模型與一個常數基準

| 模型 | 計算 | 可訓練參數 |
|---|---|---:|
| logistic | sigmoid(x·w+b) | 3 |
| MLP | 2→tanh(2)→sigmoid(1)，含偏差 | 9 |
| VQC | 兩個量子位元、positive_half 編碼、一層可調電路模板、p1=(1−ZZ)/2 | 4 |
| constant | 訓練資料標籤平均值=0.5 | 不訓練 |

logistic 模型將輸入乘上權重、加上偏差，再用 `sigmoid(z)=1/(1+exp(−z))` 轉到 0 與 1 之間。偏差是可調的加法常數。MLP 的隱藏層位於輸入與輸出之間，`tanh` 將其數值轉到 −1 與 1 之間，使模型能形成非線性關係。VQC 的可調電路模板（ansatz）提供旋轉權重，ZZ 期望值則把兩位元相同記為 +1、不同記為 −1，再依機率平均。

**本日 logistic 使用 Brier 損失，而非標準最大概似邏輯斯迴歸的對數損失。** Brier 損失是預測機率與 0／1 答案的平均平方誤差；最大概似方法則讓已觀察答案的機率盡可能大，常用等價的負對數損失來求解。兩種目標的誤差計算規則不同。官方對數損失文件明確說明常見邏輯斯迴歸的負對數概似目標。[D16] 這裡保留 logistic 輸出轉換，刻意統一目標函數；不能把它稱作 scikit-learn LogisticRegression 的效能或最佳一般模型比較基準。

分類邊界分隔預測為 0 與 1 的區域。logistic 在本例的 0.5 邊界是一條直線，無法單靠它分開 XOR 中對角分布的兩類；MLP 與 VQC 能形成不同的非線性函數。XOR 刻意需要非線性，因此 logistic 結果差不表示一般機器學習整體不如 QML。MLP 容量、初始化及最佳化器也沒有經廣泛搜尋。

## 3. 相同迴圈次數不等於相同工作量

座標搜尋每次只改一個權重，嘗試增加或減少，只有損失降低時才接受。梯度描述參數微小改變時目標的變化率，這個方法直接比較候選值，不需計算梯度。

Day 10 座標搜尋每輪對 P 個權重各試正負方向，需要 2P 次目標函數。如果讓三個模型都跑相同輪數，九參數 MLP 會得到比三參數 logistic 更多次評估。

目標函數每次計算整批訓練資料的平均誤差。步長是單次嘗試增減權重的幅度，完整一輪會依序處理所有權重。

本日改用**固定 109 次目標函數評估**：一次初始損失，剩餘 108 次形成 54 對候選。只接受完整的正負候選配對，不超出預算；每一對之後保存已接受的權重與損失。每輪沒有改善才把步長=0.4 減半；預算耗盡直接停止。

| 模型 | 完整完整輪次 | 最後未完成整輪的更新 |
|---|---:|---|
| logistic（3 參數） | 18 | 無 |
| MLP（9 參數） | 6 | 無 |
| VQC（4 參數） | 13 | 再處理兩個座標 |

未完成整輪的更新不觸發步長縮減。所有模型都取得同樣的候選評估數，但參數較多的模型更新每個座標的機會較少；座標順序也可能影響最後結果。這是此資源分配規則的取捨，不是唯一公平定義。

## 4. 相同目標函數次數仍不等於相同成本

NumPy 是 Python 數值運算套件；批次矩陣運算將多筆樣本一起排成數值表計算。`observe` 取得量子電路指定量測的期望值，精確模式由模擬器保存的狀態直接求值，仍有有限數值精度的誤差。

一個目標函數都看相同 12 筆訓練資料資料：

- logistic／MLP 使用 NumPy 批次矩陣運算。
- VQC 逐筆執行精確 `observe`，每次訓練有 109×12=1,308 次訓練電路呼叫。
- 每模型各兩個隨機種子；VQC 每執行後端共 2,616 次訓練 observe，驗證資料／測試資料／參考值另計。

本日選擇不需梯度的共同最佳化器，避免拿不同梯度估計成本假裝同等；Day 17 的參數位移法沒有在此呼叫。這也不代表座標搜尋是每個模型的最佳最佳化器。

CPU 是中央處理器，GPU 是擅長平行運算的圖形處理器；執行後端指定使用的模擬器。JIT 是即時編譯，執行時將程式轉成可用的運算形式；主控端成本包括 Python 呼叫與流程處理。暖機是正式計時前先執行，減少首次啟動成本的影響；快取則保存可重用的結果或編譯產物。

記錄 `fit_seconds_including_first_compilation`，包含訓練內的首次 JIT 及主控端額外成本，不包含後續驗證資料／測試資料。CPU／GPU 實驗可能同時執行、後續隨機種子也可能使用快取，未做隔離、暖機或重複計時。因此此耗時僅供工程成本追蹤，**不能推論 GPU 加速幅度或一般／量子演算法複雜度**。GPU 執行裡的一般模型仍在 NumPy CPU 上執行。

## 5. 選模與報告

每種模型內只依最終驗證 Brier 選隨機種子，平手取較小隨機種子；不挑紀錄的最低驗證資料輪次，也不以測試資料選最佳模型家族。

準確率是分類正確的比例；混淆矩陣分別記錄每種真實類別被判成哪種類別的次數。分類門檻將機率轉成類別，本例固定在 0.5。

報告訓練資料／驗證資料／測試資料 Brier、準確率與混淆矩陣；列是真實類別、欄是預測類別。常數 p1=0.5 依相同分類門檻全預測 1，所以類別均衡的測試資料的準確率=50%、Brier=0.25。

JSON 是以欄位名稱保存資料的文字格式。報告不只公布表現最好者：`candidates.json` 保存所有隨機種子的驗證資料、初始／最終權重、每對候選後的訓練資料紀錄、預算與時間。`selected.json` 保存每模型選定權重及所有資料切分機率與評估指標。原始標籤與原始特徵在資料集，縮放規則在縮放器，模型定義在比較規則。

## 6. 如何解讀結果

![Benchmark](../../results/day18/qpp-cpu/benchmark.png)

曲線橫軸是目標函數評估，不是完整輪次或秒。右圖的測試資料 Brier 只描述此 12 筆資料切分。數字、候選及時間見 [結果紀錄](../../results/day18/README.md)。

兩個初始化隨機種子不等於兩個獨立資料切分；CPU／GPU 同一資料的結果也不是額外統計樣本。顯著性檢定在指定假設下評估觀察差異的統計證據；信賴區間描述估計方法對未知真值的不確定性。這些需要合適的重複與評估設計，本章未進行，也不宣稱量子方法優於一般方法。若要推廣結論，仍需更大資料、多資料切分、合理的各模型調整設定的預算與真正的一般模型比較基準，後續 Day 20／25 會繼續擴充任務。

## 7. 重跑與測試

`.venv` 是專案獨立保存 Python 套件的虛擬環境；`OMP_NUM_THREADS=1` 將 CPU 平行工作的執行緒數設為 1。

沿用 `.venv`，無新增依賴；[requirements-day18.txt](../../requirements-day18.txt)。從專案根目錄：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day18/benchmark.py
OMP_NUM_THREADS=1 python articles/day18/benchmark.py --backend nvidia
OMP_NUM_THREADS=1 python articles/day18/plot_results.py
OMP_NUM_THREADS=1 python articles/day18/plot_results.py --backend nvidia

OMP_NUM_THREADS=1 python -m unittest discover -s articles/day18 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY18_TARGET=nvidia python -m unittest discover -s articles/day18 -p 'test_*.py' -v
```

預設寫入 `results/day18/<backend>/`，重跑會更新。比較實驗支援 `--output-dir /tmp/day18-check`；繪圖工具讀取預設目錄。

推論是使用已選定模型產生預測，不再更新權重。載入已保存的三個模型做單筆推論：

```bash
OMP_NUM_THREADS=1 python articles/day18/demo.py --features -0.7 0.7
```

四個測試涵蓋三種參數維度都恰好用完 109 次評估、損失不增加、輸入不被改動、logistic 解析輸出、資料切分／縮放器邊界與 VQC NumPy 參考值。實驗通過條件只要求預算、損失與數值核對，不要求某個模型擊敗另一個。

所有 VQC 結果為無噪聲精確模擬器；本日沒有有限次量測或真實量子處理器（QPU）比較實驗。下一篇 [Day 19](../day19/README.md) 將探索一般神經網路層與量子電路層的混合神經網路。

## 8. 來源

- [D14] [scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：訓練資料-only 前處理與資料洩漏。
- [D16] [scikit-learn log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)：標準邏輯斯迴歸的對數損失目標；與本文 Brier 控制實驗區別。

查閱日期 2026-09-07。本日採 NumPy 自訂實作，沒有安裝 scikit-learn；共用索引見 [REFERENCES.md](../../REFERENCES.md)。
