# Day 25｜第二個真實資料集：Wine 分類實驗

[Day24](../day24/README.md) 比較不同電路與起始權重下的梯度，也就是參數小幅改動時目標函數的變化率，釐清梯度過小與 Barren Plateau（貧瘠高原）的診斷條件。Day25 回到分類任務，將既有模型移到 Wine 資料集，觀察同一套流程面對更多特徵與不同類別比例時的表現。

**第二個資料集可以檢查建模流程能否用在 Iris 以外的資料。** 資料縮放與壓縮規則需要依新的訓練資料重新建立，模型結構與比較預算則保持一致。本章使用 UCI Wine 中兩個葡萄栽培品種的 119 筆紀錄，以十三個化學測量值判斷類別，並非預測葡萄酒品質。傳統神經網路、可訓練量子電路與兩者結合的混合模型，共用由十三個特徵壓縮成兩個座標的輸入；另加入使用完整十三個特徵的傳統模型，觀察壓縮造成的差距。完整輸入模型取得的資訊不同，不能將成績差異全部歸因於量子層。實驗以驗證資料選擇訓練結果，再用測試資料評分，並記錄資料處理與計算成本。所有訓練先由 NumPy 在 CPU 完成，再由 CUDA-Q 的 CPU／GPU 模擬器核對固定權重的輸出。小型測試集的成績差異不代表穩定優勢，這次實驗也沒有驗證 GPU 訓練加速或解決貧瘠高原。

[Day24](../day24/README.md) compared gradient distributions across circuit sizes, objectives, and initializations, clarifying the evidence needed to diagnose small gradients and barren plateaus. Day25 returns to classification, transferring the existing classical, quantum, and hybrid models to Wine to examine the workflow with more features and different class proportions.

**A second dataset checks whether the modeling workflow can be reused beyond Iris.** Preprocessing must be fitted again on the new training data, while model structures and comparison budgets remain fixed. The task uses 119 records from two UCI Wine cultivars and thirteen chemical features for binary classification, rather than wine-quality prediction. The MLP, VQC, and hybrid model share training-only standardization and principal component analysis (PCA), reducing thirteen features to two. A classical model using all thirteen features provides a comparison between compressed representations and full inputs. Because this model receives different information, performance differences cannot be attributed entirely to the quantum layer. Models receive the same loss-evaluation budget, validation selects initialization results, and test scoring follows selection. A constant predictor uses the mean training label to account for the class proportions. Training runs on the NumPy CPU, followed by CUDA-Q CPU/GPU verification with frozen weights. Data provenance, preprocessing, selection, and cost records make the comparison reproducible. Accuracy differences on a small test set do not establish a stable advantage. This experiment does not test GPU training acceleration or barren-plateau mitigation.


---

程式：[wine_models.py](wine_models.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[圖表與報告產生器](plot_results.py)、[測試](test_wine.py)、[實驗報告](../../results/day25/README.md)。沿用 `.venv` 虛擬環境，也就是專案獨立保存套件的目錄，不新增套件。

## 1. 為何需要第二個資料集？

Day20 的 Iris 鳶尾花分類結果只對當時的資料、輸入表示與訓練預算成立。Wine 有十三個化學特徵，類別比例也不同，適合檢查前處理與模型介面能否重用。特徵是描述每筆樣本的數值；前處理則是在模型接收資料前，先建立縮放或壓縮表示的規則。這仍是小型教學實驗，不能代表大型部署的效果。

UCI 是提供資料集的機器學習資料庫。Wine 包含 178 筆紀錄、十三個化學測量值與三個栽培品種。[D21] 本章預先選定原始類別 2 與 3，分別改記為標籤 0 與 1；標籤就是模型需要預測的正確類別。這樣保留 71＋48＝119 筆資料，沿用既有的二分類模型，沒有依模型成績挑選類別組合。載入程式檢查特徵與標籤皆相同的重複紀錄，本子集沒有這類重複資料。

原始檔未修改，保存於 [資料目錄](../../data/day25/README.md)，附來源、下載日期與 CC BY 4.0 授權及署名資訊。`protocol.json` 記錄實驗設定、特徵順序與 SHA-256；SHA-256 是由檔案內容計算的摘要，用來確認使用的檔案是否一致。

## 2. 資料切分與僅使用訓練資料的轉換

隨機種子用來重現資料打亂的順序。本章預先指定 2030／2031，分別打亂各類別，再取約 60% 為訓練資料、20% 為驗證資料，兩者筆數都向下取整，其餘為測試資料。

| 資料用途 | 栽培品種 2／標籤 0 | 栽培品種 3／標籤 1 | 合計 |
|---|---:|---:|---:|
| 訓練：建立轉換規則與更新權重 | 42 | 28 | 70 |
| 驗證：選擇訓練結果 | 14 | 9 | 23 |
| 測試：對選定模型評分 | 15 | 11 | 26 |

原始資料列編號從 0 開始，與全部原始特徵一起保存。同一組切分的訓練、驗證與測試資料互不重疊；不同種子產生的測試集則可能重疊，因此不是兩個獨立抽樣實驗。

三個模型共用以下轉換：

```text
raw13 → train mean/std standardization → train SVD PCA2
      → train PC minmax → clip [-1,1] → MLP / VQC / Hybrid
```

標準化先將各欄位減去訓練平均值，再除以標準差，使不同數值尺度的欄位較容易一起處理。標準差描述分散程度；此處計算時以訓練筆數為分母，對應程式中的母體標準差慣例。

PCA（主成分分析）將原始欄位線性組合成新座標，選出資料變化較大的方向。PCA2 表示保留兩個主成分，這裡透過 SVD（奇異值分解，一種把矩陣拆解以找出主要方向的計算方法）求得。主成分方向整體反號仍代表同一條軸，因此固定每個主成分中絕對值最大的係數為正，讓結果容易重現。

接著用訓練資料在兩個主成分上的最小值與最大值縮放到 [-1, 1]；截斷（clip）將超出範圍的值改成最近的邊界值。平均值、標準差、主成分與縮放範圍全部只由訓練資料決定，驗證與測試資料只套用既有規則。PCA 雖然不使用標籤，仍會從資料學習方向，若用全部資料建立規則，測試資訊就會提前進入建模流程。做法沿用 [Day20](../day20/README.md) 與 [Day21](../day21/README.md)。

完整十三維的傳統比較模型只做標準化，不做 PCA 或截斷。這個對照可檢查壓縮後模型與完整輸入的差距，但因可用資訊不同，無法單獨辨認量子層的貢獻。

## 3. 模型與固定搜尋預算

| 模型 | 結構 | 可訓練參數 | 量子位元 |
|---|---|---:|---:|
| MLP（多層感知器） | 兩個主成分 → 兩個隱藏單元 → 一個輸出 | 9 | 0 |
| VQC（變分量子分類器） | 兩個主成分 → 角度編碼 → 單層可調電路 → 輸出 | 4 | 2 |
| Hybrid（混合模型） | 兩個主成分 → 傳統編碼層 → 可調量子電路 → 傳統輸出層 | 12 | 2 |
| Logistic13 | 十三個標準化特徵 → 加權求和與 sigmoid | 14 | 0 |

MLP 是傳統神經網路，中間的隱藏單元將輸入加權組合後，透過 tanh 函數轉到 −1 與 1 之間；最後使用 sigmoid 函數，將分數轉到 0 與 1 之間。權重與偏移量都是可訓練參數，也就是訓練時調整的數值。

VQC 將資料轉成旋轉角度，再調整電路中的其他角度來學習。此處 `positive_half` 編碼以 `a=π(x+1)/2` 將 [-1, 1] 映射到 [0, π]。Ansatz 是預先選定的可調電路模板，PQC 則泛指帶有可調參數的量子電路。輸出為 `(1−⟨ZZ⟩)/2`：ZZ 對兩位元量測結果相同記 +1、不同記 −1，`⟨ZZ⟩` 是依機率計算的平均值，再轉成標籤 1 的預測值。

Hybrid 在量子電路前後加入傳統運算。前段先做仿射轉換，也就是乘上權重再加偏移，接著使用 tanh 產生編碼值；後段再以仿射轉換與 sigmoid 產生輸出。MLP、VQC 與 Hybrid 直接重用 Day20 的模型。Hybrid 仍接收 PCA2，沒有改成直接從十三維學習壓縮到兩維的瓶頸層，因此跨資料集的模型結構保持一致。

Logistic13 使用 Brier 損失訓練 sigmoid 輸出。Brier 損失是預測值與 0／1 標籤的平均平方誤差，越小表示越接近標籤。這與常見邏輯斯迴歸使用的交叉熵不同；交叉熵會對信心很高卻預測錯誤的結果給予較大的懲罰。

兩組資料切分 × 四個模型 × 初始化種子 42／43，共 **16 次訓練**。初始化是設定訓練開始前的權重，此處從平均值 0、標準差 0.2 的常態分布抽取，大部分值集中在 0 附近；混合模型輸出層的縮放係數另設為 0.8。

沿用 Day18 的座標搜尋：每次嘗試增加或減少一個權重，保留損失較低的候選。步長 0.4 表示每次嘗試的改動幅度。每次訓練評估損失函數 73 次，每次使用全部 70 筆訓練資料，共 **5,110 筆模型輸入評估**。

每個模型在每組切分中，依最終驗證 Brier 選擇初始化結果，平手取較小種子；測試資料不參與選擇。初始與最終權重、搜尋歷程、驗證結果及全部候選都有保存，沒有依測試成績追加預算。

相同的 73 次評估只代表呼叫次數一致。參數量不同，逐一調整所有參數的完整輪次就不同，每次呼叫的成本也不同，因此不代表相同執行時間或已充分收斂；收斂是指繼續更新時，結果已趨於穩定。`fit_seconds_numpy` 不包含建立 PCA 的時間，不能當作整條資料處理與訓練流程的總耗時。

## 4. 實測與結果判讀

![Wine 模型比較](../../results/day25/comparison.png)

左圖呈現驗證資料選定模型後的測試 Brier；右圖保留種子 2030 切分下，兩種初始化的訓練曲線。完整準確率、混淆矩陣、PCA 變異比例、截斷紀錄與各候選耗時見 [結果紀錄](../../results/day25/README.md)。

常數對照不讀取特徵，對每筆資料都輸出訓練資料中標籤 1 的比例，即 `28/70=0.4`。兩類筆數不同，因此這個對照比只看固定輸出 0.5 更能反映資料原有的類別比例。

準確率是分類正確的比例；此處以 0.5 為分類門檻。混淆矩陣按真實類別分列、預測類別分欄，記錄每類被判對或判錯的筆數。PCA 解釋變異比例描述保留了多少標準化訓練資料的數值變化，不代表保留了同樣比例的分類資訊。

每個測試集只有 26 筆，一筆分類差異就約為 3.85 個百分點，不能據此外推為整類模型的穩定優勢。Iris 與 Wine 的任務不同，也不能直接用準確率排出跨資料集名次。本章採預先固定的小預算，屬探索性比較，沒有透過交叉驗證建立信賴區間；交叉驗證是輪替不同訓練與評估資料的做法，信賴區間則用來表達估計的不確定程度。

## 5. NumPy 訓練與 CUDA-Q 驗證

NumPy 是 Python 數值運算套件。全部搜尋使用 CPU（中央處理器），包括量子模型的完整狀態向量模擬；狀態向量保存各量子基底狀態的振幅，其絕對值平方決定量測機率。`cudaq_training_calls=0` 表示訓練期間沒有呼叫 CUDA-Q。

CUDA-Q 是執行量子程式的工具。本章待權重固定後，才以 CPU 與 GPU（擅長平行運算的圖形處理器）上的模擬器，逐筆核對 VQC 與 Hybrid：

```text
2 splits × 119 samples × 2 quantum models = 476 observe / backend
```

後端（backend）是實際執行模擬的工具。每個後端有 476 次 `observe` 呼叫，也就是計算指定量測量的期望值。MLP 與 Logistic13 仍由 NumPy 在 CPU 計算。每個後端保存 2 組切分 × 4 個模型 × 3 種資料用途，共 24 筆驗證紀錄，包含輸出、評分、誤差與可能包含編譯的時間；編譯是將程式轉成後端可執行形式的處理。

`shots=-1` 表示直接從模擬狀態計算期望值，不以有限次量測抽樣估計。模擬沒有加入硬體噪聲，也不是在真實量子處理器（QPU）執行。驗證固定權重與搜尋權重是不同工作，不能將兩者耗時相除來宣稱 GPU 訓練加速。

此處維持既有小型電路。Day24 的梯度診斷不代表本章已解決貧瘠高原，也不是任意增加電路層數的依據。Day23 以樣本相似度建模的量子核方法，未納入這次比較。

## 6. 重跑與單筆示範

使用 [requirements-day25.txt](../../requirements-day25.txt)，資料已保存，可離線重訓。`OMP_NUM_THREADS=1` 將 CPU 平行工作的執行緒數設為 1。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day25/experiment.py train
OMP_NUM_THREADS=1 python articles/day25/experiment.py verify
OMP_NUM_THREADS=1 python articles/day25/experiment.py verify --backend nvidia
OMP_NUM_THREADS=1 python articles/day25/plot_results.py

# 預設使用保存的切分 2030 中第一筆測試輸入；預測時不傳入正確標籤。
OMP_NUM_THREADS=1 python articles/day25/demo.py

OMP_NUM_THREADS=1 python -m unittest discover -s articles/day25 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY25_TARGET=nvidia python -m unittest discover -s articles/day25 -p 'test_*.py' -v
```

示範支援 `--split-seed 2031`、`--backend nvidia`，也可在 `--features` 後依序提供十三個原始數值：

| 順序 | 原始欄位 | 白話說明 |
|---|---|---|
| 1 | alcohol | 酒精含量 |
| 2 | malic acid | 蘋果酸含量 |
| 3 | ash | 灰分，樣本燃燒後留下的無機殘留物 |
| 4 | alcalinity of ash | 灰分的鹼度測量 |
| 5 | magnesium | 鎂含量 |
| 6 | total phenols | 總酚類含量，酚類是一類化學成分 |
| 7 | flavanoids | 類黃酮含量，屬於酚類成分 |
| 8 | nonflavanoid phenols | 非類黃酮的酚類含量 |
| 9 | proanthocyanins | 原花青素含量，另一類酚類成分 |
| 10 | color intensity | 顏色強度 |
| 11 | hue | 色調測量值 |
| 12 | OD280/OD315 | 稀釋葡萄酒在 280 與 315 奈米波長的光學密度比值；光學密度描述光通過樣本後的衰減 |
| 13 | proline | 脯胺酸含量，脯胺酸是一種胺基酸 |

數值與單位沿用原始資料慣例，不使用 Iris 長度特徵的公分單位。

五項測試涵蓋來源與切分、僅使用訓練資料建立 PCA、各模型與參考計算的對照、完整十三維模型與共用 PCA 介面，以及錯誤輸入。預設 CPU 可獨立執行；完整報告需要兩個後端的驗證摘要。

輸出位於 `results/day25/`，重跑會更新同名檔案；重訓後需重新執行 `verify` 與繪圖程式。檔案的 SHA-256 將實驗設定、資料與選定模型連結起來，避免混用過期的後端結果。已保存的 GPU 執行紀錄在結束時仍有 `cudaErrorCudartUnloading`，但驗證與測試的退出碼為 0，表示程式回報成功；該結束錯誤的根因尚未定位。

## 7. 從資料表示到噪聲實驗

Day21–25 依序比較資料壓縮與特徵選擇、重複編碼、量子核方法、梯度診斷與第二個真實資料集。這些實驗分別觀察輸入表示、模型與訓練條件如何影響結果；各章的資料與預算不同，需要連同設定一起判讀。

Day26 將加入 Quantum Noise（量子噪聲），也就是使量子操作或量測偏離理想結果的干擾，檢查理想模擬與含噪聲實驗的差距。

來源：[D21] S. Aeberhard and M. Forina (1992). *Wine* [Dataset]. UCI Machine Learning Repository. [UCI 頁面](https://archive.ics.uci.edu/dataset/109/wine)，DOI [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)，CC BY 4.0。查閱／下載日期 2026-09-07；完整 [參考索引](../../REFERENCES.md) 與 [資料處理紀錄](../../data/day25/README.md)。

接續：[Day26｜Quantum Noise](../day26/README.md)。

## 延伸研究

[N7] Jan Schnabel and Marco Roth. “Quantum kernel methods under scrutiny: a benchmarking study.” Quantum Machine Intelligence 7, 58 (2025)；研究論文。[原始來源](https://doi.org/10.1007/s42484-025-00273-5)；[完整書目](../../REFERENCES.md#n7)。

本章將比較延伸到 Wine；這篇研究有助於理解跨資料集評估的必要性。其量子核結果不等於本章分類模型的結果。
