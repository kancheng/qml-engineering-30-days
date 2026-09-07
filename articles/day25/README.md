# Day 25｜第二個真實資料集：Wine Benchmark

今天把前面建立的Classical、VQC、Hybrid流程移到另一個資料集。**使用UCI Wine的cultivar2／3二分類、13個原始特徵；不是完整三分類，也不是Wine Quality的品質預測。**

交付：[wine_models.py](wine_models.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[圖表／報告產生器](plot_results.py)、[tests](test_wine.py)、[第二份Benchmark Report](../../results/day25/README.md)。沿用獨立`.venv`，不新增套件。

## 1. 為何需要第二份Benchmark？

Day20的Iris結果只對特定資料、表示與預算成立。Wine讓同一流程面對13個化學特徵及不同類別比例，檢查資料前處理和模型介面能否重用。這仍是小型教學資料，不代表大型工業部署或量子優勢。

UCI Wine包含178筆、13個化學測量、三個cultivars。[D21] 本日預先固定class2→label0、class3→label1，沿用已驗證的binary模型，不依模型表現挑class pair。保留71＋48=119筆；loader檢查完全相同features＋label，本子集沒有duplicate。

原始檔未修改，保存於[data/day25](../../data/day25/README.md)，附UCI來源、CC BY4.0授權、attribution與下載日期。`protocol.json`記錄SHA-256與feature order。

## 2. 切分、Labels與Train-only轉換

預先指定dataset seeds2030／2031，各類別分別打亂，取floor60% train、floor20% validation，其餘test：

| Split | Cultivar2 / label0 | Cultivar3 / label1 | Total |
|---|---:|---:|---:|
| train | 42 | 28 | 70 |
| validation | 14 | 9 | 23 |
| test | 15 | 11 | 26 |

保存原始zero-based row IDs與全部raw features。同一split內無共享紀錄；不同dataset seeds的test會重疊，不視為兩個獨立抽樣實驗。

三個模型共用表示：

```text
raw13 → train mean/std standardization → train SVD PCA2
      → train PC minmax → clip [-1,1] → MLP / VQC / Hybrid
```

PCA component的最大絕對值係數固定為正，以處理符號不定性。所有mean、population std、components與PC minmax只在train fit；validation／test只transform。PCA不使用labels，但依然是資料驅動轉換，不能全資料fit。做法沿用[Day20](../day20/README.md)與[Day21](../day21/README.md)。

完整13維classical baseline只用train standardization，不做PCA或clipping。它能檢查壓縮後模型與完整輸入的差距，但資訊量不同，不是隔離量子層貢獻的公平配對。

## 3. 模型與固定搜尋預算

| Model | 結構 | 可訓練參數 | Qubits |
|---|---|---:|---:|
| MLP | PCA2→tanh2→sigmoid1 | 9 | 0 |
| VQC | PCA2→positive_half encoding→單層Ansatz→(1−ZZ)/2 | 4 | 2 |
| Hybrid | PCA2→affine+tanh encoder→PQC→affine sigmoid head | 12 | 2 |
| Logistic13 | Standardized13→sigmoid | 14 | 0 |

MLP／VQC／Hybrid直接重用Day20模型；Hybrid在這裡接收PCA2，沒有改成13→2的learned bottleneck。這個選擇保持跨資料集模型結構一致。Logistic13是以共同Brier訓練的logistic-link，不是標準cross-entropy LogisticRegression solver。

兩splits×四模型×初始化seeds42／43=**16次fit**。共同normal(0,0.2)初始化，hybrid head scale起始為0.8；Day18座標搜尋step=0.4，每次73個objective，每個objective看全部70筆train，即**5,110筆模型輸入評估／fit**。

每模型每split以final validation Brier選seed，平手取較小seed；test不參與。完整保存初始／最終weights、搜尋history、validation與所有候選，沒有依test成績增加預算。

73個objective只代表相同呼叫預算。參數數量不同，完成full sweeps數也不同；單個objective的成本亦不同，不能稱作相同wall time或充分收斂。PCA fit成本不包含於`fit_seconds_numpy`，本日不做完整pipeline latency比較。

## 4. 實測與可重現結果

![Wine comparison](../../results/day25/comparison.png)

左圖呈現validation選定模型的test Brier；右圖保留split2030的兩種初始化曲線。完整accuracy、confusion、PCA variance、clipping、candidate時間見[結果紀錄](../../results/day25/README.md)。

Train-prior baseline使用p(label1)=28/70=0.4；因類別不平衡，不以固定0.5作唯一對照。Confusion matrix row=true、column=predicted；accuracy threshold固定0.5。PCA variance ratio是標準化train的變異比例，不是保留分類訊號的比例。

每個test只有26筆，一筆分類差異就是約3.85個百分點。不要把這種差距外推成model family優勢；也不把Iris與Wine不同任務的accuracy直接排行。本日是預先固定小預算的探索性benchmark，沒有cross-validation信賴區間或量子優勢主張。

## 5. NumPy訓練與CUDA-Q驗證分開

全部搜尋使用NumPy CPU，包括量子模型的exact statevector reference，`cudaq_training_calls=0`。CPU／GPU在weights凍結後逐筆核對VQC與Hybrid：

```text
2 splits × 119 samples × 2 quantum models = 476 observe / backend
```

MLP與Logistic13仍為NumPy CPU。每backend有2splits×4models×3partitions=24筆驗證records，保存probabilities、metrics、誤差與可能包含compilation的時間。shots=-1、無noise，非QPU；不能將驗證時間除以reference訓練時間宣稱GPU speedup。

這裡有意維持既有小型電路，不因Day24的gradient趨勢任意加深Ansatz或宣稱本日已解決Barren Plateau。Day23的kernel benchmark也不是本日的重跑範圍。

## 6. 重跑與單筆示範

使用[requirements-day25.txt](../../requirements-day25.txt)，資料已保存，可離線重訓：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day25/experiment.py train
OMP_NUM_THREADS=1 python articles/day25/experiment.py verify
OMP_NUM_THREADS=1 python articles/day25/experiment.py verify --backend nvidia
OMP_NUM_THREADS=1 python articles/day25/plot_results.py

# 預設使用保存的split2030第一筆test輸入；label不傳入inference。
OMP_NUM_THREADS=1 python articles/day25/demo.py

OMP_NUM_THREADS=1 python -m unittest discover -s articles/day25 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY25_TARGET=nvidia python -m unittest discover -s articles/day25 -p 'test_*.py' -v
```

Demo支援`--split-seed 2031`、`--backend nvidia`，也可用`--features`後接13個原始數值，順序為：alcohol、malic acid、ash、alcalinity of ash、magnesium、total phenols、flavanoids、nonflavanoid phenols、proanthocyanins、color intensity、hue、OD280/OD315、proline。單位依原始資料慣例，不沿用Iris的cm。

五項測試涵蓋來源／切分、train-only PCA、所有模型reference對照、完整13維baseline與共用PCA介面，以及錯誤輸入。預設CPU可獨立run；完整報告需兩backend summaries。

輸出位於`results/day25/`，重跑更新同名檔案；重訓後重新verify與plot。artifact SHA-256綁定protocol／datasets／selected，避免引用過期backend summary。GPU退出仍出現`cudaErrorCudartUnloading`，但驗證與測試exit code=0，根因未定位。

## 7. Day21–25交付與下一步

這一階段已完成降維／feature selection／bottleneck、re-uploading、quantum kernel、gradient trainability診斷與第二份真實資料集報告。各篇保存自己的protocol與限制，不把不同資料／模型／預算混成單一排行榜。

Day26將進入Quantum Noise，檢查exact simulator結果與含噪聲實驗的差距。

來源：[D21] S. Aeberhard and M. Forina (1992). *Wine* [Dataset]. UCI Machine Learning Repository. [UCI頁面](https://archive.ics.uci.edu/dataset/109/wine)，DOI[10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)，CC BY4.0。查閱／下載日期2026-09-07；完整[參考索引](../../REFERENCES.md)與[資料處理紀錄](../../data/day25/README.md)。

接續：[Day26｜Quantum Noise](../day26/README.md)。
