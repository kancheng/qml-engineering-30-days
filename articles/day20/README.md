# Day 20｜Iris：Classical vs Quantum vs Hybrid

今天把 Day 18 的比較流程與 Day 19 的 hybrid model 放到 Iris。**本日是 versicolor／virginica 二分類、四個原始特徵經 train-only PCA2 的受限 benchmark，不是完整三分類 Iris 排行榜。**

程式：[iris_models.py](iris_models.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[plot_results.py](plot_results.py)。所有訓練先在 NumPy CPU 的精確 reference 完成，再用 CUDA-Q CPU／GPU 核對選定模型；這兩個階段的時間與作用分開保存。

## 1. 從真實資料開始

UCI Iris 有 150 筆、四個數值特徵、三類各 50 筆。[D17] 本日保留 versicolor=0、virginica=1，排除 setosa，以沿用已驗證的 binary models。

原始檔案保存在 [data/day20](../../data/day20/README.md)，包含 UCI 來源、CC BY 4.0 授權與下載日期。程式記錄原始檔案 SHA-256，資料可離線重跑。

二分類子集原本 100 筆，其中一筆 features＋label 完全重複；切分前保留第一筆、移除原始 zero-based row ID 142，剩 99 筆。這避免完全相同的紀錄跨 split；不代表有足夠資訊確認所有植物樣本的個體獨立性。原始下載檔案不修改。

四個欄位順序固定：sepal length、sepal width、petal length、petal width，單位 cm。不同版本的 Iris 數字可能有差異，因此本日以保存的 UCI 檔案和雜湊為準。

## 2. 兩組 Stratified Splits

預先指定 dataset seeds 2028／2029，每一類各自打亂，前 60% train、接下來 20% validation、其餘 test，取整數後得到：

| Split | 總數 | versicolor | virginica |
|---|---:|---:|---:|
| train | 59 | 30 | 29 |
| validation | 19 | 10 | 9 |
| test | 21 | 10 | 11 |

每個 split 保存原始 row IDs、features、labels。不同 dataset seeds 的資料會重疊，所以兩個 test 成績不是獨立實驗樣本，不做顯著性推論。

## 3. 四維到二維，全部只在 Train Fit

```text
raw 4D → standardize → PCA2 → PC min/max scaling → clip to [-1,1]
```

1. 用 train 計算每欄 mean／population std。
2. 對 train standardized matrix 做 NumPy SVD，取前兩個 components。
3. 固定 component 符號：每個 component 最大絕對值係數為正，減少符號不定性。
4. 以 train 的兩個 PC 範圍 fit min/max scaler。
5. validation／test 僅使用這些保存的統計量 transform，保留 clipping 數量。

PCA 是資料驅動轉換，即使不使用 labels 也不能先對全資料 fit。[D14] 所有模型收到同一個二維表示；classical MLP 沒有額外取得四維原始資料。因此本日不代表 classical model 在全部四個特徵上的最佳表現。

`datasets.json` 保存 mean、std、components、explained variance ratio 與 PC scaler，demo 載入同一組 preprocessing，不會依新輸入重新 fit。

## 4. 三個模型與共同預算

| Model | 架構 | 參數 |
|---|---|---:|
| MLP | Day 16：2→tanh(2)→sigmoid(1) | 9 |
| VQC | Day 14：positive_half、兩 qubit、一層 Ansatz、p1=(1−ZZ)/2 | 4 |
| Hybrid | Day 19：affine+tanh encoder→PQC→affine sigmoid head | 12 |

每種模型每個 split 各 seeds 42／43，共 2×3×2=12 次 fit。使用 Day 18 同一份座標搜尋、初始 step=0.4、每次 73 次 objective、共同 Brier loss。參數初始化 normal(0,0.2)，hybrid head scale 依 Day 19 固定起始為 0.8。

每次 fit 看 59 筆 train，共 73×59=4,307 筆模型輸入評估。相同 objective 預算不代表相同計算量、參數更新頻率或最佳 optimizer。Hybrid 不在本日使用 Day 19 的 gradient descent，VQC 也不使用 parameter-shift，避免把不同訓練成本混在同一個預算名稱下。

每模型依 final validation Brier 選 seed，平手取較小 seed；test 不參與。所有 candidates 保留，不只公布 winner。

## 5. 訓練引擎與 CUDA-Q 驗證必須分清楚

本日完整搜尋使用 NumPy CPU：

- MLP 用 dense matrix operations。
- VQC 用 Day 14 的明確 statevector matrix reference。
- Hybrid 用 Day 19 的 classical layers 與 NumPy quantum reference。

這些 reference 表示與既有電路相同的兩個 qubit 無噪聲模型，使整個資料搜尋可快速重跑；**不是 CUDA-Q 訓練，也不是 GPU training acceleration**。`training_summary.json` 明確記錄 `cudaq_training_calls=0`。

模型選定並保存 weights 後，`verify` 才切換 CUDA-Q target，逐筆執行相同電路並比較 probability。每 backend：2 splits × 99 筆 × 2 個含量子層的模型=396 次 observe；MLP 仍在 CPU NumPy。總共 18 筆驗證紀錄，對應 2 splits × 3 models × 3 data partitions。

訓練的 `fit_seconds_numpy` 與驗證的 `verification_seconds_including_compilation` 不可相除當作 speedup。後者可能含 JIT／cache，CPU／GPU 驗證也可能同時執行，未做隔離效能測量。

## 6. 評分與穩定性

每模型保存 train／validation／test probabilities、Brier、accuracy、confusion matrix（row=true、column=predicted），另有 train-prior constant baseline。因 train 不是完全平衡，constant p1=29/59，不能沿用 Day 18 的固定 0.5。

![Iris comparison](../../results/day20/comparison.png)

左圖顯示兩個 dataset splits 的選定模型 test Brier；右圖保留 split2028 的兩個 initialization histories。完整 accuracy、選定 seed、PCA variance、時間與 CUDA-Q 誤差見 [結果紀錄](../../results/day20/README.md)。

穩定性報告只展示兩個 split 與各兩個 initialization，不稱作完整交叉驗證或可靠信賴區間。不同 seed 的最終模型、資料難度與 validation 選擇都可能影響 test；單次較好不能外推為模型家族優勢。

## 7. 重跑與單筆推論

沿用 `.venv`，本日無新增套件；資料已保存在 repository，[requirements-day20.txt](../../requirements-day20.txt) 延續依賴鏈。

```bash
source .venv/bin/activate

# 一次建立共同 protocol、preprocessing、12 次 NumPy 訓練及選定模型。
OMP_NUM_THREADS=1 python articles/day20/experiment.py train

# 驗證保存模型，不重新訓練或挑選參數。
OMP_NUM_THREADS=1 python articles/day20/experiment.py verify
OMP_NUM_THREADS=1 python articles/day20/experiment.py verify --backend nvidia
OMP_NUM_THREADS=1 python articles/day20/plot_results.py

# 輸入四個原始 cm 數值，預設載入 split2028 選定的模型。
OMP_NUM_THREADS=1 python articles/day20/demo.py --features 6.0 2.9 4.5 1.5
```

demo 可加 `--split-seed 2029`、`--backend nvidia`，會印出 scaled PCs、clipping flags 與三個模型的 p(virginica)。不需下載資料或重新 fit scaler。

結果位於 `results/day20/`：protocol、datasets、candidates、selected、training_summary、comparison.png。`qpp-cpu/`、`nvidia/` 子目錄分別保存 verification 與 summary。重跑使用固定路徑並更新檔案；重訓後應重新執行兩個 verify 和 plot，避免引用舊驗證。

## 8. 測試與第四階段交付

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day20 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY20_TARGET=nvidia python -m unittest discover -s articles/day20 -p 'test_*.py' -v
```

四個測試涵蓋來源／去重與 split 不交疊、train-only PCA／正交性／scaler 不被重 fit、三模型 CUDA-Q／NumPy 輸出及錯誤輸入。驗證通過不要求某模型擊敗另一個，也不以 test accuracy 作程式正確性的門檻。

Day 16–20 至此完成模型結構、梯度、受控 benchmark、hybrid joint updates 與第一份 Iris 比較報告。這仍是受限 binary task、兩維表示、小預算與 exact simulator 結果，沒有完整三分類、含噪聲訓練、QPU、GPU training 或量子優勢結論。

## 9. 來源

- [D17] [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris)，DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76)：資料、欄位、類別與授權資訊。
- [D14] [scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：train-only preprocessing，本文以 NumPy 實作 PCA，不安裝 scikit-learn。

查閱日期 2026-09-07；資料 attribution 與處理政策見 [data/day20/README.md](../../data/day20/README.md)，共用來源索引見 [REFERENCES.md](../../REFERENCES.md)。

下一日：[Day21｜Qubit 不夠、Feature 太多怎麼辦？](../day21/README.md)，比較 PCA、feature selection 與 learned bottleneck。
