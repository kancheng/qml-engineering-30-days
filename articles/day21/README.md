# Day 21｜Qubit 不夠、Feature 太多怎麼辦？

## 本章摘要｜初學者學習筆記

### 中文

[Day20](../day20/README.md) 將 Iris 的四個原始特徵經 PCA 轉成兩個主成分，再比較經典、量子與混合模型的二分類結果。Day21 接著檢查這個降維選擇，固定兩個量子位元與量子讀出，觀察不同的四維到二維表示如何影響分類與資訊保留。

這一章目標在於理解 **「當輸入特徵比目前電路接受的角度數更多時，如何選擇資料表示，並辨認壓縮的代價」**。本章比較三種方法：PCA（主成分分析）將四個特徵線性組合成兩個高變異方向；feature selection（特徵選擇）依訓練資料與標籤的關係保留兩個原始欄位；learned bottleneck（可訓練的壓縮層）則將四維輸入轉成兩個數值，並與量子電路權重一起依分類 loss 更新。重點是三種方法保留資訊的依據不同：高變異不保證有助分類，單欄排序可能漏掉特徵間的聯合作用，可訓練壓縮則增加參數與搜尋成本。本章另加入使用完整四維輸入的經典模型作對照，避免把所有比較都限制在同一份壓縮資料上。實作沿用 Day20 的資料切分，全部訓練先在 NumPy CPU 完成，再以 CUDA-Q CPU／GPU 核對保存模型的輸出。讀完本章，應能說明特徵數與量子位元數並非固定的一對一關係，理解降維造成的資訊損失不會因接上量子電路而自動恢復，並分清楚資料表示、模型容量與訓練預算對比較結果的影響。

### English

[Day20](../day20/README.md) reduced Iris's four original features to two principal components and compared classical, quantum, and hybrid models on a binary classification task. Day21 examines that reduction choice, keeping two qubits and the quantum readout fixed while exploring how different four-to-two-dimensional representations affect classification and information retention.

This chapter aims to explain **how to choose a data representation when the input has more features than the current circuit accepts as encoding angles, and how to recognize the costs of compression**. Three approaches are compared. Principal component analysis (PCA) combines the four features into two high-variance directions. Feature selection retains two original columns based on their relationship with training labels. A learned bottleneck maps four inputs to two values and updates its parameters jointly with the quantum circuit using classification loss. Each approach prioritizes information differently: high variance does not guarantee useful class information, single-feature ranking can miss joint effects, and a trainable bottleneck adds parameters and search cost. A classical model using all four features provides an additional comparison beyond compressed inputs. The implementation reuses Day20's data splits, trains entirely on the NumPy CPU, and then verifies saved model outputs with CUDA-Q CPU/GPU execution. The learning goal is to explain why feature count and qubit count need not have a fixed one-to-one relationship, recognize that a downstream quantum circuit cannot automatically recover discarded information, and distinguish the effects of representation, model capacity, and training budget on the results.

---

[Day20](../day20/README.md) 把四維 Iris 壓到兩個 PCs，再送入兩個 qubit。今天把這個決定拆開實驗：**固定兩個 qubit，改變四維到二維的表示方法，觀察分類、資訊損失與訓練成本。**

交付：[表示與模型](reduction.py)、[訓練／驗證](experiment.py)、[單筆推論](demo.py)、[圖表／報告產生器](plot_results.py)、[測試](test_reduction.py)、[實測結果](../../results/day21/README.md)。沿用獨立 `.venv` 與本機 Iris 資料，不增加套件。

## 1. Feature 數不等於 Qubit 數

本系列目前的 angle feature map 每個 qubit 使用一個資料角度，所以一次準備需要兩個數值。這是此電路的介面限制，不是「兩個 qubit 永遠只能使用兩個特徵」的定律。

| 方案 | 如何處理四維輸入 | 代價／限制 |
|---|---|---|
| PCA2 | 線性組合成兩個主成分 | 保留高變異方向，不保證保留分類訊號 |
| Feature selection | 保留兩個原始欄位 | 易解釋，但丟棄其他欄位及可能的交互作用 |
| Learned bottleneck | 可訓練的 4→2 encoder | 增加參數與搜尋難度，需連同下游 loss 訓練 |
| 增加 qubit | 可擴充一次 angle encoding 的寬度 | statevector simulator 的振幅數是 2^n，增加記憶體與運算需求 |
| Amplitude encoding | 四個數值正規化成兩 qubit 的振幅 | 丟失整體尺度，需要 state preparation，不能直接讀回所有數值 |
| 分段輸入／re-uploading | 在多個編碼區段使用資料 | 增加電路操作；下一日再實作比較 |

今天實測前三種與 classical baseline；amplitude encoding 的正規化與 gate 準備已在 [Day13](../day13/README.md) 實作。四維到二維的投影不是無損壓縮，也不會因下游接量子電路就恢復被丟掉的資訊。

## 2. 延用相同 Iris 切分

使用 [Day20 原始資料](../../data/day20/README.md)：versicolor=0、virginica=1，去除一筆完全重複 features＋label 後 99 筆。dataset seeds 2028／2029，各為 train 59、validation 19、test 21；row IDs 與來源 SHA-256 保存於 JSON。這不是完整三分類 Iris。

所有方法只用 train 的 mean／population std 標準化：

```text
z = (raw - train_mean) / train_std
```

PCA／feature ranking／minmax scaler 都在 train fit。validation 只挑初始化 seed；test 只在 seed 選定後評分。即使 PCA 不需要 labels，也不能先在全資料 fit。[D14]

因為沿用 Day20 已公開的 test，本日是探索性系列實驗，不視為新的盲測。若要依多日實驗挑最終方案，需另留未查看的 holdout 或用 nested cross-validation。

## 3. 三種表示，一個相同的量子 readout

### PCA2：優先保留 train variance

直接重用 Day20 的 SVD、component 符號規則與 train PC minmax scaling。`pca_vqc` 是連續性檢查，應重現 Day20 VQC 的初始化、訓練與預測。

PCA 尋找高變異的線性子空間。[D18] explained variance ratio 是標準化 train 資料的變異占比，不能解釋成「保留多少分類能力」。程式的 collision 測試沿 discarded direction 改變四維輸入，確認兩個不同輸入可以產生一樣的 PCs；這類差異無法再由下游模型分辨。

### Feature selection：只用 Train Labels 排序

本日自行以 NumPy 計算每個欄位與 binary label 的絕對 Pearson correlation（point-biserial correlation）：

```python
z = (x_train - train_mean) / train_std
score = abs(np.mean(z * ((y_train - y_train.mean()) / y_train.std())[:, None], axis=0))
indices = sorted(range(4), key=lambda j: (-score[j], j))[:2]
```

固定取前兩欄，平手取較小 column index；對選中欄位 fit train minmax，再 clip 至 [-1,1]。不看 test 挑欄位，也沒有嘗試全部欄位組合後挑最高 test score。

這是單變量監督式 ranking 示範，參照 feature selection 的流程概念。[D19] 並未呼叫 scikit-learn 的 `SelectKBest`，也不做 p-value 或顯著性推論。相關欄位可能互相冗餘，單欄弱但聯合作用強的特徵也可能被漏掉。

### Learned bottleneck：以分類 Loss 聯合更新

```text
raw4 → train standardize → tanh(z @ W + b) → angle encoding → Ansatz → ZZ → p1
                            W:4×2, b:2                         p1=(1-ZZ)/2
```

encoder 有 8+2=10 個參數，量子 Ansatz 有 4 個，合計 14。此處不接 Day19 的 sigmoid head，讓三個量子方案共用相同的 `p1=(1−ZZ)/2` readout。bottleneck 直接接收標準化四維資料，沒有先做 PCA。

`tanh` 將兩個輸出限制在 [-1,1]，不需另 fit latent minmax。過大的絕對值會趨近飽和；結果保存 `abs(h)>0.99` 的座標數，這只是描述性門檻，不是梯度消失檢定。encoder 與 Ansatz 都由 train Brier 的座標搜尋更新，並非預訓練 autoencoder，也沒有 reconstruction loss。

三個量子模型皆重用 Day14：兩 qubit、positive_half angle encoding、單層四參數 Ansatz、exact ZZ。電路寬度與 Ansatz 相同，前處理的監督資訊及參數容量並不相同。

## 4. Classical Baseline 與預算

| Model | Classical 輸入／表示 | 可訓練參數 | Qubits |
|---|---|---:|---:|
| pca_vqc | PCA2＋train minmax | 4 | 2 |
| selection_vqc | train ranking top2＋minmax | 4 | 2 |
| bottleneck_vqc | 四維標準化→tanh2 | 14 | 2 |
| logistic4 | 完整四維標準化→sigmoid | 5 | 0 |

Classical baseline 使用完整四維輸入，補上 Day20 所有模型都只能看到 PCA2 的限制。它是以 Brier loss 訓練的 logistic-link，並非標準 cross-entropy LogisticRegression solver。

共同 protocol：兩個 splits × 四模型 × 初始化 seeds 42／43 = **16 次 fit**。每次 normal(0,0.2) 初始化，Day18 座標搜尋、step=0.4、73 次 objective；每次 objective 用全部 59 筆 train，因此各 4,307 筆輸入評估。每模型每 split 依 final validation Brier 選 seed，平手取較小 seed。

73 次 objective 意味 36 次座標嘗試，每次比較 ±step；4、14、5 參數模型完成的 full sweeps 不同。參數較多不會自動獲得額外預算，也不能把這個小預算結果當作充分收斂後的模型排名。PCA／ranking 的 fit 成本不包含於 `fit_seconds_numpy`；本日不做完整 pipeline latency benchmark。

## 5. 實測與圖表

![Model comparison](../../results/day21/comparison.png)

![Representations](../../results/day21/representations.png)

第一張圖保存兩個 split 的 test Brier 與兩個初始化的 train curves；第二張圖顯示 split2028 的凍結表示，test 點只 transform，不重新 fit。完整 metrics、confusion matrices、特徵欄位、clipping、saturation、全部候選時間見 [實驗報告](../../results/day21/README.md)。

不要只比較 accuracy：同樣的分類決策可能有不同的機率與 Brier。也不要因 bottleneck 較好就推論量子優勢；它增加了 supervised encoder，與 PCA 的資訊目標不同。若要隔離量子層貢獻，還需要容量／預算相近的 classical bottleneck 對照，本日未做這個消融。

## 6. NumPy 訓練，CUDA-Q 驗證

所有搜尋在 NumPy CPU exact statevector reference 上完成，`cudaq_training_calls=0`。CUDA-Q 在選定 weights 後逐筆核對三個量子模型，CPU／GPU 每個 backend 共 2×99×3=**594 次 observe**，另核對 classical baseline；合計 24 筆 partition records。沒有 finite shots、noise 或 QPU。

保存訓練與驗證的不同時間欄位；驗證可能包含 compilation／cache，不能與訓練時間相除宣稱 GPU speedup。`artifact_sha256` 綁定 protocol、datasets、selected；報告產生器會拒絕引用與當前模型不符的 backend summaries。

## 7. 重跑與示範

使用現有 `.venv`；[requirements-day21.txt](../../requirements-day21.txt) 延續 Day20 依賴。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day21/experiment.py train
OMP_NUM_THREADS=1 python articles/day21/experiment.py verify
OMP_NUM_THREADS=1 python articles/day21/experiment.py verify --backend nvidia
OMP_NUM_THREADS=1 python articles/day21/plot_results.py

# 四個 cm 數值：sepal length、sepal width、petal length、petal width
OMP_NUM_THREADS=1 python articles/day21/demo.py --features 6.0 2.9 4.5 1.5

OMP_NUM_THREADS=1 python -m unittest discover -s articles/day21 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY21_TARGET=nvidia python -m unittest discover -s articles/day21 -p 'test_*.py' -v
```

Demo 可加 `--backend nvidia` 或 `--split-seed 2029`；載入保存的 preprocessing 與 weights，印出表示、clipping、p(virginica) 及類別。沒有 GPU 時可執行預設 CPU train／verify／demo／tests；完整雙 backend 報告需要兩邊驗證。

輸出在 `results/day21/`，重跑會更新同名檔案。更改訓練後重新執行 verify 與 plot。五項測試涵蓋 train-only 邊界與資料切分、ranking／Day20 PCA 相容、丟失方向的碰撞、CUDA-Q／reference 與 encoder／quantum 參數作用，以及錯誤輸入。

本機 GPU 驗證結束時仍出現 `cudaErrorCudartUnloading`，但程序 exit code=0 且所有數值檢查通過；退出訊息的根因未在本日定位。

## 8. 下一步與來源

Day22 將實作 Data Re-uploading：在保持小 qubit 數時，讓資料進入多個電路區段，繼續檢查表示能力與成本。

- [D14] [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：preprocessing／feature selection 的資料洩漏邊界。
- [D18] [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)：線性子空間與 explained variance。
- [D19] [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)：單變量選擇流程；本文使用自訂 binary correlation score。
- [D17] 資料來源與授權沿用 [Day20 UCI 紀錄](../../data/day20/README.md)。

官方文件查閱日 2026-09-07；本日以 NumPy 實作，沒有安裝 scikit-learn。完整索引見 [REFERENCES.md](../../REFERENCES.md)。

接續：[Day22｜Data Re-uploading](../day22/README.md)。
