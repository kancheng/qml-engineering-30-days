# Day 22｜Data Re-uploading：讓資料再次進入量子電路

[Day21](../day21/README.md) 比較了四維到二維的壓縮。今天改變電路的資料輸入位置：**在同一個 quantum state 上交替執行資料編碼與 trainable blocks，讓資料被多次使用。**

程式：[reuploading.py](reuploading.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[plot_results.py](plot_results.py)、[tests](test_reuploading.py)。完整數值見 [實驗報告](../../results/day22/README.md)。沿用 `.venv`，無新增套件或資料下載。

## 1. 多一層 Ansatz，不一定是 Re-uploading

令 E(x) 為 data encoding，A(θ) 為 trainable block。以下箭頭依實際執行順序：

```text
single upload：|00> → E(x) → A(θ0) → A(θ1) → observe ZZ
re-uploading： |00> → E(x) → A(θ0) → E(x) → A(θ1) → observe ZZ
```

兩者都有兩個 trainable blocks，第二者額外使用一次 E(x)。中途不量測、不 reset；不是重複抽樣，也不是把同一筆資料當成兩筆 train samples。

Pérez-Salinas 等人的原始研究以重複資料編碼與可訓練操作建立 classifier，討論 qubit 數與層數的取捨。[P4] **本日是 RY／CNOT／ZZ 的受限教學模型，不是該論文完整架構或 universal classifier 定理的復現。**

## 2. 四維資料可以分段進入兩個 Qubit

不用先做 PCA，也可以分兩段輸入：

```text
E01 = RY(a0) on q0, RY(a1) on q1
E23 = RY(a2) on q0, RY(a3) on q1

一次完整資料 pass：E01 → A0 → E23 → A1
再次完整資料 pass：E01 → A2 → E23 → A3
```

第一次 E23 使用後兩個特徵，是分段上傳；第二次 E01／E23 才是把各特徵再次上傳。不能把「四維拆成兩段」本身就當作「每個特徵使用兩次」。特徵順序固定為 sepal length、sepal width、petal length、petal width，沒有依 test 結果挑順序。

這讓兩個 qubit 接收四個原始特徵，但不是無損記憶體，也不能從一次量測讀回四個值。資料仍受 angle mapping、電路與 readout 限制。

## 3. 配對控制：相同參數與 Ansatz，只改 Schedule

`SCHEDULES` 的每個元素對應一個 trainable block 前的資料操作：0=輸入欄位 0／1；2=輸入欄位 2／3；−1=不輸入。

| Model | 表示 | Schedule | Trainable parameters | Upload blocks | CNOT |
|---|---|---|---:|---:|---:|
| pca_once2 | Day20 PCA2 | [0, −1] | 8 | 1 | 2 |
| pca_repeat2 | 同一份 PCA2 | [0, 0] | 8 | 2 | 2 |
| chunks_once4 | 四維 raw minmax | [0, 2, −1, −1] | 16 | 2 | 4 |
| chunks_repeat4 | 同一份四維表示 | [0, 2, 0, 2] | 16 | 4 | 4 |
| logistic4 | 四維 standardization | 不適用 | 5 | 0 | 0 |

每個 A block 固定為 RY⊗RY → CNOT(q0→q1) → RY⊗RY，四個獨立參數；所有量子模型用兩 qubit、`p1=(1−⟨ZZ⟩)/2`。配對內相同 seed 產生完全相同 initial weights。配對間的輸入表示、參數量與層數不同，不能把所有差異都歸因於 re-uploading。

未合併 gate 的邏輯資源：若有 L 個 A blocks、U 個 upload blocks，則 RY 數=4L+2U、CNOT 數=L。允許不同 qubit 上的單 qubit gates 平行時，排程深度=3L+U，不含 readout。因此四個量子模型的深度分別為 7、8、14、16。這是原始排程的計數，**不是最佳化後電路深度、QPU native gates 或實測延遲**。

## 4. 重複並不自動增加可用表達能力

相鄰同軸旋轉可以合併：

```text
RY(b) RY(a) = RY(a+b)
```

如果只是連續堆相同資料旋轉，中間沒有適當的其他操作，可能只得到角度相加。程式有矩陣測試驗證這個等式。本日 upload 前後部分 RY 也能合併，故資源表刻意標明「未合併」。CNOT 與其他操作之間的次序則會影響完整電路。

我們另固定 seed42 的八個 weights，比較 once2 與 repeat2 的二維機率反應：

![Fixed-weight schedule probe](../../results/day22/schedule_probe.png)

這是合成 scaled inputs 的 reference 計算，沒有 train labels、沒有挑權重，也不是 test decision boundary。結果保存所有 grid probabilities，以及四點 mixed contrast：

```text
Δ = p(+0.5,+0.5) − p(+0.5,−0.5) − p(−0.5,+0.5) + p(−0.5,−0.5)
```

非零 Δ 表示這四點的反應無法寫成 f(x0)+g(x1)，是局部非加性例子；不是通用 expressibility 指標。單次 upload 本來就可能有 interaction，不能把所有 interaction 都歸功於重複輸入。圖形改變也不能證明函數族包含關係或 universal approximation。

## 5. 沿用 Iris 與 Train-only 邊界

使用 Day20 保存的 binary Iris：去重後 99 筆，dataset seeds 2028／2029；各 split 為 train59／validation19／test21。原始資料 SHA-256、row IDs、mean／std、PCA components 與 minmax 都保存。

- PCA 配對直接重用 Day20 的 train-only PCA2 與 PC minmax，clip 至 [-1,1]。
- Chunks 配對以各 raw 欄位的 train min／max 縮放至 [-1,1]；不做 PCA、不用 labels。
- Logistic baseline 以 train mean／std 標準化全部四維，與 Day21 相同。

每個量子輸入角度為 `a=π(x+1)/2`，即 positive_half。Clipping 只按 transform 的座標數記錄，不因同一座標重複上傳而加倍。所有 validation／test 僅 transform，不重新 fit。[D14]

## 6. 訓練與實測

兩個 splits × 五個模型 × 初始化 seeds42／43，共 **20 次 fit**。與 Day21 相同，normal(0,0.2) 初始化，Brier loss、Day18 座標搜尋、step=0.4、73 次 objective，每次用全部59筆 train，共4,307筆輸入評估。

每模型每 split 依 final validation Brier 挑 seed，平手取較小 seed；保留全部 candidates。test 不參與當次選 seed。相同 objective 次數不代表相同 gates／runtime：8／16參數模型只有4／2次 full sweeps，不聲稱充分收斂。Classical baseline 是 Brier logistic-link，不是標準 cross-entropy LogisticRegression solver。

![Benchmark comparison](../../results/day22/comparison.png)

這次結果沒有顯示重複輸入必然比較好。完整 Brier、accuracy、confusion matrix、clipping、曲線與各 fit 時間都在 [結果報告](../../results/day22/README.md)。不依 test 成績更換 seed、提高特定模型預算或刪除較差的候選。

本日所有訓練在 NumPy CPU exact statevector reference 執行，`cudaq_training_calls=0`。選定參數後，CUDA-Q CPU／GPU 分別驗證四個量子模型：2×99×4=**792 次 observe／backend**，另核對 classical baseline，合計各30筆 partition records。shots=-1、無噪聲，非 QPU／GPU training benchmark；時間可能包含 JIT 與 cache，不相除作 speedup。

兩個 splits 互相重疊且 test 已在 Day20 公開，所以只是探索性系列實驗，不做顯著性／量子優勢推論。正式決策應另外使用未查看 holdout 或 nested cross-validation。

## 7. 執行範例與測試

依賴入口：[requirements-day22.txt](../../requirements-day22.txt)。資料與模型均已保存，demo 不需要重新訓練。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day22/experiment.py train
OMP_NUM_THREADS=1 python articles/day22/experiment.py verify
OMP_NUM_THREADS=1 python articles/day22/experiment.py verify --backend nvidia
OMP_NUM_THREADS=1 python articles/day22/plot_results.py

# 原始四個 cm 數值，輸出五個模型的表示與 p(virginica)
OMP_NUM_THREADS=1 python articles/day22/demo.py --features 6.0 2.9 4.5 1.5

OMP_NUM_THREADS=1 python -m unittest discover -s articles/day22 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY22_TARGET=nvidia python -m unittest discover -s articles/day22 -p 'test_*.py' -v
```

Demo 可加 `--split-seed 2029`、`--backend nvidia`。CPU 可獨立執行 train／verify／demo／tests；完整報告產生器需要兩 backend summaries。輸出位於 `results/day22/`，重跑覆寫同名檔案。重訓後須重跑 verify、plot；artifact SHA-256 防止報告混用舊的 backend summaries。

六項測試涵蓋 train-only／clipping、不重複輸入與 Day14 的相容性、兩 backend 對照、四個特徵都影響輸出、相鄰旋轉合併，以及錯誤輸入。GPU 驗證與測試 exit code=0，但結束時有 `cudaErrorCudartUnloading`，根因未定位。

## 8. 下一步與來源

Day23 將實作 Quantum Kernel，從「訓練電路權重」轉向「用量子特徵空間計算樣本相似度」。

- [P4] Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, José I. Latorre. “Data re-uploading for a universal quantum classifier.” *Quantum* 4, 226 (2020). [出版頁與 DOI](https://quantum-journal.org/papers/q-2020-02-06-226/)。原始概念來源，不宣稱本日是論文完整復現。
- [D14] [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：train-only preprocessing。
- [D17] 資料來源與授權：[Day20 UCI 紀錄](../../data/day20/README.md)。

出版頁查閱日期 2026-09-07；頁面標示發表版本對應 arXiv v2，另列較新的 v3，本日引用期刊版本。完整書目見 [REFERENCES.md](../../REFERENCES.md)。

接續：[Day23｜Quantum Kernel](../day23/README.md)。
