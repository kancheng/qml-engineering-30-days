# Day 22｜資料重複編碼：讓資料再次進入量子電路

[Day21](../day21/README.md) 固定兩個量子位元，比較 PCA（主成分分析，將多個欄位組合成較少的新座標）、特徵選擇（保留部分原始欄位）與可訓練壓縮層（透過訓練學習如何合併欄位），檢查四維資料縮成二維時的資訊取捨。Day22 接著改變資料進入電路的位置，透過分段編碼與重複編碼，探索少量量子位元如何使用更多特徵或多次使用同一筆資料。

**增加可調電路層，與讓資料再次進入電路，是兩種不同的改動。** 前者增加處理狀態的操作，後者讓輸入再次決定電路中的旋轉角度。Data re-uploading（資料重複編碼）是在同一次狀態演化中，交替執行資料編碼與可訓練操作，中途不量測或重設；這與增加 shots（重複準備量子狀態並量測的次數）、重新抽樣是不同的事。本章先比較二維資料只編碼一次或重複編碼，再將四個特徵分成兩段送入兩個量子位元，區分「每個特徵第一次進入」與「同一特徵再次進入」。配對實驗保留相同的可訓練參數數量、Ansatz（可調電路模板）與初始權重，只改變資料編碼的排程，並記錄新增的旋轉閘與深度。重點是重複操作不保證帶來更好的分類結果；相鄰同軸旋轉可能合併，相同訓練評估次數也不代表相同計算成本。實作沿用 Iris 資料，在 NumPy CPU 完成訓練，再以 CUDA-Q CPU／GPU 核對保存模型。單次、分段與重複編碼需要分開比較。配對控制有助於追查排程的影響，但輸出曲線改變不等於分類能力提升，也不是量子優勢的證據。

[Day21](../day21/README.md) kept two qubits fixed while comparing PCA, feature selection, and a learned bottleneck, examining the information tradeoffs of reducing four features to two. Day22 changes where data enters the circuit, using segmented and repeated encoding to explore how a small number of qubits can use more features or reuse the same input.

**Adding trainable circuit layers and introducing the data again are different changes.** The former adds operations on the state; the latter lets the input determine circuit angles again. Data re-uploading alternates data encoding with trainable operations during a single state evolution, without intermediate measurement or reset. It is distinct from increasing shots or sampling again. The chapter first compares encoding two-dimensional data once with encoding it repeatedly, then feeds four features into two qubits in two segments, distinguishing a feature's first use from its subsequent reuse. Within each experimental pair, the trainable parameter count, Ansatz, and initial weights are held fixed while the encoding schedule changes. Additional rotation gates and depth are recorded. Repetition does not guarantee better classification: adjacent rotations around the same axis may combine, and equal training evaluation budgets do not imply equal computational cost. Training uses the existing Iris data and NumPy CPU reference, followed by CUDA-Q CPU/GPU verification of saved models. Single, segmented, and repeated encoding require separate comparisons. Paired controls help isolate schedule changes, but changed output curves do not establish better classification or quantum advantage.

---

[Day21](../day21/README.md) 比較了四維到二維的壓縮。今天改變電路的資料輸入位置：**在同一個量子狀態上交替執行資料編碼與可調操作區塊，讓資料被多次使用。**

程式：[reuploading.py](reuploading.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[plot_results.py](plot_results.py)、[tests](test_reuploading.py)。完整數值見 [實驗報告](../../results/day22/README.md)。沿用 `.venv`，無新增套件或資料下載。

## 1. 多一層可調電路模板，不一定是資料重複編碼

資料重複編碼（data re-uploading）是在同一次電路執行中，再次用輸入資料設定操作角度。可調電路模板（ansatz）則提供由訓練更新的權重。

令 E(x) 為資料編碼，A(θ) 為可調操作區塊。以下箭頭依實際執行順序：

```text
single upload：|00> → E(x) → A(θ0) → A(θ1) → observe ZZ
re-uploading： |00> → E(x) → A(θ0) → E(x) → A(θ1) → observe ZZ
```

兩者都有兩個可調操作區塊，第二者額外使用一次 E(x)。重設是將量子位元重新準備回指定狀態，shots 是重複準備與量測的次數。這裡中途不量測、不重設；不是重複抽樣，也不是把同一筆資料當成兩筆訓練樣本。

Pérez-Salinas 等人的原始研究以重複資料編碼與可訓練操作建立分類器，討論量子位元數與層數的取捨。[P4] **本日是 RY／CNOT／ZZ 的受限教學模型，不是該論文完整架構或通用分類器定理的復現；通用性討論的是在指定條件與足夠資源下能表示哪些分類函數。**

## 2. 四維資料可以分段進入兩個量子位元

特徵是描述樣本的數值，四維表示每筆有四個數值。PCA（主成分分析）將原始欄位線性組合成較少座標。這裡不用先做 PCA，也可以分兩段輸入：

```text
E01 = RY(a0) on q0, RY(a1) on q1
E23 = RY(a2) on q0, RY(a3) on q1

一次完整資料 pass：E01 → A0 → E23 → A1
再次完整資料 pass：E01 → A2 → E23 → A3
```

第一次 E23 使用後兩個特徵，是分段上傳；第二次 E01／E23 才是把各特徵再次上傳。不能把「四維拆成兩段」本身就當作「每個特徵使用兩次」。特徵順序固定為花萼長度、花萼寬度、花瓣長度、花瓣寬度，沒有依測試資料結果挑順序。

這讓兩個量子位元接收四個原始特徵，但不是無損記憶體，也不能從一次量測讀回四個值。資料仍受角度映射、電路與輸出讀取限制。

## 3. 配對控制：相同參數與可調電路模板，只改排程

排程（schedule）記錄每個可調操作區塊之前是否編碼，以及使用哪兩個特徵。

`SCHEDULES` 的每個元素對應一個可調操作區塊前的資料操作：0=輸入欄位 0／1；2=輸入欄位 2／3；−1=不輸入。

| 模型 | 表示 | 排程 | 可訓練參數 | 編碼區塊 | CNOT |
|---|---|---|---:|---:|---:|
| pca_once2 | Day20 PCA2 | [0, −1] | 8 | 1 | 2 |
| pca_repeat2 | 同一份 PCA2 | [0, 0] | 8 | 2 | 2 |
| chunks_once4 | 四維原始欄位最小值／最大值縮放 | [0, 2, −1, −1] | 16 | 2 | 4 |
| chunks_repeat4 | 同一份四維表示 | [0, 2, 0, 2] | 16 | 4 | 4 |
| logistic4 | 四維標準化 | 不適用 | 5 | 0 | 0 |

RY 是以角度旋轉單一量子位元的操作；`⊗` 表示兩個位元各自執行操作。CNOT 是受控反相閘，q0 為 1 時翻轉 q1。ZZ 將兩位元量測結果相同記為 +1、不同記為 −1，期望值是依機率計算的平均值。

每個 A 區塊固定為 RY⊗RY → CNOT(q0→q1) → RY⊗RY，四個獨立參數；所有量子模型使用兩個量子位元，以 `p1=(1−⟨ZZ⟩)/2` 將 ZZ 期望值轉成類別 1 的預測機率。配對內相同隨機種子產生完全相同初始權重。配對間的輸入表示、參數量與層數不同，不能把所有差異都歸因於資料重複編碼。

電路深度計算必須依序完成的操作層數，可同時進行的操作算在同一層。原生操作是硬體直接支援的基本操作，未必和程式中的閘一對一對應。

未合併量子閘的邏輯資源：若有 L 個 A 區塊、U 個編碼區塊，則 RY 數=4L+2U、CNOT 數=L。允許不同量子位元上的單量子位元量子閘平行時，排程深度=3L+U，不含輸出讀取。因此四個量子模型的深度分別為 7、8、14、16。這是原始排程的計數，**不是最佳化後電路深度、量子硬體原生操作或實測延遲**。

## 4. 重複並不自動增加可用表達能力

相鄰同軸旋轉可以合併：

```text
RY(b) RY(a) = RY(a+b)
```

如果只是連續堆相同資料旋轉，中間沒有適當的其他操作，可能只得到角度相加。程式有矩陣測試驗證這個等式。本日編碼前後部分 RY 也能合併，故資源表刻意標明「未合併」。CNOT 與其他操作之間的次序則會影響完整電路。

本章另固定隨機種子 42 所產生的八個權重，比較 once2 與 repeat2 的二維機率反應：

![Fixed-weight schedule probe](../../results/day22/schedule_probe.png)

這張圖以人工建立、已縮放的輸入網格計算輸出機率，沒有訓練標籤、沒有挑權重，也不是測試資料的決策邊界。結果保存所有網格機率，以及四點混合差值：

```text
Δ = p(+0.5,+0.5) − p(+0.5,−0.5) − p(−0.5,+0.5) + p(−0.5,−0.5)
```

`Δ` 讀作 delta，這裡將四個點的輸出交叉相減。若反應能拆成各特徵獨立貢獻的 `f(x0)+g(x1)`，差值就會抵消為 0。

非零 Δ 表示這四點的反應無法寫成 f(x0)+g(x1)，是局部非加性例子；表達能力指模型能表示哪些輸入與輸出關係，這個四點差值不是通用的衡量方式。單次編碼本來就可能有交互作用，不能把所有交互作用都歸功於重複輸入。圖形改變也不能證明某種電路能表示另一種電路的全部函數，或能在足夠資源下逼近任意指定範圍內的連續函數。

## 5. 沿用 Iris 與僅使用訓練資料的邊界

Iris 是鳶尾花資料集，本章只比較兩個花種。訓練資料建立縮放規則與更新權重，驗證資料選擇初始化結果，測試資料留到選定後評分。SHA-256 是根據檔案內容計算的摘要，用來確認來源；隨機種子控制打亂與初始化序列。

使用 Day20 保存的 Iris 二分類資料：去重後 99 筆，資料切分隨機種子 2028／2029；每次切分均為訓練資料 59 筆、驗證資料 19 筆、測試資料 21 筆。原始資料 SHA-256、資料列編號、平均值／標準差、PCA 主成分與最小值／最大值縮放都保存。

- PCA 配對直接重用 Day20 的僅由訓練資料決定的兩維 PCA 與主成分範圍縮放，截斷至 [-1,1]。
- Chunks 配對以各原始欄位的訓練資料最小值／最大值縮放至 [-1,1]；不做 PCA、不用標籤。
- logistic 比較基準以訓練資料平均值／標準差標準化全部四維，與 Day21 相同。

標準化先減去平均值再除以標準差，標準差描述分散程度；最小值／最大值縮放把訓練範圍轉到指定區間。截斷把越界值改成邊界值。

每個量子輸入角度為 `a=π(x+1)/2`，程式將此規則命名為 `positive_half`，把 [-1, 1] 對應到 [0, π] 的旋轉角度。截斷只按套用轉換的座標數記錄，不因同一座標重複上傳而加倍。所有驗證／測試資料僅套用轉換，不重新訓練。[D14]

## 6. 訓練與實測

Brier 損失是預測機率與 0／1 標籤的平均平方誤差。座標搜尋每次嘗試增加或減少一個權重，接受誤差較低的候選；步長是改動幅度，目標函數每次評估整批訓練資料。`normal(0,0.2)` 從平均值 0、標準差 0.2 的常態分布產生起始權重。

兩個資料切分 × 五個模型 × 初始化隨機種子 42／43，共 **20 次訓練**。與 Day21 相同，normal(0,0.2) 初始化，Brier 損失、Day18 座標搜尋、步長=0.4、73 次目標函數評估，每次用全部 59 筆訓練資料，共 4,307 筆輸入評估。

每模型每資料切分依最終驗證 Brier 挑隨機種子，平手取較小隨機種子；保留全部候選模型。測試資料不參與當次選隨機種子。相同目標函數次數不代表相同量子閘／執行時間：8／16 參數模型分別只有 4／2 輪逐一調整所有權重的機會，尚不能認定訓練已收斂，也就是繼續調整仍可能改善結果。logistic 模型將輸入加權後，以 sigmoid 函數轉成 0 與 1 之間的值。一般模型比較基準使用 Brier 訓練這個輸出轉換，與常見的邏輯斯迴歸訓練方式不同；常見方式使用交叉熵，會對信心很高卻預測錯誤的結果給予較大的懲罰。

![Benchmark comparison](../../results/day22/comparison.png)

準確率是分類正確比例；混淆矩陣分別記錄各真實類別被判成哪種類別。

這次結果沒有顯示重複輸入必然比較好。完整 Brier、準確率、混淆矩陣、截斷、曲線與各訓練時間都在 [結果報告](../../results/day22/README.md)。不依測試資料成績更換隨機種子、提高特定模型預算或刪除較差的候選。

NumPy 是 Python 數值運算套件。CPU 是中央處理器，GPU 是擅長平行運算的圖形處理器；後端指定實際使用的模擬器。本例模擬器保存完整狀態向量，直接計算期望值，仍有有限數值精度的誤差。

本日所有訓練都使用 NumPy 在 CPU 上模擬完整量子狀態，`cudaq_training_calls=0`。選定參數後，CUDA-Q CPU／GPU 分別驗證四個量子模型：2×99×4=**每個後端 792 次 `observe` 呼叫**（計算指定量測量的期望值），另核對一般模型比較基準，合計各 30 筆資料分組紀錄。`shots=-1` 表示直接計算期望值，不以有限次抽樣估計；模擬也未加入硬體噪聲，不是真實量子處理器（QPU）或 GPU 訓練效能評測；JIT 是執行時將程式轉成可用形式的即時編譯，快取保存可重用的編譯產物。時間可能包含首次編譯成本或使用快取，不相除作加速幅度。

兩個資料切分互相重疊且測試資料已在 Day20 公開，所以只是探索性系列實驗，不據此判定差異是否超出隨機波動，也不宣稱量子方法優於傳統方法。若需要更強的結論，應另用未查看的保留資料，或採巢狀交叉驗證：內層選擇設定，外層保留資料評估選定方案。

## 7. 執行範例與測試

`.venv` 是專案獨立保存套件的虛擬環境，`OMP_NUM_THREADS=1` 將 CPU 平行工作的執行緒數設為 1。

依賴入口：[requirements-day22.txt](../../requirements-day22.txt)。資料與模型均已保存，示範不需要重新訓練。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day22/experiment.py train
OMP_NUM_THREADS=1 python articles/day22/experiment.py verify
OMP_NUM_THREADS=1 python articles/day22/experiment.py verify --backend nvidia
OMP_NUM_THREADS=1 python articles/day22/plot_results.py

# 四個以公分為單位的特徵，輸出五個模型使用的特徵與判為 virginica 花種的機率
OMP_NUM_THREADS=1 python articles/day22/demo.py --features 6.0 2.9 4.5 1.5

OMP_NUM_THREADS=1 python -m unittest discover -s articles/day22 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY22_TARGET=nvidia python -m unittest discover -s articles/day22 -p 'test_*.py' -v
```

示範可加 `--split-seed 2029`、`--backend nvidia`。CPU 可獨立執行訓練、驗證、示範與測試；完整報告產生器需要兩後端摘要。輸出位於 `results/day22/`，重跑覆寫同名檔案。重訓後須重跑 verify、plot；產物 SHA-256 防止報告混用舊的後端摘要。

六項測試涵蓋僅以訓練資料建立轉換規則與截斷處理、不重複輸入與 Day14 的相容性、兩執行後端對照、四個特徵都影響輸出、相鄰旋轉合併，以及錯誤輸入。GPU 驗證與測試退出碼（結束時回報的狀態數字）為 0，但結束時有 `cudaErrorCudartUnloading`，根因未定位。

## 8. 下一步與來源

量子核方法用量子編碼後的狀態計算兩筆資料的相似程度，這裡的 kernel 指核函數，不是描述電路的量子核心程式。Day23 將實作量子核方法，從「訓練電路權重」轉向「用量子特徵空間計算樣本相似度」。

- [P4] Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, José I. Latorre. “Data re-uploading for a universal quantum classifier.” *Quantum* 4, 226 (2020). [出版頁與 DOI](https://quantum-journal.org/papers/q-2020-02-06-226/)。原始概念來源，不宣稱本日是論文完整復現。
- [D14] [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：僅使用訓練資料的前處理。
- [D17] 資料來源與授權：[Day20 UCI 紀錄](../../data/day20/README.md)。

出版頁查閱日期 2026-09-07；頁面標示發表版本對應 arXiv v2，另列較新的 v3，本日引用期刊版本。完整書目見 [REFERENCES.md](../../REFERENCES.md)。

接續：[Day23｜Quantum Kernel](../day23/README.md)。
