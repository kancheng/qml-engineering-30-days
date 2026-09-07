# Day 23｜Quantum Kernel：QML 不只有 QNN

[Day22](../day22/README.md) 訓練電路權重。今天固定 feature map，改用量子態之間的相似度建立 kernel matrix，再交給 classical kernel ridge 求解。**Quantum kernel function 與 Day07 的 CUDA-Q `@cudaq.kernel` 程式函式是不同概念。**

程式：[kernels.py](kernels.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[圖表與報告](plot_results.py)、[測試](test_kernels.py)。實測見 [results/day23](../../results/day23/README.md)。沿用獨立 `.venv`，無新套件、無新資料下載。

## 1. 從單筆預測到兩筆資料的相似度

令 `|φ(x)⟩=U(x)|00⟩`，本日的 fidelity kernel 為：

```text
k(x,z) = |⟨φ(z)|φ(x)⟩|²
K[i,j] = k(x_i, x_j)
```

訓練資料有 n 筆，Gram matrix K 是 n×n。對 m 筆新資料建立的矩陣是 m×n，columns 必須對應保存的 train row 順序，不能改成 new×new。預測需要與 training samples 比較，並非只保存一小組量子電路 weights。

Havlíček 等人的研究展示 quantum feature space 與 kernel estimation 的分類方法。[P5] 本日實作自己的小型 RY feature map 與 ridge solver，不是原論文 feature map／SVM 實驗的完整復現。

## 2. Compute–Uncompute 電路

沿用兩 qubit 接收四個特徵的思路，固定執行順序：

```text
U(x): RY(a0)⊗RY(a1) → CNOT(q0→q1) → RY(a2)⊗RY(a3)
a_j = π(x_j+1)/2
```

沒有 trainable quantum parameters。要比較 x 與 z，在同一個 state 上先做 U(x)，再做 U(z)†：

```text
|00> → U(x) → U(z)† → P(00)
P(00) = |⟨00|U(z)†U(x)|00⟩|² = k(x,z)
```

Inverse 必須倒轉 gate 順序；RY 的角度取負，CNOT 是自身的 inverse。[kernels.py](kernels.py) 明確寫出這些操作。兩 qubit 的 all-zero projector 是：

```text
|00⟩⟨00| = (I+Z0)(I+Z1)/4
```

本日以 `cudaq.observe(..., shots_count=-1)` 取得 exact projector expectation。一次 observe 包含此 observable 的組合，不等於真實硬體只需一次 shot。原始 overlap circuit 有8個RY與2個CNOT，編譯器可能合併相鄰旋轉；未做 native gate成本或QPU latency量測。

NumPy reference 分別準備兩個 state，再以 `abs(vdot(state_z,state_x))**2` 計算。這與 compute–uncompute 使用不同計算路徑，供CPU／GPU核對。

## 3. 為何它是合法的 Kernel？

對 normalized states，k(x,x)=1、k(x,z)=k(z,x)、值落在[0,1]。更重要的是，以 `ρ(x)=|φ(x)⟩⟨φ(x)|` 為 feature：

```text
k(x,z) = Tr[ρ(x)ρ(z)]
cᵀKc = ||Σ_i c_i ρ(x_i)||²_F ≥ 0
```

因此精確 Gram matrix 是 positive semidefinite（PSD），不一定 positive definite；相似或重複的表示可能使矩陣奇異。測試核對density feature inner product、對稱性、對角線、最小特徵值。浮點運算可能出現很小的負特徵值，結果保留原始數值，不偷換成PSD投影。

另一個容易忽略的細節：若所有states最後都加相同、不依賴資料的unitary V，overlap不變，因為V†V=I。因此只在feature map尾端增加固定entangling gate不會自動讓kernel更有表達能力。程式有此項測試。本日CNOT放在兩段資料旋轉中間，後面仍有依資料變化的操作。

## 4. Classical Baselines 與 Ridge 求解

所有方法接收同一份raw4 train-minmax表示、clip至[-1,1]，不使用PCA或labels fit scaler：

| Kernel | 定義 | 設定 |
|---|---|---|
| Quantum | squared state overlap | 固定兩qubit feature map |
| RBF | exp(−γ‖x−z‖²) | 預先固定γ=1 |
| Linear＋constant | 1+x·z/4 | 等價feature [1,x/2] |

Linear加入constant feature，使模型能表示常數位移；其係數也受regularization，不等於額外不受懲罰的intercept。它的對角線不要求等於1，各kernel的scale不同，所以相同λ也不表示完全相同正則化效果。

本日用NumPy實作kernel ridge，概念參照官方KRR文件。[D20]

```python
alpha = np.linalg.solve(K_train + lam * np.eye(n), y_train)
raw_score = K_query_train @ alpha
score = np.clip(raw_score, 0, 1)
predicted_class = score >= 0.5
```

求解對應未clip squared-error＋RKHS norm regularization：`sum_i(f(x_i)-y_i)^2 + λ||f||²`，沒有除以n的loss慣例，沒有額外intercept。使用solve而不是顯式matrix inverse。每個λ都在相同K上求解，無需重新執行量子電路。

這是regression-to-binary-label的分類示範，不是SVM。Ridge輸出可能超出[0,1]，所以明確保存raw與clipped scores，並用clipped score計算Brier與0.5 threshold accuracy；它不是經校準的機率。求解器也不是直接最小化clipped Brier。測試另以linear primal解核對dual解及training column重排一致性。

## 5. 資料、選參數與實測

使用Day20的binary Iris：去重99筆、dataset seeds2028／2029，每組train59／validation19／test21。所有row IDs、來源SHA-256與scaler保存。各模型只測λ∈{0.01,0.1,1}，依validation clipped-score Brier挑選，平手取較小λ。沒有初始化seed；共2×3×3=**18次classical ridge solves**。

![Kernel matrices](../../results/day23/gram_matrices.png)

圖中train按原有類別分組順序顯示，白線為類別交界；labels不參與kernel計算。每圖使用獨立色階，不能由亮度直接比較不同kernel的品質。

![Comparison](../../results/day23/comparison.png)

完整數值見 [結果報告](../../results/day23/README.md)。Quantum兩組test accuracy皆90.48%，本次並未一致勝過classical kernels。不能從熱圖區塊或小樣本test成績推論量子優勢。

沿用已公開的Day20 test及重疊splits，屬探索性實驗；正式選模型需另留未查看holdout或nested cross-validation。RBF gamma固定、λ grid很小，也不是最佳調參排行榜。與Day22的73次objective搜尋預算並不相同，不混稱為跨日公平speed／accuracy benchmark。

## 6. Kernel 的成本移到哪裡？

本日先在NumPy reference建立matrices、選λ與alpha，CUDA-Q training calls=0；再用CPU／GPU重建**全部quantum矩陣元素**並以凍結alpha核對分數。

每個split需要：

| Matrix | Shape | Pair evaluations |
|---|---|---:|
| Train×train | 59×59 | 3,481 |
| Validation×train | 19×59 | 1,121 |
| Test×train | 21×59 | 1,239 |
| 合計 | 99×59 | 5,841 |

每backend兩splits共**11,682次exact observe**。本日連對角線與對称元素都實測；一般可利用對稱性減少train配對數，但這裡不省略以便核對。預測一筆新資料需59個kernel values。

Dense kernel方法需要O(n²)矩陣儲存，直接dense求解通常O(n³)；query配對為O(mn)。NumPy reference可先cache每筆state再做矩陣乘法，與逐pair執行compute–uncompute的成本不同，不能直接用兩種時間宣稱GPU加速。此兩qubit模型可有效classically simulate，沒有計算優勢結論。

GPU fp32的kernel誤差會經alpha放大，因此分別設定matrix error<1e-5、clipped-score error<1e-4。完整reference與backend矩陣以NPZ保存，沒有拿GPU matrix重新fit或選λ。此範圍是exact、無噪聲simulator；finite shots可能破壞實測矩陣的PSD，估計／修正策略留待後續，未在本日假裝完成。

## 7. 重跑與單筆示範

沿用[requirements-day23.txt](../../requirements-day23.txt)，不安裝scikit-learn。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day23/experiment.py train
OMP_NUM_THREADS=1 python articles/day23/experiment.py verify
OMP_NUM_THREADS=1 python articles/day23/experiment.py verify --backend nvidia
OMP_NUM_THREADS=1 python articles/day23/plot_results.py

OMP_NUM_THREADS=1 python articles/day23/demo.py --features 6.0 2.9 4.5 1.5
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day23 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY23_TARGET=nvidia python -m unittest discover -s articles/day23 -p 'test_*.py' -v
```

Demo可加`--split-seed 2029`或`--backend nvidia`，載入保存的train順序、scaler與alpha，不需重訓。預設CPU可獨立執行train／verify／demo／tests；完整報告需要兩backend summaries。

輸出位於`results/day23/`，重跑覆寫同名檔案；重訓後重新verify與plot。報告檢查protocol／datasets／selected的artifact hash，避免引用舊backend summary。GPU退出仍有`cudaErrorCudartUnloading`；數值驗證與測試exit code=0，根因未定位。

## 8. 下一步與來源

Day24回到trainability，實作Barren Plateau相關梯度實驗。

- [P5] Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta. “Supervised learning with quantum-enhanced feature spaces.” *Nature* 567, 209–212 (2019). [DOI與出版頁](https://www.nature.com/articles/s41586-019-0980-2)。
- [D20] [Kernel ridge regression](https://scikit-learn.org/stable/modules/kernel_ridge.html)：KRR與kernel trick；本文以NumPy自行實作。
- [D17] [UCI資料紀錄](../../data/day20/README.md)：binary subset與去重政策。

查閱日期2026-09-07；完整索引見[REFERENCES.md](../../REFERENCES.md)。

接續：[Day24｜Barren Plateau](../day24/README.md)。
