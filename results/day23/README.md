# Day23 實驗結果

由 `plot_results.py` 依保存的 JSON／NPZ 產生。Binary Iris99筆、兩個Day20 splits，train／validation／test=59／19／21。三種 kernel×三個 λ×兩splits，共18次 NumPy ridge solves。

## 選定結果

| Split | Kernel | λ | Test Brier | Accuracy | Confusion | Clipped scores train/val/test |
|---|---|---:|---:|---:|---|---|
| 2028 | quantum | 0.1 | 0.066907 | 90.48% | [[10, 0], [2, 9]] | [22, 7, 9] |
| 2028 | rbf | 0.1 | 0.067819 | 90.48% | [[9, 1], [1, 10]] | [22, 10, 11] |
| 2028 | linear | 0.01 | 0.041345 | 95.24% | [[9, 1], [0, 11]] | [20, 9, 12] |
| 2029 | quantum | 0.01 | 0.071703 | 90.48% | [[10, 0], [2, 9]] | [22, 9, 12] |
| 2029 | rbf | 0.01 | 0.028753 | 100.00% | [[10, 0], [0, 11]] | [23, 5, 7] |
| 2029 | linear | 0.1 | 0.047579 | 90.48% | [[9, 1], [1, 10]] | [14, 3, 7] |

Train-prior baseline p=29/59：test Brier=0.250475、accuracy=47.62%（兩split相同）。Confusion row=true／column=predicted。Clipped ridge scores 不是經校準的機率。

![Comparison](comparison.png)

## Gram matrix

![Gram matrices](gram_matrices.png)

圖中前30筆為class0、後29筆為class1；排序只為顯示，未使用labels計算kernel。各圖色階獨立。

| Split | Kernel | Symmetry error | Min eigenvalue | Diagonal error from 1 |
|---|---|---:|---:|---:|
| 2028 | quantum | 0.000e+00 | -6.053e-15 | 8.882e-16 |
| 2028 | rbf | 0.000e+00 | 2.202e-05 | 0.000e+00 |
| 2028 | linear | 0.000e+00 | -1.357e-14 | 9.124e-01 |
| 2029 | quantum | 0.000e+00 | -6.496e-15 | 8.882e-16 |
| 2029 | rbf | 0.000e+00 | 7.939e-05 | 0.000e+00 |
| 2029 | linear | 0.000e+00 | -1.689e-14 | 8.538e-01 |

Linear kernel 不要求 diagonal=1；接近零的負特徵值可能是浮點誤差。未做 PSD projection 或替換對角線。

## 全部 validation 候選

| Split | Kernel | λ | Validation Brier | NumPy solve seconds |
|---|---|---:|---:|---:|
| 2028 | quantum | 0.01 | 0.061040 | 0.000192 |
| 2028 | quantum | 0.1 | 0.059647 | 0.000076 |
| 2028 | quantum | 1.0 | 0.061662 | 0.000062 |
| 2028 | rbf | 0.01 | 0.037982 | 0.000085 |
| 2028 | rbf | 0.1 | 0.032506 | 0.000063 |
| 2028 | rbf | 1.0 | 0.035188 | 0.000055 |
| 2028 | linear | 0.01 | 0.041602 | 0.000088 |
| 2028 | linear | 0.1 | 0.041752 | 0.000066 |
| 2028 | linear | 1.0 | 0.057464 | 0.000059 |
| 2029 | quantum | 0.01 | 0.056529 | 0.000130 |
| 2029 | quantum | 0.1 | 0.058932 | 0.000065 |
| 2029 | quantum | 1.0 | 0.075824 | 0.000060 |
| 2029 | rbf | 0.01 | 0.070726 | 0.000102 |
| 2029 | rbf | 0.1 | 0.076835 | 0.000064 |
| 2029 | rbf | 1.0 | 0.075356 | 0.000065 |
| 2029 | linear | 0.01 | 0.074047 | 0.000087 |
| 2029 | linear | 0.1 | 0.072871 | 0.000074 |
| 2029 | linear | 1.0 | 0.079113 | 0.000058 |

## CPU／GPU 完整矩陣驗證

| Backend | Observe calls | Max kernel error | Max clipped-score error | Passed |
|---|---:|---:|---:|---|
| qpp-cpu | 11682 | 8.882e-16 | 5.479e-14 | True |
| nvidia | 11682 | 3.625e-07 | 1.527e-05 | True |

每backend核對2×(59²+19×59+21×59)=11,682個矩陣元素，未利用對稱性省略量測；每元素一次projector observe，shots=-1、無噪聲。
GPU fp32 kernel 誤差會經alpha加權放大，故矩陣與下游分數分開檢查。alpha凍結於NumPy結果，沒有用GPU矩陣重訓或重新選λ。
GPU退出有cudaErrorCudartUnloading，數值驗證與程序exit code=0；根因未定位。時間含可能的JIT／cache，不宣稱GPU加速。

## 限制

固定小型 feature map、固定RBF gamma=1、三個λ，沒有窮盡模型調參。Linear使用常數feature，其係數也受regularization。
KRR解的是未clip squared-error正則化問題，validation才以clipped-score Brier選λ；不是SVM、不是直接最小化clipped Brier。
沿用已公開test與重疊splits，屬探索性比較；不宣稱量子優勢。本次量子模型未一致勝過classical kernels。

## 原始紀錄

- [Protocol](protocol.json)、[資料／scaler](datasets.json)、[Candidates](candidates.json)、[Selected](selected.json)
- [Training summary](training_summary.json)、[Reference matrices NPZ](matrices.npz)
- [CPU summary](qpp-cpu/summary.json)、[CPU verification](qpp-cpu/verification.json)、[CPU matrices](qpp-cpu/matrices.npz)
- [GPU summary](nvidia/summary.json)、[GPU verification](nvidia/verification.json)、[GPU matrices](nvidia/matrices.npz)
- [教學與重跑](../../articles/day23/README.md)
