# Day24 實驗結果

本檔由 `plot_results.py` 產生；768 個初始化電路、1,536 個單參數梯度。沒有資料集分類或optimizer訓練。

## 梯度統計

![Variance](variance.png)

每組32個seeds，以ddof=1估計variance。Local=Z0，global=全域Z parity；都用第一個RY參數，不混合不同參數的梯度。

| Qubits | Blocks | Init | Cost | Mean | Variance | RMS | Median abs | Fraction abs<0.01 |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 2 | 1 | uniform | local | -6.589e-02 | 5.680e-01 | 7.447e-01 | 7.546e-01 | 0.00% |
| 2 | 1 | uniform | global | -4.505e-02 | 2.312e-01 | 4.754e-01 | 3.426e-01 | 0.00% |
| 2 | 1 | small | local | 2.360e-03 | 1.060e-02 | 1.014e-01 | 6.731e-02 | 9.38% |
| 2 | 1 | small | global | 2.389e-03 | 1.048e-02 | 1.008e-01 | 6.644e-02 | 9.38% |
| 2 | 4 | uniform | local | -7.087e-02 | 2.557e-01 | 5.027e-01 | 4.611e-01 | 3.12% |
| 2 | 4 | uniform | global | -4.080e-02 | 9.659e-02 | 3.086e-01 | 2.249e-01 | 0.00% |
| 2 | 4 | small | local | -2.558e-02 | 3.503e-02 | 1.860e-01 | 1.503e-01 | 0.00% |
| 2 | 4 | small | global | -2.471e-02 | 3.444e-02 | 1.843e-01 | 1.490e-01 | 3.12% |
| 2 | 8 | uniform | local | -4.522e-02 | 1.002e-01 | 3.148e-01 | 1.992e-01 | 3.12% |
| 2 | 8 | uniform | global | 5.240e-02 | 1.581e-01 | 3.949e-01 | 2.807e-01 | 0.00% |
| 2 | 8 | small | local | 9.838e-03 | 5.105e-02 | 2.226e-01 | 1.617e-01 | 0.00% |
| 2 | 8 | small | global | 1.142e-02 | 4.838e-02 | 2.168e-01 | 1.612e-01 | 3.12% |
| 4 | 1 | uniform | local | -6.589e-02 | 5.680e-01 | 7.447e-01 | 7.546e-01 | 0.00% |
| 4 | 1 | uniform | global | 3.364e-02 | 6.471e-02 | 2.526e-01 | 1.159e-01 | 6.25% |
| 4 | 1 | small | local | 2.360e-03 | 1.060e-02 | 1.014e-01 | 6.731e-02 | 9.38% |
| 4 | 1 | small | global | 2.351e-03 | 1.020e-02 | 9.944e-02 | 6.592e-02 | 9.38% |
| 4 | 4 | uniform | local | 8.180e-02 | 1.465e-01 | 3.855e-01 | 2.918e-01 | 0.00% |
| 4 | 4 | uniform | global | 3.096e-03 | 3.275e-02 | 1.781e-01 | 1.036e-01 | 9.38% |
| 4 | 4 | small | local | -1.854e-02 | 2.630e-02 | 1.607e-01 | 1.268e-01 | 3.12% |
| 4 | 4 | small | global | -1.808e-02 | 2.241e-02 | 1.485e-01 | 1.147e-01 | 6.25% |
| 4 | 8 | uniform | local | -8.075e-02 | 6.582e-02 | 2.651e-01 | 1.705e-01 | 0.00% |
| 4 | 8 | uniform | global | -3.054e-02 | 2.117e-02 | 1.464e-01 | 1.016e-01 | 3.12% |
| 4 | 8 | small | local | -7.040e-02 | 4.931e-02 | 2.296e-01 | 1.462e-01 | 0.00% |
| 4 | 8 | small | global | -6.125e-02 | 3.941e-02 | 2.048e-01 | 1.401e-01 | 3.12% |
| 6 | 1 | uniform | local | -6.589e-02 | 5.680e-01 | 7.447e-01 | 7.546e-01 | 0.00% |
| 6 | 1 | uniform | global | -3.142e-02 | 1.040e-02 | 1.052e-01 | 2.575e-02 | 25.00% |
| 6 | 1 | small | local | 2.360e-03 | 1.060e-02 | 1.014e-01 | 6.731e-02 | 9.38% |
| 6 | 1 | small | global | 2.310e-03 | 1.007e-02 | 9.881e-02 | 6.413e-02 | 9.38% |
| 6 | 4 | uniform | local | 4.705e-02 | 1.505e-01 | 3.847e-01 | 1.581e-01 | 6.25% |
| 6 | 4 | uniform | global | -1.051e-02 | 7.723e-03 | 8.713e-02 | 5.285e-02 | 21.88% |
| 6 | 4 | small | local | 7.978e-03 | 3.132e-02 | 1.744e-01 | 1.416e-01 | 3.12% |
| 6 | 4 | small | global | 9.402e-03 | 2.676e-02 | 1.613e-01 | 1.364e-01 | 3.12% |
| 6 | 8 | uniform | local | -5.631e-02 | 6.690e-02 | 2.607e-01 | 1.635e-01 | 3.12% |
| 6 | 8 | uniform | global | -1.363e-02 | 8.013e-03 | 8.915e-02 | 6.154e-02 | 15.62% |
| 6 | 8 | small | local | -2.869e-02 | 8.006e-02 | 2.800e-01 | 2.269e-01 | 3.12% |
| 6 | 8 | small | global | -1.927e-02 | 5.526e-02 | 2.322e-01 | 1.887e-01 | 6.25% |
| 8 | 1 | uniform | local | -6.589e-02 | 5.680e-01 | 7.447e-01 | 7.546e-01 | 0.00% |
| 8 | 1 | uniform | global | 1.016e-02 | 3.594e-03 | 5.987e-02 | 1.335e-02 | 46.88% |
| 8 | 1 | small | local | 2.360e-03 | 1.060e-02 | 1.014e-01 | 6.731e-02 | 9.38% |
| 8 | 1 | small | global | 2.204e-03 | 9.871e-03 | 9.781e-02 | 6.374e-02 | 9.38% |
| 8 | 4 | uniform | local | -1.082e-01 | 1.998e-01 | 4.531e-01 | 3.713e-01 | 6.25% |
| 8 | 4 | uniform | global | -4.286e-03 | 4.034e-04 | 2.023e-02 | 1.209e-02 | 40.62% |
| 8 | 4 | small | local | -1.753e-02 | 4.433e-02 | 2.080e-01 | 1.278e-01 | 15.62% |
| 8 | 4 | small | global | -1.266e-02 | 3.399e-02 | 1.819e-01 | 1.164e-01 | 15.62% |
| 8 | 8 | uniform | local | 2.937e-02 | 7.004e-02 | 2.621e-01 | 2.186e-01 | 0.00% |
| 8 | 8 | uniform | global | -3.874e-03 | 1.863e-03 | 4.266e-02 | 3.354e-02 | 18.75% |
| 8 | 8 | small | local | -1.128e-02 | 6.581e-02 | 2.527e-01 | 1.429e-01 | 0.00% |
| 8 | 8 | small | global | -2.453e-03 | 3.938e-02 | 1.953e-01 | 1.190e-01 | 0.00% |

## 可解析 classical 對照

![Product control](product_control.png)

Product RY states、uniform angles的精確ensemble variance為local=1/2、global=2^(-n)。樣本只有32筆，sample variance不會恰好等於理論值。對照另保存自己的angles sampling seeds；與主實驗的交錯RY／RZ隨機向量不逐筆相同。

## CUDA-Q 驗證

| Backend | Gradient records | Observe calls | Max absolute gradient error | Passed |
|---|---:|---:|---:|---|
| qpp-cpu | 1536 | 3072 | 1.867e-15 | True |
| nvidia | 1536 | 3072 | 8.710e-07 | True |

CUDA-Q逐個樣本以±π/2計算parameter-shift；NumPy使用tangent-state導數。每gradient兩次observe，共3,072次／backend。沒有finite shots或noise。
GPU退出有cudaErrorCudartUnloading，程序exit code=0且檢查通過；根因未定位。CUDA-Q時間含可能的compilation／cache，未做隔離效能比較。

## 解讀邊界

32 seeds與n≤8不足以由曲線證明漸近exponential scaling；不擬合直線後宣稱證明Barren Plateau。
單層CZ與RZ皆對Z讀出對易，所以depth1的local／global成本與product-state解析式相同；global variance下降在此不需要深電路或複雜entanglement解釋。
小角度初始化的variance小也可能因靠近stationary point，並非較差或較好的訓練能力證明。全零權重兩個cost皆1、梯度為0，是明確測試的例子。
Global與local是不同objective，換成local不保證仍解同一個任務。只測index0，不代表全gradient norm或所有參數的可訓練性。

## 原始紀錄

- [Protocol](protocol.json)、[全部weights／costs／gradients](samples.json)、[Statistics](statistics.json)
- [Product control](product_control.json)、[Reference summary](summary.json)
- [CPU summary](qpp-cpu/summary.json)、[CPU shifted expectations](qpp-cpu/verification.json)
- [GPU summary](nvidia/summary.json)、[GPU shifted expectations](nvidia/verification.json)
- [教學與重跑](../../articles/day24/README.md)
