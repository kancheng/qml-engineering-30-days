# Day 17 實驗紀錄

[教學與重跑方式](../../articles/day17/README.md)。兩個 backend 各六組電路梯度、24 筆 finite-difference 對照、五筆 loss／gradient 紀錄、20 筆抽樣梯度；各五個測試通過。

| Backend | Parameter-shift 最大誤差 | Loss gradient 最大誤差 | 初始 loss | 四次更新後 loss |
|---|---:|---:|---:|---:|
| qpp-cpu | 5.55e-16 | 1.82e-11 | 0.47983457 | 0.18944952 |
| nvidia | 8.31e-08 | 4.93e-08 | 0.47983455 | 0.18944950 |

## 有限差分與抽樣診斷

| Backend | h | 最大 FD 絕對誤差 |
|---|---:|---:|
| qpp-cpu | 0.1 | 0.00166209 |
| qpp-cpu | 0.001 | 1.66292e-07 |
| qpp-cpu | 1e-05 | 3.8212e-11 |
| qpp-cpu | 1e-07 | 3.66586e-09 |
| nvidia | 0.1 | 0.00166239 |
| nvidia | 0.001 | 6.64024e-05 |
| nvidia | 1e-05 | 0.00542093 |
| nvidia | 1e-07 | 0.794028 |

小 h 的誤差不是 parameter-shift 測試失敗；FD 是診斷項目，不要求每個 step 小於 1e-5。GPU fp32 的相減誤差會被 1/(2h) 放大。

| Backend | 每側 shots | 10 次抽樣的 gradient RMSE |
|---|---:|---:|
| qpp-cpu | 100 | 0.0473179 |
| qpp-cpu | 1000 | 0.0138883 |
| nvidia | 100 | 0.0473179 |
| nvidia | 1000 | 0.0138883 |

抽樣僅為固定一個導數的小型展示，不保證每次增加 shots 都讓實際誤差下降。訓練使用 exact expectation；兩筆同為 class 1 的資料只驗證 chain-rule 更新，不評估分類泛化。

## 原始檔案

- qpp-cpu：[summary](qpp-cpu/summary.json)、[梯度](qpp-cpu/gradients.json)、[更新](qpp-cpu/training.json)、[抽樣 counts](qpp-cpu/sampling.json)、[誤差圖](qpp-cpu/gradient_error.png)。
- nvidia：[summary](nvidia/summary.json)、[梯度](nvidia/gradients.json)、[更新](nvidia/training.json)、[抽樣 counts](nvidia/sampling.json)、[誤差圖](nvidia/gradient_error.png)。

## 環境與限制

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6、Matplotlib 3.10.6；OMP_NUM_THREADS=1。RTX 3060 Laptop 6 GiB、driver 570.211.01；CPU fp64／GPU fp32，沿用 .venv 無新增依賴。UTC 2026-09-06 21:xx 對應台灣 2026-09-07 05:xx。

GPU 測試與實驗退出時有 cudaErrorCudartUnloading，退出碼 0、數值檢查通過，根因尚未確認。沒有 QPU、autograd 整合、adjoint differentiation 或量子優勢驗證。
