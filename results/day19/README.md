# Day 19 Hybrid Model 實驗紀錄

[教學與重跑方式](../../articles/day19/README.md)。十二個參數分成 encoder 6、quantum 4、head 2，三筆固定資料，八次更新。CPU／GPU 各五個測試通過。

| Backend | 初始 Brier | 最終 Brier | 全參數梯度最大誤差 | Reload error |
|---|---:|---:|---:|---:|
| qpp-cpu | 0.236562908 | 0.200477584 | 3.26e-11 | 0 |
| nvidia | 0.236562905 | 0.200477583 | 2.02e-08 | 0 |

三組參數相對初始值的 L2 改變量約為 encoder=0.329142、quantum=0.0391053、head=0.0423885。各組單位／維度不同，這些數字只驗證參數有更新，不排名重要性。

最終三筆 probabilities 約 [0.681411,0.588929,0.608725]，labels=[1,0,1]；若以 0.5 分類，中間樣本仍錯。這是完整梯度的整合測試，沒有 test split、分類成效或泛化宣稱。

## 紀錄與圖表

- qpp-cpu：[summary](qpp-cpu/summary.json)、[全部更新與梯度](qpp-cpu/history.json)、[checkpoint](qpp-cpu/checkpoint.json)、[訓練圖](qpp-cpu/hybrid_training.png)。
- nvidia：[summary](nvidia/summary.json)、[全部更新與梯度](nvidia/history.json)、[checkpoint](nvidia/checkpoint.json)、[訓練圖](nvidia/hybrid_training.png)。

每筆 history 保存 12 個參數、實作梯度與 NumPy 完整 loss finite difference。summary 保留所有資料、labels、最終 hidden／angles／quantum expectation／logits，checkpoint roundtrip 已驗證。九次 loss＋gradient 評估各 3×13 次 observe，共 351 次訓練呼叫；最終推論／reload 另計。

## 環境與限制

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6、Matplotlib 3.10.6，OMP_NUM_THREADS=1。RTX 3060 Laptop 6 GiB、driver 570.211.01，CPU fp64／GPU fp32；classical 部分皆在 NumPy CPU，無新增依賴。UTC 2026-09-06 23:xx 對應台灣 2026-09-07 07:xx。

GPU 測試與實驗結束時出現 cudaErrorCudartUnloading，退出碼為 0，數值核對通過，根因尚未確認。全部為無噪聲 exact simulator，沒有 finite-shot 訓練、PyTorch autograd bridge、QPU 或量子優勢結論。
