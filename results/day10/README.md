# Day 10 實驗紀錄

本日沿用 [Day 9 PQC](../../articles/day09/pqc.py)，以九筆固定輸入訓練四個 weights，另評估四筆 holdout。教學與重跑指令見 [Day 10](../../articles/day10/README.md)。

## 實驗設定

- Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6，沿用 `.venv`。
- CPU：qpp-cpu fp64；GPU：RTX 3060 Laptop 6 GiB、nvidia fp32，driver 570.211.01。
- 無噪聲、exact expectation 訓練，`OMP_NUM_THREADS=1`。
- 一層四參數，初始化 normal(0,0.2)，seeds 42 與 43。
- 座標搜尋：step=0.4，無改善的一輪後減半，step tolerance=1e-4，最多 40 輪。
- holdout 只在訓練完成後評估；常數 baseline 使用訓練標籤平均值。

## 結果

| Backend | Seed | 初始 train MSE | 最終 train MSE | 初始 holdout MSE | 最終 holdout MSE |
|---|---:|---:|---:|---:|---:|
| qpp-cpu | 42 | 0.381884 | 1.06833e-05 | 0.347774 | 1.04005e-05 |
| qpp-cpu | 43 | 0.153582 | 0.000253063 | 0.123284 | 0.000238059 |
| nvidia | 42 | 0.381884 | 1.06843e-05 | 0.347774 | 1.04214e-05 |
| nvidia | 43 | 0.153582 | 0.000253063 | 0.123284 | 0.000238058 |

四次實驗均在 40 輪預算耗盡時停止（`max_sweeps`），各 321 次 objective／2,889 次訓練 observe。最終誤差均下降，但沒有宣稱達到全域最小值。常數 baseline 的 train MSE 為 0.363920499，holdout MSE 為 0.505393415。

最終 CUDA-Q／NumPy expectation 最大差異：CPU 約 4.44e-16，GPU 約 1.56e-7。四次訓練每輪保存的 loss 也已逐筆以 NumPy reference 重新核對（誤差 < 1e-5）。

Seed 43 的最終誤差高於 seed 42，展示初始化與有限搜尋預算會影響結果。所有 seeds 都公開保留。

## 檔案與驗證範圍

每個 backend／seed 子目錄保存 `summary.json`、`history.json`、`training_curve.csv`、`predictions.json`。CSV 的 loss 是每輪結束後的訓練 MSE，第 0 輪是初始化；step 是該輪使用的角度步長。

CPU／GPU 各 5 個 unittest 通過，涵蓋 objective、optimizer、解析標籤與實際 hybrid 更新。每次完整訓練另核對 loss 不增加、最終訓練 loss 改善，以及最終 forward 對 NumPy 的誤差小於 1e-5。

finite-shot 讀出只在訓練完成後進行，每筆 1,000 shots；sampling seed 與 counts 存在 predictions。它沒有參與模型選擇或 optimizer。

GPU 測試與實驗結束時出現 `cudaErrorCudartUnloading`，程序退出碼為 0，數值核對通過；此訊息的根因尚未確認。CPU／GPU 的浮點 precision 不同，接近的候選 loss 可能產生不同搜尋路徑，不要求最終 weights 一致。

這是可表示解析函數的合成回歸示範，沒有 QPU 實測、含噪聲訓練、分類 accuracy 或量子優勢結論；四筆 holdout 不足以估計真實任務的泛化能力。本日不作速度比較。
