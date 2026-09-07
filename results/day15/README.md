# Day 15 實驗紀錄

[教學與重跑方式](../../articles/day15/README.md)。固定 XOR dataset seed=2026，train／validation／test 各 12 筆；兩個初始化 seeds 42／43，各 16 輪座標搜尋。

| Backend | 選定 seed | Test accuracy | Test Brier | 1,000-shot accuracy | 最大 probability reference 誤差 |
|---|---:|---:|---:|---:|---:|
| qpp-cpu | 43 | 100.0% | 0.050717993 | 100.0% | 4.44e-16 |
| nvidia | 43 | 100.0% | 0.050717989 | 100.0% | 6.7e-08 |

兩個 backend 的 test confusion matrix 都是 [[6,0],[0,6]]，row=true、column=prediction。常數 baseline p1=0.5，accuracy=50%、Brier=0.25。sampled test Brier=0.04921875；低於 exact Brier 是本次抽樣結果，不代表 finite shots 更好。

## 模型選擇與訓練

以兩個 seeds 的最終 validation Brier 選模型，沒有用 test 或挑最佳 epoch。四次訓練皆達到 max_sweeps 預算後停止，各 129 次 objective／1,548 次訓練 observe；不能把預算耗盡稱為收斂。CPU／GPU 各四個測試通過，checkpoint 重新載入後的 CUDA-Q 與 NumPy 預測一致至 1e-5。

## 原始資料與圖表

每個 backend 子目錄有 dataset.json、candidates.json、checkpoint.json、predictions.json、test_sampling.json、summary.json、training_curve.csv、training_curve.png、decision_boundary.png、boundary_grid.npz。

- qpp-cpu：[摘要](qpp-cpu/summary.json)、[兩個候選訓練紀錄](qpp-cpu/candidates.json)、[Checkpoint](qpp-cpu/checkpoint.json)、[逐 split 評分](qpp-cpu/predictions.json)、[曲線](qpp-cpu/training_curve.png)、[決策邊界](qpp-cpu/decision_boundary.png)。
- nvidia：[摘要](nvidia/summary.json)、[兩個候選訓練紀錄](nvidia/candidates.json)、[Checkpoint](nvidia/checkpoint.json)、[逐 split 評分](nvidia/predictions.json)、[曲線](nvidia/training_curve.png)、[決策邊界](nvidia/decision_boundary.png)。

boundary 網格使用保存 weights 的 NumPy reference，不是 GPU 全網格執行；每輪 validation 曲線也標示為 NumPy。圖上的 raw [−1,1]² 包含訓練 scaler 範圍外會被 clipping 的位置。模型在座標軸附近不完全符合 XOR；test 的 12 筆資料與座標軸有間隔，100% 只描述這份 test split。

## 環境與限制

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6；RTX 3060 Laptop 6 GiB、driver 570.211.01，CPU fp64／GPU fp32。OMP_NUM_THREADS=1；新增 Matplotlib 3.10.6 與固定版本繪圖依賴，見 requirements-day15.txt。時間戳使用 UTC，2026-09-06 19:xx 對應台灣 2026-09-07 03:xx。

GPU 測試與實驗退出時出現 cudaErrorCudartUnloading，退出碼為 0、數值檢查通過，根因尚未確認。首次 CPU 繪圖有 Fontconfig cache 警告但成功匯出；繪圖程式現已將快取預設放到 /tmp。

本日沒有 QPU、含噪聲訓練、真實資料泛化或量子優勢結論。常數 baseline 只是基本對照，不取代後續公平 classical ML benchmark。
