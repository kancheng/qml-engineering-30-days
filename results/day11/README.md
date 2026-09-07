# Day 11 實驗紀錄

教學及執行指令見 [Day 11](../../articles/day11/README.md)。本日驗證 raw features → train-only scaler → RY angles → quantum state → readout，不進行訓練。

| Backend | 案例 | 最大 observable 誤差 | 最大 fidelity 與 1 的差異 | 測試 |
|---|---:|---:|---:|---:|
| qpp-cpu（fp64） | 8 | 2.22e-16 | 2.22e-16 | 6 通過 |
| nvidia（fp32） | 8 | 8.74e-8 | 5.03e-8 | 6 通過 |

兩個 backend 全數通過 1e-5 的數值檢查。每個 backend 包含四筆 train、四筆 holdout；其中四個 holdout 欄位值被 clipping。逐筆保留 1,000 shots counts 與 sampling seed，counts 總數均正確。

## 可重現資料

- CPU：[summary.json](qpp-cpu/summary.json)、[predictions.json](qpp-cpu/predictions.json)。
- GPU：[summary.json](nvidia/summary.json)、[predictions.json](nvidia/predictions.json)。

scaler 使用 train minimum=[10,100]、maximum=[40,200]。同一筆 [25,150] 依正確 scaler 得到 [0,0]；錯誤合併 holdout 後 fit 則得到約 [-0.1111,-0.1765]。反例另存於 summary，不用它準備實驗電路。

basis 與 amplitude 的表示比較僅執行 NumPy 測試；本日的 GPU 結果全部來自 angle encoding，不能當作 amplitude loading 效能或正確性報告。

## 環境與限制

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6、OMP_NUM_THREADS=1；GPU 為 RTX 3060 Laptop 6 GiB，driver 570.211.01。沿用 `.venv`，沒有新增套件。

所有 exact 結果均為無噪聲模擬器資料。GPU 測試與實驗退出時出現 cudaErrorCudartUnloading，退出碼皆為 0；根因尚未確認。本文沒有硬體 QPU、速度比較、模型 accuracy 或量子優勢宣稱。
