# Day 16 實驗紀錄

[教學與重跑方式](../../articles/day16/README.md)。本日比較未訓練的兩個 qubit PQC（4 參數）與 2→2→1 MLP（9 參數），沒有 labels、loss、accuracy 或模型排名。

| QNN backend | Forward pairs | 參數反應 | 最大 forward probability reference 誤差 | 測試 |
|---|---:|---:|---:|---:|
| qpp-cpu（fp64） | 42 | 39 | 3.33e-16 | 6 通過 |
| nvidia（fp32） | 42 | 39 | 6.68e-8 | 6 通過 |

每個 backend 兩個 seeds × 21 個輸入，固定 scaled x1=0.25；39 組反應為 seed42 的 QNN 4×3 與 MLP 9×3。QNN 所有 forward／response reference 檢查均通過 1e-5；MLP 的 matrix 計算由 scalar unit tests 核對，在兩個 backend 實驗中都由 NumPy CPU 執行。

## 解析核對

RY 矩陣線性組合殘差為 0；cosθ 在 θ=π/4 與 θ=0、π/2 端點輸出平均的差異約為 0.207107。這分別檢查 state 線性和角度輸出非線性，沒有把量測解釋成 MLP activation。

單筆 seed42、features=[0.25,-0.4] 示範：QNN p1≈0.0998343，MLP p1≈0.502383。未訓練模型的不同輸出沒有好壞結論。

## 檔案

- CPU：[summary](qpp-cpu/summary.json)、[forward](qpp-cpu/predictions.json)、[參數反應](qpp-cpu/parameter_response.json)、[圖表](qpp-cpu/forward_comparison.png)。
- GPU：[summary](nvidia/summary.json)、[forward](nvidia/predictions.json)、[參數反應](nvidia/parameter_response.json)、[圖表](nvidia/forward_comparison.png)。

所有設定保留 features、weights、seed，圖表直接使用保存的結果。未重訓或挑選 Day 15 test 上的模型。

## 環境與限制

Ubuntu 22.04.5、Python 3.12.7、CUDA-Q 0.15.1、NumPy 2.2.6、Matplotlib 3.10.6，OMP_NUM_THREADS=1；RTX 3060 Laptop 6 GiB、driver 570.211.01。沿用 .venv，本日無新增依賴。UTC 2026-09-06 20:xx 對應台灣 2026-09-07 04:xx。

GPU 測試與實驗退出時出現 cudaErrorCudartUnloading，退出碼為 0，數值檢查通過，根因尚未確認。本日只有無噪聲 exact simulator forward，沒有 finite shots、梯度、QPU、速度或量子優勢宣稱。
