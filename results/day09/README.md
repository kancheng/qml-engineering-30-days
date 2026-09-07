# Day 09 PQC 驗證結果

執行日期：2026-09-06。代碼與重跑指令見 [Day 09](../../articles/day09/README.md)。

| Backend | Forward cases | Response cases | 測試 | 最大 exact expectation 誤差 | 最大 fidelity 誤差 |
|---|---:|---:|---|---:|---:|
| [qpp-cpu](qpp-cpu/summary.json) | 24 | 12 | 7／7 | 約 4.44×10⁻¹⁶ | 約 1.11×10⁻¹⁵ |
| [nvidia](nvidia/summary.json) | 24 | 12 | 7／7 | 約 9.71×10⁻⁸ | 約 1.79×10⁻⁷ |

全部 forward 與 response checks 通過；state／expectation 絕對容差為 1e-5。每個 backend 的檔案：

- `pqc_sweep.csv`：features、layers、weight setting、完整 weights、exact／NumPy／sampled output 與 fidelity。
- `raw_results.json`：全部 forward 參數、counts、seed 與 checks。
- `summary.json`：環境、初始化規則、參數個數、完整 response scan、誤差與執行時間。
- `circuit.txt`：預設一層模型的實際電路圖。

PQC 使用 2 qubits、L=1／2、每層 4 weights，readout 為 ZZ。初始化使用零向量與 normal(0,0.2) seeds 42／43；finite-shot 使用 N=1000、seed 42、無 noise。參數反應測試固定 features=[0.25,-0.4]，每次只改一個 weight。

沿用 `.venv` 的 Python 3.12.7、NumPy 2.2.6、CUDA-Q 0.15.1，OMP_NUM_THREADS=1。CPU target 為 fp64，GPU target 為 fp32；GPU 使用沙箱外 RTX 3060，主機詳細資訊見 [Day 06](../day06/environment.json)。本日沒有新增套件。

GPU 實驗與測試結束時仍印出 `cudaErrorCudartUnloading`，退出碼為 0、結果保存成功，根因仍未定位。

這是電路正確性與局部參數反應測試，不是 gradient、training、generalization 或效能 benchmark。程式沒有選出「最佳 weights」，零 weights 與隨機初始值都是對照設定。NumPy 提供 classical correctness reference，不由此宣稱量子優勢。
