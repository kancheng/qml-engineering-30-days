# Day 07 Kernel 驗證結果

執行日期：2026-09-06。文章與指令見 [Day 07](../../articles/day07/README.md)。

| Backend | 組合數 | 通過 | Target precision | 最大 fidelity error | 最大 sampled TVD |
|---|---:|---|---|---:|---:|
| [qpp-cpu](qpp-cpu/summary.json) | 36 | 全數通過 | fp64 | 0 | 約 0.007 |
| [nvidia](nvidia/summary.json) | 36 | 全數通過 | fp32 | 約 3.422854×10⁻⁸ | 約 0.007 |

各 backend 另通過同一套 6 個測試，涵蓋 single qubit、n=1 空迴圈、register 大小、分支、非對稱 bit ordering、seed 與 host 輸入驗證。Fidelity／norm／probability 檢查使用絕對容差 1e-5。

掃描 n=2、3、4，theta=0、π/2、π，entangle 與 flip_last 各兩種值；每組 1,000 shots、seed 42、無 noise。不同設定重用 seed 是為了追溯，不作為獨立多 seed 統計實驗。

每個 backend 目錄包含：

- `kernel_sweep.csv`：36 筆參數、counts、fidelity、norm、TVD、checks 與時間。
- `summary.json`：環境、precision 與聚合檢查。
- `circuit.txt`：n=3、theta=π/2、entangle=True、flip_last=False 的實際電路文字圖。

本次使用既有 `.venv`（Python 3.12.7、NumPy 2.2.6、CUDA-Q 0.15.1），OMP_NUM_THREADS=1。GPU 以沙箱外 RTX 3060 執行；主機詳細資訊見 [Day 06 環境報告](../day06/environment.json)。未新增依賴、未修改 driver。

GPU 實驗與測試結束時仍印出 `cudaErrorCudartUnloading`，退出碼為 0，結果已保存且檢查通過。此現象延續 Day 06 紀錄，根因仍未定位；不以隱藏輸出消除訊息。

這是 kernel correctness sweep，未執行訓練或效能 benchmark。時間包含編譯／初始化；本次不同 target precision 也已明確記錄，不能直接以數字推論速度或模型優勢。
