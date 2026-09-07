# Day 06 環境診斷結果

執行日期：2026-09-06。完整說明與重跑指令見 [Day 06](../../articles/day06/README.md)。

| 報告 | 執行位置 | CPU | GPU | 診斷退出碼 |
|---|---|---|---|---:|
| [environment.json](environment.json) | 沙箱外 Ubuntu 主機 | PASS | PASS | 0 |
| [sandbox_environment.json](sandbox_environment.json) | 受限工具沙箱 | PASS | FAIL：GPU 不可見 | 1 |

主機 GPU inventory：RTX 3060 Laptop GPU、6144 MiB、compute capability 8.6、driver 570.211.01。Python 3.12.7、NumPy 2.2.6、CUDA-Q 0.15.1；完整套件清單在每份報告的 `packages`。

CPU／GPU 各通過五項 smoke checks。GPU 的 Bell fidelity 約 0.999999966，CPU 約 1.0；counts 都為 `00:493`、`11:507`。GPU 子程序 stderr 保留 `cudaErrorCudartUnloading`，其退出碼為 0。總 `passed=true` 代表指定檢查通過，不代表 stderr 空白或所有 CUDA-Q 功能均已驗證。

`loaded_gpu_libraries` 僅記錄 Linux loader 的 library 名稱；CPU 程序也可能載入這些 libraries。`tools` 的狀態與 backend 的執行狀態分開保存。時間包含 initialization／JIT，本日沒有效能比較結論。

報告中的暫存檔路徑是該次子程序呼叫的紀錄，完成後已清理。要重跑請使用文章中的 `check_environment.py`，不要直接複製帶有已刪除暫存目錄的內部 command。
