# Day 05 環境、驗證與限制

執行日期：2026-09-06。沿用 Day 04 的 Ubuntu／RTX 3060 與專案 `.venv`，增加 Notebook 依賴，沒有修改系統 driver 或 Conda base。

| 項目 | 值 |
|---|---|
| Ubuntu | 22.04.5 LTS，x86_64 |
| Python／NumPy | 3.12.7／2.2.6 |
| CUDA-Q | cuda-quantum-cu12 0.15.1 |
| GPU／driver | RTX 3060 6144 MiB／570.211.01（Day 04 已核對） |
| Notebook | nbclient 0.11.0、nbformat 5.11.1、ipykernel 7.3.0 |
| 完整 Python 依賴 | [requirements-day05-lock.txt](../../requirements-day05-lock.txt) |

## 驗證成果

- NumPy、CUDA-Q qpp-cpu、CUDA-Q nvidia 各產生 27 筆結果，共 81 筆。
- Day 05 NumPy／CPU：7 個測試通過。
- Day 05 GPU：3 個整合測試通過。
- Notebook：10 個 code cells 全部成功，輸出已保存在 `.ipynb`。
- `pip check` 通過。
- Day 03／Day 04 回歸測試：7／7 與 12／12 通過。

Notebook runner 使用目前 `.venv` interpreter 的暫時 kernel，不做全域 kernel 註冊；執行完成會關閉 kernel。未設定 `OMP_NUM_THREADS` 時，runner 預設為 1，以減少兩 Qubit 示範的執行緒負擔。CLI 實驗沿用呼叫端的執行緒環境，未用它和 Notebook 計時做比較。

## 結果核對

| Backend | Sampled MSE | 每筆 shots | Repeat seeds |
|---|---:|---:|---|
| numpy | 0.000471168897 | 1,000 | 42、43、44 |
| qpp-cpu | 0.000364608162 | 1,000 | 42、43、44 |
| nvidia | 0.000364608162 | 1,000 | 42、43、44 |

每組 seed 涵蓋 9 個固定 inputs，各 input 的實際抽樣 seed 由 `SeedSequence` 衍生並寫入 CSV。解析預期 sampling MSE 為 0.000444444444；constant-zero baseline 為 0.555555555556；解析 classical reference 的 MSE 依定義為 0。

Notebook 以 qpp-cpu 重跑同一套程式，不依賴事先存在的結果檔。Notebook 內的檔案保存範例寫入暫存目錄，正式實驗檔由 CLI 保存。

## 實際執行觀察

GPU 執行使用沙箱外的 RTX 3060。完整 GPU pipeline 程序結束時再次印出 `cudaErrorCudartUnloading`，退出碼為 0 且結果正常寫入；本次 GPU 測試也通過。Day 04 的相關紀錄與尚未定位的根因見 [ENVIRONMENT.md](../day04/ENVIRONMENT.md)。

Notebook 首次在受限工具環境中未完成啟動，已中止該次執行；改用非同步 kernel manager、明確清理 kernel，並在允許本機 kernel 通訊的沙箱外完成驗證。此紀錄不將首次停滯的原因歸因於已確認的特定 bug。

本日是固定電路的 forward-pass verification：沒有 trainable parameters、optimizer、資料切分或泛化測試。執行時間包含初次編譯及初始化，沒有建立效能 benchmark。CNOT 對本例 `⟨Z0⟩` 輸出不是必要操作；沒有宣稱此任務需要糾纏或具有量子優勢。
