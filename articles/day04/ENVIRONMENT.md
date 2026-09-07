# Day 04 環境與執行結果

驗證日期：2026-09-06。沿用專案 `.venv`，未將套件安裝進 Conda base。

## 實際環境

| 項目 | 實測／安裝值 |
|---|---|
| OS | Ubuntu 22.04.5 LTS，Linux 6.8.0-138-generic，x86_64 |
| Python | 3.12.7 |
| NumPy | 2.2.6 |
| CUDA-Q distribution | cuda-quantum-cu12 0.15.1 |
| GPU | NVIDIA GeForce RTX 3060 筆電 GPU，6144 MiB |
| NVIDIA driver | 570.211.01 |
| nvidia-smi 顯示 CUDA Version | 12.8 |
| 系統 nvcc | release 12.8，V12.8.93 |
| venv NVIDIA CUDA runtime 套件 | nvidia-cuda-runtime-cu12 12.9.79 |

OS／nvcc 由本機指令核對，GPU／driver 在沙箱外以 `nvidia-smi` 核對。系統工具鏈與虛擬環境安裝的 runtime 套件分開記錄；版本表不代表所有 CUDA-Q 功能皆已驗證。完整 Python 依賴固定於 [requirements-day04-lock.txt](../../requirements-day04-lock.txt)。

## 已完成的檢查

| 檢查 | 結果 |
|---|---|
| Day 03 regression tests | 7／7 通過 |
| Day 04 NumPy + CUDA-Q CPU tests | 12／12 通過 |
| Day 04 CUDA-Q GPU tests | 5／5 通過 |
| pip check | 通過 |
| NumPy experiment | 90 筆 |
| CUDA-Q qpp-cpu experiment | 90 筆 |
| CUDA-Q nvidia experiment | 90 筆 |

GPU 測試包含與 NumPy Bell state 的 fidelity 核對、明確量測位元順序、兩個基底的 Bell counts、product state 與 mixture 抽樣。

## 本次結果

Bell Z 基底在五個 seeds（42–46）的平均 total variation distance：

| Backend | 100 shots | 1,000 shots | 10,000 shots |
|---|---:|---:|---:|
| numpy | 0.02800 | 0.01540 | 0.00542 |
| qpp-cpu | 0.02800 | 0.00900 | 0.00434 |
| nvidia | 0.02800 | 0.00900 | 0.00434 |

在本次 10,000 shots 設定，CUDA-Q CPU／GPU 的 Bell Z 與 X 平均 correlation 皆為 1；mixture 的 Z 為 1、X 約 0.0024，符合本日理論對照。跨 backend 的 counts 相同不是必要條件；不同準備也會重用同一組 seeds，因此這些對照不能當成彼此獨立的統計樣本。

CSV 保留每次 counts 與時間，JSON 保留五個 seeds 的聚合指標及環境。執行時間包含 JIT／初始化，且未控制負載與暖機，不能當作 CPU／GPU 效能結論。

## 執行時觀察

這次工具沙箱中 `nvidia-smi` 無法連接 driver，CUDA-Q `nvidia` 也回報沒有可見的 CUDA 裝置；在獲准的沙箱外執行則成功。因此 GPU 實驗與測試使用沙箱外執行，不把沙箱限制判為主機 driver 故障。

GPU 示範、實驗與測試在程序結束時印出 `cudaErrorCudartUnloading`。三者退出碼皆為 0，實驗檔案已寫入，測試全部通過。尚未定位本環境的根因，也未以隱藏 stderr 或強制退出來消除訊息。NVIDIA 的 [runtime 錯誤定義](https://nvidia.github.io/cuda-python/cuda-bindings/13.3.1/module/runtime.html) 將此名稱描述為程序關閉期間 driver 卸載後的 Runtime API 呼叫；這是錯誤名稱的說明，不是本次根因已獲證實。

使用者在一般 Ubuntu 終端執行的指令見 [Day 04 文章](README.md)。若 GPU 執行失敗，可明確選擇 `--backend qpp-cpu`；程式不會靜默改用 CPU 再把結果標成 GPU。
