# Day 08 執行介面比較結果

執行日期：2026-09-06。完整範例與重跑指令見 [Day 08](../../articles/day08/README.md)。

| Backend | 設定數 | 檢查 | Precision | 最大 exact Z0 絕對誤差 |
|---|---:|---|---|---:|
| [qpp-cpu](qpp-cpu/summary.json) | 16 | 全數通過 | fp64 | 約 1.61×10⁻¹⁶ |
| [nvidia](nvidia/summary.json) | 16 | 全數通過 | fp32 | 約 1.89×10⁻¹⁵ |

兩個 backend 各通過 5 個測試，涵蓋兩位元整數對應、invalid inputs、deterministic endpoints、非平凡角度的 Z0／XX 以及各 API 內的固定 seed 重跑。

每個 backend 目錄包含：

- `api_comparison.csv`：16 組 Z0 reference 與 sample／run／observe finite／observe exact 輸出。
- `raw_results.json`：原始 counts、全部 run 回傳值、run histogram、XX exact 值與每個 API 的時間／checks。
- `summary.json`：Python／NumPy／CUDA-Q／precision 與實驗設定。

CPU 的 theta=π/3、32 shots、seed=42 範例：sample counts 為 `00:25, 11:7`；run histogram 相同；三種 finite-shot Z0 估計皆為 0.5625，exact Z0 約為 0.5。這是此版本與設定的觀察，程式不要求不同 API 的抽樣序列相同。

每組的 sample、run、finite-shot observe 各有自己的 N shots，且各自在呼叫前重設 seed；不能將它們當成三份獨立亂數實驗，也不能將它們說成同一批 shots 的三種顯示。XX 僅作 exact observable 示範，finite-shot 成本比較限定單一 Z0 term。

本次使用既有 `.venv`：Python 3.12.7、NumPy 2.2.6、CUDA-Q 0.15.1、OMP_NUM_THREADS=1，沒有新增依賴。GPU 在沙箱外的 RTX 3060 執行，硬體紀錄見 [Day 06](../day06/environment.json)。GPU 實驗與測試程序結束時仍有 `cudaErrorCudartUnloading`，退出碼為 0，結果已保存；根因未定位。

Target precision 不等於每個中間結果或 observable 累加都使用同一種 dtype，不能僅憑這幾個誤差數字作一般精度結論。本日不做效能 benchmark 或模型訓練比較。
