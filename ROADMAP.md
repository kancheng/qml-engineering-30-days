# 30 天 Roadmap

正式開賽日：**2026-09-15**。每一天延續同一個 QML Engineering Project，而不是 30 個互不相關的範例。

## Part 1｜Day 01–05：最小理論，立即實作

1. AI Engineer 為什麼現在要理解 QML？
2. 從 Bit 到 Qubit：工程師需要懂多少量子力學？
3. Quantum Gate：量子世界的 Layer？——用 NumPy 實作 gate matrix 與 state transformation
4. [Superposition、Entanglement、Measurement](articles/day04/README.md)——已完成 NumPy／CUDA-Q CPU／RTX 3060 Bell state、Z／X 基底與 shots 實驗
5. [從 Classical ML Pipeline 看懂 QML Pipeline](articles/day05/README.md)——完整前向流程、Notebook、架構圖與 NumPy／CUDA-Q CPU／GPU 驗證

Day 1–2 完成概念、數學語言與文獻入口；**Day 3 起每日必須包含可執行程式或可重現實驗**。

成果：Quantum Fundamentals Notebook、NumPy gate simulator、CUDA-Q Bell state、架構圖與第一個完整 Quantum Circuit Demo。

Day 04 已先完成獨立環境中的 CUDA-Q 安裝與 CPU／GPU 基本驗證，三個 backend 各保留 90 筆實驗紀錄。Day 06 再系統整理 CUDA-Q／CUDA／cuQuantum、環境設定與版本條件。

Day 05 已完成第一階段交付：[Quantum Fundamentals Notebook](notebooks/day05_quantum_fundamentals.ipynb)、[架構圖](figures/day05_pipeline.svg) 與完整 circuit demo。每個 backend 保存 27 筆輸入到量測輸出的驗證結果；此階段沒有模型訓練，optimizer 留待 Day 09–10。

## Part 2｜Day 06–10：正式進入 NVIDIA CUDA-Q

6. [CUDA-Q、CUDA、cuQuantum 有何不同？](articles/day06/README.md)——已完成環境診斷、CPU／GPU Hello Quantum 與主機／沙箱結果紀錄
7. [建立第一個 CUDA-Q Quantum Kernel](articles/day07/README.md)——已完成 qubit／qvector、型別化參數、迴圈、條件分支與 CPU／GPU 驗證
8. [`sample`、`run`、`observe` 到底差在哪？](articles/day08/README.md)——已完成 counts、逐次回傳值、finite／exact observe，CPU／GPU 各 16 組比較
9. [Parameterized Quantum Circuit：讓量子電路開始可以學習](articles/day09/README.md)——已完成 data／weights 分離、4L 參數 Ansatz 與 CPU／GPU 驗證；尚未訓練
10. [第一個 CUDA-Q Optimization Loop](articles/day10/README.md)——已完成 MSE、座標搜尋與可重現訓練紀錄

成果：Parameterized Circuit、Variational Optimization Demo、Hybrid Loop。

Day 06 已完成工具分工與環境驗證，主機 CPU／RTX 3060 smoke tests 通過，並保留沙箱下 GPU 不可見的失敗報告。Day 07 已完成 kernel 深入教學，CPU／GPU 各保存 36 組設定與 6 個測試的驗證成果。Day 08 已完成執行介面比較，CPU／GPU 各 16 組實驗與 5 個測試通過。Day 09 已完成 PQC，每個 backend 有 24 組 forward、12 組參數反應與 7 個測試通過。Day 10 已完成 loss、座標搜尋與 Hybrid Optimization Loop，保留 CPU／GPU、兩個初始化 seeds 的訓練與 holdout 結果。

## Part 3｜Day 11–15：把 Machine Learning Data 放進 Quantum Circuit

11. [Classical Data 怎麼變成 Quantum Data？](articles/day11/README.md)——已完成 train-only preprocessing、編碼比較、資訊碰撞與 CPU／GPU 驗證
12. [Angle Encoding](articles/day12/README.md)——已完成三種角度範圍、四種旋轉準備方式與 CPU／GPU 各 96 組設定驗證
13. [Amplitude Encoding](articles/day13/README.md)——已完成正規化、simulator loading、RY／CNOT state preparation 與 CPU／GPU 各 12 組驗證
14. [Feature Map + Ansatz](articles/day14/README.md)——已完成可替換編碼、共用 Ansatz／ZZ／batch API 與 CPU／GPU 短訓練驗證
15. [第一個 CUDA-Q Quantum Classifier](articles/day15/README.md)——已完成 XOR train／validation／test、checkpoint、curve 與 boundary

成果：可訓練的 Quantum Classifier、Training Curve、Decision Boundary。

## Part 4｜Day 16–20：Quantum Machine Learning Engineering

16. [QNN 到底是不是 Neural Network？](articles/day16/README.md)——已完成 QNN／MLP 前向、參數角色與非線性來源比較，CPU／GPU 驗證
17. [QML 怎麼 Backprop？Gradient 從哪裡來？](articles/day17/README.md)——已完成 parameter-shift／finite difference／matrix derivative、loss chain rule 與 CPU／GPU 驗證
18. [Classical ML vs QML：第一次公平 Benchmark](articles/day18/README.md)——已完成同 split／Brier／評估預算的 logistic-link、MLP、VQC 比較與成本紀錄
19. [Hybrid Neural Network：Classical Layer + Quantum Layer](articles/day19/README.md)——已完成 12 參數 classical／quantum／classical 模型、全梯度與聯合更新驗證
20. Iris：Classical vs Quantum vs Hybrid

成果：Classical Baseline、VQC、Hybrid QNN 與第一份 Benchmark Report。

## Part 5｜Day 21–25：不只 Toy Example

21. Qubit 不夠、Feature 太多怎麼辦？
22. Data Re-uploading
23. Quantum Kernel：QML 不只有 QNN
24. Barren Plateau：為什麼 QNN 突然學不動？
25. 第二個真實資料集 Benchmark

成果：降維、Quantum Kernel、Trainability 與真實資料集比較。

## Part 6｜Day 26–30：GPU、Noise、QPU 與真正的 QML Engineering

26. 用 CUDA-Q 模擬 Quantum Noise
27. CPU vs GPU Quantum Simulation
28. 從單 GPU 到 Multi-GPU
29. Simulator → QPU
30. 寫了 30 天 CUDA-Q 與 QML，我還相信 QML 嗎？

成果：Noise-aware 實驗、模擬效能比較、硬體落差分析與完整專案總結。

## 實驗紀錄原則

每個實驗盡量記錄 Dataset、Model、Qubit 數、Circuit Depth、Encoding、Ansatz、Optimizer、Learning Rate、Iterations、Shots、Noise Model、Backend、Execution Time、Metrics 與 Random Seed。所有重要 QML 實驗至少提供一個合理的 Classical Baseline。

## 文獻原則

Survey 用於建立研究地圖，具體方法與結論回到原始論文；軟體與硬體條件使用官方文件。所有書目先依 [文獻與資源索引](REFERENCES.md) 的規則核對題名、作者、出版資訊、DOI 與版本狀態。
