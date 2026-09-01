# 30 天 Roadmap

正式開賽日：**2026-09-15**。每一天延續同一個 QML Engineering Project，而不是 30 個互不相關的範例。

## Part 1｜Day 01–05：最小理論，立即實作

1. AI Engineer 為什麼現在要理解 QML？
2. 從 Bit 到 Qubit：工程師需要懂多少量子力學？
3. Quantum Gate：量子世界的 Layer？——用 NumPy 實作 gate matrix 與 state transformation
4. Superposition、Entanglement、Measurement——提前用 CUDA-Q 建立 Bell state 與 shots 實驗
5. 從 Classical ML Pipeline 看懂 QML Pipeline——整理 Notebook、架構圖與完整 circuit demo

Day 1–2 完成概念、數學語言與文獻入口；**Day 3 起每日必須包含可執行程式或可重現實驗**。

成果：Quantum Fundamentals Notebook、NumPy gate simulator、CUDA-Q Bell state、架構圖與第一個完整 Quantum Circuit Demo。

## Part 2｜Day 06–10：正式進入 NVIDIA CUDA-Q

6. CUDA-Q 是什麼？GPU 跟 Quantum Computing 為什麼放在一起？
7. 建立第一個 CUDA-Q Quantum Kernel
8. `sample`、`run`、`observe` 到底差在哪？
9. Parameterized Quantum Circuit：讓量子電路開始可以學習
10. 第一個 CUDA-Q Optimization Loop

成果：Parameterized Circuit、Variational Optimization Demo、Hybrid Loop。

## Part 3｜Day 11–15：把 Machine Learning Data 放進 Quantum Circuit

11. Classical Data 怎麼變成 Quantum Data？
12. Angle Encoding
13. Amplitude Encoding
14. Feature Map + Ansatz
15. 第一個 CUDA-Q Quantum Classifier

成果：可訓練的 Quantum Classifier、Training Curve、Decision Boundary。

## Part 4｜Day 16–20：Quantum Machine Learning Engineering

16. QNN 到底是不是 Neural Network？
17. QML 怎麼 Backprop？Gradient 從哪裡來？
18. Classical ML vs QML：第一次公平 Benchmark
19. Hybrid Neural Network：Classical Layer + Quantum Layer
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
