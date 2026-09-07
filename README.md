# 從 AI Engineering 到 Quantum Machine Learning：30 天 CUDA-Q 實戰

> Classical ML × CUDA-Q × Hybrid Quantum-Classical AI

這是一個以 AI Engineer 視角進入 Quantum Machine Learning（QML）的 30 天工程系列。正式開賽日為 **2026-09-15**，使用 Python 與 NVIDIA CUDA-Q，逐步完成可執行、可測試、可比較、可重現的 QML 專案。

## 系列原則

- Day 1–2 建立完整概念、數學語言與研究脈絡。
- **Day 3 起每篇都包含程式或可重現實驗**，不連續堆疊純理論。
- CUDA-Q 是主要工程工具；PennyLane、Qiskit 僅在比較或驗證時使用。
- 重要 QML 實驗必須提供 Classical Baseline，不預設 Quantum 一定更快或更準。
- 論文優先連到 DOI／出版商原始頁；preprint 明確標示 arXiv，並和正式出版版本分開。
- 文獻核對規則與已驗證書目集中在 [文獻與資源索引](REFERENCES.md)。

## 設備分工

Day 3 已提供 [Ubuntu 獨立環境與示範步驟](articles/day03/README.md#8-實作只用-numpy-建立-gate-simulator)，以及 [設備紀錄](articles/day03/ENVIRONMENT.md)。使用專案 `.venv` 與固定 NumPy 版本執行。

| 設備 | 主要用途 |
|---|---|
| Windows Surface Pro 7 | 寫作、Git、資料整理、NumPy 與小型 CPU simulation |
| Ubuntu + NVIDIA RTX 3060 筆電 | CUDA-Q、GPU simulation、QML training、noise 與效能 benchmark |

CUDA-Q 官方目前列出 Linux、macOS ARM64，以及透過 WSL2 使用 Windows；GPU simulation 支援包含 Ampere 在內、compute capability 7.5 以上的架構。因此主要 CUDA-Q 環境放在 Ubuntu RTX 3060 筆電，Surface 作為輕量開發端。安裝時仍會鎖定實際 CUDA-Q、driver 與 Python 版本，避免把「目前官方條件」當成永久不變的設定。

## 30 天章節規劃與目錄

狀態說明：✅ 完稿／📝 初稿／🧪 待實作／📌 已規劃。

| 日期 | Day | 章節 | 實作／成果 | 狀態 |
|---|---:|---|---|---|
| 09/15 | 01 | [AI Engineer 為什麼現在要理解 QML？](articles/day01/README.md) | 專案定位、研究問題、文獻入口 | ✅ |
| 09/16 | 02 | [從 Bit 到 Qubit](articles/day02/README.md) | NumPy state vector 與 Born rule | ✅ |
| 09/17 | 03 | [Quantum Gate：量子世界的 Layer？](articles/day03/README.md) | NumPy gates、測試、CSV／JSON 實驗資料 | ✅ |
| 09/18 | 04 | [Superposition、Entanglement、Measurement](articles/day04/README.md) | NumPy／CUDA-Q CPU／RTX 3060 Bell state、Z／X 基底與 shots 實驗 | ✅ |
| 09/19 | 05 | [從 Classical ML Pipeline 看懂 QML Pipeline](articles/day05/README.md) | 完整前向流程、已執行 Notebook、架構圖與三 backend 實驗 | ✅ |
| 09/20 | 06 | [CUDA-Q、CUDA、cuQuantum 有何不同？](articles/day06/README.md) | Ubuntu RTX 3060 環境診斷、CPU／GPU Hello Quantum 與 JSON 報告 | ✅ |
| 09/21 | 07 | [第一個 CUDA-Q Quantum Kernel](articles/day07/README.md) | allocation、參數、迴圈／分支、CPU／GPU 各 36 組驗證 | ✅ |
| 09/22 | 08 | [`sample`、`run`、`observe`](articles/day08/README.md) | 共用 state preparation、回傳契約、finite／exact observe 與 CPU／GPU 比較 | ✅ |
| 09/23 | 09 | [Parameterized Quantum Circuit](articles/day09/README.md) | data／weights 分離、4L 參數 Ansatz、CPU／GPU forward 與參數反應驗證 | ✅ |
| 09/24 | 10 | 第一個 CUDA-Q Optimization Loop | Classical optimizer + quantum circuit | 📌 |
| 09/25 | 11 | Classical Data 怎麼變成 Quantum Data？ | feature 維度與 qubit 成本實驗 | 📌 |
| 09/26 | 12 | [Angle Encoding](articles/day12/README.md) | 2D dataset、角度範圍與旋轉軸比較 | ✅ |
| 09/27 | 13 | [Amplitude Encoding](articles/day13/README.md) | normalization、明確 gate 準備與成本 | ✅ |
| 09/28 | 14 | [Feature Map + Ansatz](articles/day14/README.md) | 可訓練 model skeleton、雙編碼與 optimizer 整合 | ✅ |
| 09/29 | 15 | [第一個 CUDA-Q Quantum Classifier](articles/day15/README.md) | XOR 分類、checkpoint、curve、boundary | ✅ |
| 09/30 | 16 | [QNN 到底是不是 Neural Network？](articles/day16/README.md) | QNN／MLP 輸出、參數與非線性比較 | ✅ |
| 10/01 | 17 | [QML 怎麼 Backprop？](articles/day17/README.md) | parameter-shift、finite difference、chain rule | ✅ |
| 10/02 | 18 | [Classical ML vs QML](articles/day18/README.md) | Logistic-link、MLP、VQC 固定協定 benchmark | ✅ |
| 10/03 | 19 | [Hybrid Neural Network](articles/day19/README.md) | Classical＋Quantum＋Classical、完整 chain rule | ✅ |
| 10/04 | 20 | [Iris：Classical vs Quantum vs Hybrid](articles/day20/README.md) | Binary Iris、PCA、metrics／成本／穩定性報告 | ✅ |
| 10/05 | 21 | [Qubit 不夠、Feature 太多怎麼辦？](articles/day21/README.md) | PCA、train-only selection、learned bottleneck 與四維 baseline | ✅ |
| 10/06 | 22 | [Data Re-uploading](articles/day22/README.md) | 配對 schedule、四維分段輸入、電路成本與 CPU／GPU 驗證 | ✅ |
| 10/07 | 23 | [Quantum Kernel](articles/day23/README.md) | Fidelity／RBF／linear、kernel ridge、完整 CPU／GPU 矩陣驗證 | ✅ |
| 10/08 | 24 | [Barren Plateau](articles/day24/README.md) | qubits／depth／initialization／cost locality、解析對照與梯度驗證 | ✅ |
| 10/09 | 25 | [第二個真實資料集：Wine](articles/day25/README.md) | Classical、VQC、Hybrid與13維baseline、CPU／GPU驗證 | ✅ |
| 10/10 | 26 | [沒有完美 Qubit：Quantum Noise](articles/day26/README.md) | Kraus／density matrix、CPU／GPU抽樣與Wine末端noise | ✅ |
| 10/11 | 27 | [CPU vs GPU Quantum Simulation](articles/day27/README.md) | 依序量測、fp64配對、first／warm latency與輸出核對 | ✅ |
| 10/12 | 28 | 單 GPU 到 Multi-GPU | 可用設備實測或 reproducible scaling model | 📌 |
| 10/13 | 29 | Simulator → QPU | backend、shots、queue、hardware noise | 📌 |
| 10/14 | 30 | 我還相信 Quantum Machine Learning 嗎？ | 證據導向總結與研究 Roadmap | 📌 |

## 六個 Milestone

| 時間 | Milestone | 主要成果 |
|---|---|---|
| Day 05 | [Quantum Fundamentals ✅](articles/day05/README.md) | [Notebook](notebooks/day05_quantum_fundamentals.ipynb)、[架構圖](figures/day05_pipeline.svg)、完整 circuit demo |
| [Day 10](articles/day10/README.md) | CUDA-Q Fundamentals | ✅ MSE、座標搜尋、CPU／GPU hybrid optimization loop |
| [Day 15](articles/day15/README.md) | First QML Model ✅ | XOR Classifier、Training Curve、Decision Boundary |
| [Day 20](articles/day20/README.md) | Hybrid QML ✅ | Iris binary：MLP／VQC／Hybrid report |
| [Day 25](articles/day25/README.md) | Wine Binary Dataset | ✅ 第二份可重現 Classical／VQC／Hybrid benchmark |
| Day 30 | Final Project | Noise、GPU、QPU 與完整工程總結 |

Day 11 已完成：[Classical Data 怎麼變成 Quantum Data？](articles/day11/README.md)，包含資料縮放、編碼比較與 CPU／GPU 可重現實驗。

## Repository 目錄

```text
.
├── README.md             # 章節規劃與文章目錄
├── ROADMAP.md            # 每階段目標與交付物
├── REFERENCES.md         # 已核對文獻、官方資源與引用規則
├── articles/             # 每日文章
├── notebooks/            # 可重現教學與實驗
├── src/                  # 共用 Python 程式
├── datasets/             # 資料與取得方式
├── experiments/          # 實驗設定
├── benchmarks/           # Classical / Quantum 比較
├── figures/              # 文章圖表
└── results/              # 結果與執行環境紀錄
```

目錄隨每日成果建立，不預先加入大量空資料夾。

## 主要官方入口

- [NVIDIA CUDA-Q Quick Start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)
- [CUDA-Q by Example](https://nvidia.github.io/cuda-quantum/latest/using/examples/examples.html)
- [CUDA-Q Local Installation](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html)
- [NVIDIA cuQuantum Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/index.html)
