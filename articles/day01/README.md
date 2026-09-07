# Day 01｜AI Engineer 為什麼現在要理解 Quantum Machine Learning？

## 本章摘要｜初學者學習筆記

### 中文

這一章目標在於建立學習 Quantum Machine Learning（QML，量子機器學習）的方向，釐清兩個核心問題：**量子計算可以放進機器學習流程的哪個位置？需要哪些證據，才能判斷量子計算是否有幫助？** 本系列從一般數值資料出發，說明如何把資料轉成量子電路能處理的形式，經過電路運算，再透過量測取得可供預測與計算誤差的數值；一般電腦則負責資料處理與參數調整。理解這個分工，有助於掌握 qubit（量子位元）、encoding（資料編碼）與 measurement（量測）在流程中的用途。本章重點是：QML 仍然需要清楚的任務、資料與評估方法，而「程式能跑」「預測較準」「完成任務的成本較低」是需要分別驗證的事。讀完本章，應能描述這條基本流程，並說明為什麼每次實驗都要和解決同一任務的傳統機器學習方法比較，為後續技術學習與結果判讀建立基礎。

### English

This chapter aims to establish a foundation for learning Quantum Machine Learning (QML) by addressing two core questions: **Where can quantum computation fit into a machine learning workflow, and what evidence is needed to assess its value?** The series starts with ordinary numerical data and explains how data is encoded for a quantum circuit, processed by the circuit, and measured to obtain numerical outputs for predictions and error calculations. A classical computer handles data processing and parameter updates. Understanding this division of work clarifies the roles of qubits, encoding, and measurement. The key point is that QML still requires a well-defined task, data, and an evaluation method. A program running successfully, making better predictions, and solving a task at lower cost are separate claims that require separate evidence. The learning goal is to describe this basic workflow and explain why each experiment needs a comparison with a classical machine learning method solving the same task, establishing a foundation for later technical concepts and the interpretation of results.

---

從 Machine Learning、Deep Learning 或 AI Engineering 背景初次接觸 Quantum Machine Learning（QML）時，很容易在「它是下一代 AI」與「硬體還不成熟，現在研究沒有意義」之間擺盪。

兩個結論都下得太快。第一天先不談量子閘，而是回答更基本的問題：**AI Engineer 為什麼值得現在開始理解 QML？**

## 1. 今天要解決什麼問題？

今天要釐清 QML 的定義、它和 Quantum Computing、Quantum AI 的關係、現階段能做與不能做的事，以及這 30 天要走完的工程路線。

```text
AI Engineering → Quantum 基礎 → CUDA-Q → QML
               → Hybrid AI → GPU / Noise / Benchmark
```

目標不是記住 30 個名詞，而是最後能寫 Quantum Program、建立可訓練電路、把 Classical Data 編碼進去，並以 Classical Baseline 公平比較結果。

這也不是一次性的教學專案。後續希望發展成可投稿的研究工作，因此從 Day 1 就區分三種層次：

1. **Learning claim**：為了教學建立的直覺與簡化說法。
2. **Engineering result**：能在指定環境、資料與 seed 下重現的實驗結果。
3. **Research claim**：必須由原始論文或足夠嚴格的實驗支持，才可能寫進投稿內容。

## 2. 從熟悉的 Machine Learning 開始

Classical ML 的典型流程是：

```text
Dataset → Feature Engineering → Model → Prediction → Loss → Optimizer
```

它可以抽象成「轉換資料表示，再以可調參數模型找出完成任務的函數」。QML 沒有離開這件事，主要差異是模型的一部分改由 Quantum Circuit 負責：

```text
Classical Data → Encoding → Quantum Circuit → Measurement
               → Classical Output → Loss / Optimizer
```

資料通常仍是一般數值；經 Angle Encoding、Amplitude Encoding 或 Feature Map 寫入 Quantum State。Parameterized Quantum Circuit 的參數由 Classical Optimizer 更新，Measurement 再將結果帶回 Classical World。

所以近期最實際的 QML 工作流通常是 **Hybrid Quantum-Classical Computing**，而不是所有步驟都在量子電腦上執行。

## 3. Quantum Computing、QML 與 Quantum AI

- Quantum Computing 是底層計算領域，包含 Qubit、Quantum Gate、Circuit、Measurement、Noise 與 QPU。
- Quantum Machine Learning 聚焦於利用 Quantum Computing 建立或協助 ML 演算法，例如 Variational Quantum Classifier、Quantum Kernel 與 Hybrid QNN。
- Quantum AI 是較寬鬆的上位說法，還可能涵蓋 Quantum Optimization、Generative Model 與 Reinforcement Learning。

本系列刻意縮小範圍：只補足 AI Engineer 進入 QML 真正需要的 Quantum Computing 基礎。

## 4. Classical ML 和 QML 差在哪？

Classical Neural Network 常見的計算是 `xW + b`、activation 與 layer composition。QML 則可能把輸入轉成 Quantum State，經過 Parameterized Circuit，再由 Measurement 產生預測所需的 Classical Value。

這不是把 `nn.Linear(...)` 換成想像中的 `QuantumLinear(...)`。改變 computation model 後，也引入新的限制：

- Qubit 數量與硬體連接方式
- Circuit Depth 與 State Preparation 成本
- Measurement、Shots 與 Noise
- Gradient Evaluation 成本
- Simulator 與真正 QPU 的落差

## 5. 現階段可以做什麼？

現階段可建立並研究的模型與問題包括：

- Variational Quantum Classifier
- Quantum Kernel Method
- Classical Feature Extractor + Quantum Layer + Classical Head
- Encoding、Ansatz、Gradient 與 Noise 對訓練的影響

這些都是具體而可重現的工程實驗，但「能執行」不等於「已有 Quantum Advantage」。

2017 年的 Nature review 已把 QML 描述為結合 quantum computing 與 learning task 的研究領域，同時明確指出硬體與軟體仍有重大挑戰。[R1] 2022 年的 review 則把焦點進一步放到 trainability、資料類型與 quantum advantage 的成立條件。[R2] 因此本系列選擇「先實作、再 benchmark、最後才討論 advantage」，不是先假定結論。

## 6. 現階段不能預設什麼？

寫出 Quantum Circuit 不等於得到 Quantum Speedup；Simulator 成功不代表 QPU 上結果相同；GPU 加速 Quantum Simulation 也不等於 Quantum Computer 勝過 Classical Computer。

即使某一次實驗中 VQC accuracy 高於 MLP，仍需檢查 seed、資料切分、參數量、訓練時間、variance、qubit 數與 circuit depth。沒有這些條件，單一 accuracy 數字不足以支持 Quantum Advantage。

## 7. 為什麼 AI Engineer 現在值得理解？

AI Engineer 已熟悉 Dataset、Tensor、Model、Optimizer、GPU 與 Benchmark。QML 在旁邊加入 Qubit、State、Gate、Circuit、Measurement、QPU 與 Noise，最後在 Parameterized Circuit、Classical Optimizer 和 Hybrid Workflow 交會。

正因工具與硬體還未成熟到只剩 API，現在建立正確的 mental model，能避免把模擬器效能、模型精度與 Quantum Advantage 混為一談。

## 8. 為什麼選 CUDA-Q？

CUDA-Q 是主要工程工具，負責 Quantum Kernel、Sampling、Expectation Value、Variational Algorithm、Noise Simulation、GPU Simulation 與 QPU Backend；PennyLane、Qiskit 只在比較生態或驗證概念時輔助。

系列主題是 **QML Engineering**，不是 CUDA-Q API 大全。

### 兩台設備如何分工？

```text
Surface Pro 7
├── 寫作、Git、文獻整理
├── NumPy 與小型 CPU simulation
└── 遠端連接 Ubuntu 主機

Ubuntu + RTX 3060
├── CUDA-Q 主環境
├── GPU quantum simulation
├── QML training / benchmark
└── noise 與效能實驗
```

CUDA-Q 官方安裝頁目前將 Linux 列為主要平台，也列出 Windows via WSL2；GPU simulation 支援 Ampere 架構與 compute capability 7.5 以上。[D2] RTX 3060 屬於 Ampere，因此硬體方向合理，但 Day 6 仍會用實際的 `nvidia-smi`、Python 與 CUDA-Q 版本驗證，而不是只靠型號推定環境一定可用。

## 9. 30 天的六個階段

1. Day 01–05：最小量子計算基礎
2. Day 06–10：CUDA-Q 與第一個 Hybrid Optimization Loop
3. Day 11–15：Data Encoding 與 Quantum Classifier
4. Day 16–20：Gradient、QNN、Hybrid Model 與 Benchmark
5. Day 21–25：降維、Quantum Kernel、Barren Plateau 與真實資料
6. Day 26–30：Noise、GPU Simulation、QPU 與總結

完整題目與成果見 [30 天 Roadmap](../../ROADMAP.md)。

## 10. 今天的工程成果

Day 1 完成 Repository 首頁、Roadmap 與文章目錄，確立三條規則：

1. 30 天共同演化成一個專案。
2. 重要實驗保留環境、seed、backend、時間與 metrics。
3. QML 結果必須對照 Classical Baseline；沒有 Advantage 就明確說沒有。

從 Day 3 起，每篇都必須留下程式或可重現實驗：Day 3 用 NumPy 實作 quantum gate，Day 4 提前使用 CUDA-Q 建立 Bell state，Day 5 統整第一個完整 circuit demo。理論會繼續出現，但不再單獨占滿一篇。

建議 commit：

```text
chore: initialize 30-day QML engineering series
```

## 11. QML 研究資源怎麼讀？

建議不要從零散關鍵字一路亂讀，而是分四層：

1. **基礎語言**：Nielsen 與 Chuang 的標準教材，補 state、measurement、gate。[F1]
2. **研究地圖**：先讀 Biamonte et al. 2017，再讀 Cerezo et al. 2022，觀察研究問題如何從演算法可能性移到 NISQ 限制。[R1][R2]
3. **Supervised QML**：以 Schuld 與 Petruccione 的專書整理 encoding、inference 與 training。[R4]
4. **原始論文**：進入特定主題後回到 primary source，例如 Day 12–14 的 data encoding 會直接閱讀 Schuld、Sweke、Meyer 2021。[P1]

完整且已核對的書目、版本狀態與使用目的見 [文獻與資源索引](../../REFERENCES.md)。本系列不把 arXiv 等同 peer-reviewed publication，也不引用沒有讀到支持段落的論文。

## 12. 本日文獻

- [R1] Biamonte et al., “Quantum machine learning,” *Nature* 549, 195–202 (2017), [DOI](https://doi.org/10.1038/nature23474).
- [R2] Cerezo et al., “Challenges and opportunities in quantum machine learning,” *Nature Computational Science* 2, 567–576 (2022), [DOI](https://doi.org/10.1038/s43588-022-00311-3).
- [R4] Schuld and Petruccione, *Supervised Learning with Quantum Computers*, Springer (2018), [DOI](https://doi.org/10.1007/978-3-319-96424-9).
- [F1] Nielsen and Chuang, *Quantum Computation and Quantum Information*, 10th Anniversary Edition, Cambridge University Press (2010), [publisher page](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE).
- [P1] Schuld, Sweke, and Meyer, “Effect of data encoding on the expressive power of variational quantum-machine-learning models,” *Physical Review A* 103, 032430 (2021), [DOI](https://doi.org/10.1103/PhysRevA.103.032430).
- [D2] [NVIDIA CUDA-Q Local Installation](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html), accessed 2026-09-01.

## 13. 下一篇

[Day 02｜從 Bit 到 Qubit：工程師需要懂多少量子力學？](../day02/README.md)

下一篇從 `0`、`1` 走到 `|0⟩`、`|1⟩` 與 `α|0⟩ + β|1⟩`，建立閱讀 Quantum Circuit 與 QML 論文所需的最小數學語言。
