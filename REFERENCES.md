# 文獻與資源索引

本頁是系列共用的 source registry。文章中的文獻先在這裡核對，再加入每日文章，避免只複製搜尋結果或二手書目。

## 核對規則

每筆論文至少檢查：

1. 題名與作者順序。
2. 出版年份、期刊／會議、卷期與頁碼或 article number。
3. DOI 是否能解析到出版商頁面。
4. 是否只有 arXiv preprint；若已有 version of record，兩者分開標示。
5. 引用的主張是否真的出現在摘要或正文，不能只因題名相近便引用。
6. Survey／Review 用來建立全景；具體演算法、定理與實驗結論回到原始論文。
7. 軟體 API、安裝條件與硬體支援使用官方最新版文件，並記錄查閱日期。

原有書目查閱日期：**2026-09-01**。Day 04 於 **2026-09-06** 另查閱 D1、D3、D5 與 F2；此更新不代表其他論文已重新核對。

Day 05 同日查閱 D3 與新增的 D6；本日流程的解析期望值與抽樣變異數由代碼和測試核對，未新增 QML 效果或量子優勢宣稱。

Day 06 同日查閱 D1、D4，新增 D7–D9；driver／toolkit／runtime 版本分層記錄，實際可執行性由 [環境診斷報告](results/day06/environment.json) 核對。

Day 07 同日查閱 D5，新增 D10–D11；qubit allocation、參數、迴圈、分支與 basis label 行為由 CPU／GPU 測試確認。

Day 08 同日查閱 D3、D5，沿用 D11 的 kernel composition；實測 CUDA-Q 0.15.1 的 sample counts、run 逐次整數回傳、observe finite shots 與 shots_count=-1。以 cos(theta)／sin(theta) 核對 Z0／X0X1，不將精確模擬值視為 QPU 的直接讀出能力。

Day 09 同日查閱 D10 的 `list[float]` parameterized kernels，沿用 D3／D5 的 observe 與 state 核對方式。教學 Ansatz 的 data／weights 分離、零 weights 解析式與參數週期性由 CPU／GPU 測試確認；沒有訓練或研究效果宣稱。

Day 11 於 2026-09-06 查閱 D5 的 Quantum Embeddings 定義；本日使用 NumPy 比較 basis／amplitude 表示，實際 CUDA-Q 電路沿用 Day 9 RY feature map。資料縮放與編碼／讀出碰撞由解析式及 CPU／GPU 測試核對。

Day 12 於 2026-09-06 查閱 D5 的 angular encoding 與 Pauli rotation convention；以明寫 RX／RY／RZ gates 實作，未依賴 contrib helper。三種角度映射、H→RZ 準備順序、Bloch 符號與 product-state fidelity 由 NumPy／CPU／GPU 核對。

Day 14 於 2026-09-07 查閱 D11 的 kernel composition 與 qview 介面；重用 Day 9、12、13 kernels 和 Day 10 optimizer，CPU／GPU 核對組合模型及短訓練，未新增研究成效宣稱。

Day 19 於 2026-09-07 重新查閱 D15 的 hybrid architecture 示例；未移植歷史梯度程式，以 Day 17 shift 規則和 NumPy 完整 loss 差分核對 encoder、quantum weights 與 head 的 chain rule。

## QML 入門與 Review／Survey

### R1｜經典 QML Review

Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, and Seth Lloyd. “Quantum machine learning.” *Nature* 549, 195–202 (2017). DOI: [10.1038/nature23474](https://doi.org/10.1038/nature23474).

- 類型：Review Article。
- 用途：QML 的歷史脈絡、演算法分類，以及早期軟硬體挑戰。
- 核對：Nature 出版頁列出 2017-09-14 發表、卷 549、頁 195–202。

### R2｜近期挑戰與機會

M. Cerezo, Guillaume Verdon, Hsin-Yuan Huang, Lukasz Cincio, and Patrick J. Coles. “Challenges and opportunities in quantum machine learning.” *Nature Computational Science* 2, 567–576 (2022). DOI: [10.1038/s43588-022-00311-3](https://doi.org/10.1038/s43588-022-00311-3).

- 類型：Review Article。
- 用途：NISQ 時代的 trainability、quantum advantage 與 QML 研究限制。
- 核對：Nature 出版頁列出 2022-09-15 version of record。

### R3｜工程導向 Survey（preprint）

Kamila Zaman, Alberto Marchisio, Muhammad Abdullah Hanif, and Muhammad Shafique. “A Survey on Quantum Machine Learning: Current Trends, Challenges, Opportunities, and the Road Ahead.” arXiv:2310.10315 (2023). [arXiv record](https://arxiv.org/abs/2310.10315).

- 類型：arXiv survey；目前在本系列中以 preprint 身分引用。
- 用途：QML algorithms、hardware、software tools 與應用的廣泛索引。
- 注意：它適合找研究分支，不單獨作為強效果宣稱的證據。

### R4｜Supervised QML 專書

Maria Schuld and Francesco Petruccione. *Supervised Learning with Quantum Computers*. Springer, 2018. DOI: [10.1007/978-3-319-96424-9](https://doi.org/10.1007/978-3-319-96424-9).

- 類型：研究專書。
- 用途：information encoding、quantum inference／training 與 supervised QML 的系統化入口。
- 核對：Springer 書目頁列出 eBook 於 2018-08-30 出版，ISBN 978-3-319-96424-9。

## Quantum Computing 基礎

### F1｜標準教材

Michael A. Nielsen and Isaac L. Chuang. *Quantum Computation and Quantum Information: 10th Anniversary Edition*. Cambridge University Press, 2010. [Publisher page](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE).

- 用途：state vector、Dirac notation、measurement、quantum gate 與 quantum information 的基礎定義。
- 核對：Cambridge 出版頁列出作者與 2010 年 10th Anniversary Edition。

### F2｜Mixed State、Separability 與 Reduced State

IBM Quantum Learning, “Multiple systems and reduced states,” *General formulation of quantum information*.
[官方教材](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/multiple-systems)。

- 類型：官方教學教材，非原始研究論文。
- 用途：Day 04 的經典相關混合態、separability 與 Bell state reduced state。
- 查閱：2026-09-06；教材以 product states 的機率混合說明 separable states，並指出 Bell states 的局部 reduced states 為完全混合態。
- 本日 Z／X 基底機率表另由 NumPy density matrix 計算與自動測試核對。

## 後續實作會使用的原始論文

### P1｜Data Encoding 與 Expressive Power

Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer. “Effect of data encoding on the expressive power of variational quantum-machine-learning models.” *Physical Review A* 103, 032430 (2021). DOI: [10.1103/PhysRevA.103.032430](https://doi.org/10.1103/PhysRevA.103.032430).

- 類型：Peer-reviewed research article。
- 用途：Day 12–14 的 encoding、data re-uploading 與 Fourier spectrum 討論。
- 核對：APS 頁列出作者、article number 032430，發表日 2021-03-24。

### P2｜Quantum Circuit Learning

K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii. “Quantum circuit learning.” *Physical Review A* 98, 032309 (2018). DOI: [10.1103/PhysRevA.98.032309](https://doi.org/10.1103/PhysRevA.98.032309).

- 類型：Peer-reviewed research article。
- 用途：Day 3 的 parameterized rotation 與 QML 連接，以及 Day 9–10 的 hybrid optimization loop。
- 核對：APS version of record 列出四位作者、volume 98、article 032309，發表日 2018-09-10。
- 支持範圍：論文提出以 classical computer 迭代調整 low-depth parameterized quantum circuit 的 hybrid learning framework；本系列不由此延伸宣稱已獲得 quantum advantage。

## 官方技術文件

### D1｜CUDA-Q Quick Start

[NVIDIA CUDA-Q Quick Start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)

- 用途：安裝、第一個 kernel、Bell／GHZ state 與 `cudaq.sample`。
- 注意：`latest` 文件會更新；Day 6 將把實際安裝版本寫入環境紀錄。

### D2｜CUDA-Q Local Installation

[NVIDIA CUDA-Q Local Installation](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html)

- 用途：作業系統、Python、CUDA、driver 與 GPU simulation 條件。
- 2026-09-01 查閱結果：官方列出 Linux、macOS ARM64、Windows via WSL2；GPU 架構包含 Ampere，最低 compute capability 7.5。實際環境仍以安裝當天頁面為準。

### D3｜CUDA-Q Kernel Execution

[Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)

- 用途：`sample`、`run`、`observe`、`get_state` 與 asynchronous variants。

### D4｜cuQuantum

[NVIDIA cuQuantum Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/index.html)

- 用途：GPU-accelerated state-vector／tensor-network simulation；用來區分 CUDA-Q 與 cuQuantum 的角色。

### D5｜CUDA-Q Python API

[NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)

- 查閱日期：2026-09-06。
- 用途：Day 04 的 `sample(..., explicit_measurements=True)` 與量測輸出順序。
- 同日查閱 D1 的 Bell kernel／安裝提醒與 D3 的 sample／get_state。Day 04 實際安裝 `cuda-quantum-cu12==0.15.1`，完整依賴見 [requirements-day04-lock.txt](requirements-day04-lock.txt)。`latest` 文件會更新，實際行為另由本專案 CPU／GPU 測試確認。

### D6｜Notebook 程式化執行

[nbclient — Executing notebooks](https://nbclient.readthedocs.io/en/latest/client.html)

- 類型：Jupyter nbclient 官方文件。
- 查閱日期：2026-09-06。
- 用途：Day 05 以 `NotebookClient` 執行所有 code cells，成功後保存輸出；遇到 cell error 時失敗。
- 實際環境：nbclient 0.11.0、nbformat 5.11.1、ipykernel 7.3.0；完整版本見 [requirements-day05-lock.txt](requirements-day05-lock.txt)。

### D7｜CUDA Driver、Toolkit 與 Architecture

[NVIDIA CUDA Toolkit, Driver, and Architecture Matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)

- 查閱日期：2026-09-06。
- 用途：區分 driver 的 CUDA 支援資訊與實際 toolkit 版本；`nvidia-smi` 的 CUDA Version 不代表已安裝 toolkit 的清單。

### D8｜CUDA Minor Version Compatibility

[NVIDIA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

- 查閱日期：2026-09-06。
- 用途：說明 CUDA major family 內的 minor-version compatibility，以及新功能、PTX 等限制；不把 major version 相同當成所有功能皆可運作的保證。
- Day 06 分別保存系統 nvcc 12.8、venv runtime 套件 12.9.79 與 Hello Quantum 執行結果。

### D9｜CUDA 平台官方入口

[NVIDIA CUDA Platform](https://developer.nvidia.com/cuda)

- 查閱日期：2026-09-06；CUDA Zone 入口重新導向此頁。
- 用途：CUDA 平台、GPU 開發工具與 runtime 的基本分工；配合 D4 的 cuQuantum 與 D1 的 CUDA-Q 定位。

### D10｜Building Kernels

[NVIDIA CUDA-Q Building Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html)

- 查閱日期：2026-09-06。
- 用途：Day 07 的 qubit／qvector allocation、型別化參數與基本電路建構。

### D11｜Quantum Kernels Specification

[NVIDIA CUDA-Q Quantum Kernels Specification](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/kernels.html)

- 查閱日期：2026-09-06。
- 用途：host 與 kernel 的分界、可用型別與 classical control flow。本日實作輸入參數驅動的 if／for，未驗證 measurement-based feedback。
- 同日 D5 核對 `State.amplitude(bitstring)` 與 explicit measurements，避免以 GHZ 對稱字串推定所有位元順序均正確。CUDA-Q 0.15.1 CPU／GPU 均測試非對稱案例。

## 引用格式範本

文章內第一次出現時使用：

```text
作者指出的實際主張。[R2]
```

文章末尾列出「本日文獻」，提供題名、出版資訊與可解析連結。Repository 內以 `R`（review）、`F`（foundation）、`P`（primary research）、`D`（documentation）作為穩定索引，但投稿稿件會轉換成目標期刊要求的引用格式。

### D12｜Quantum Algorithmic Primitives

- 官方文件：[NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/algorithmic_primitives.html)。
- Day 10 於 2026-09-06 查閱：observe expectation 可作為 classical objective 最佳化的核心；本日使用自訂座標搜尋，未使用 CUDA-Q optimizer API。

### D13｜State Preparation 與 MPS 範例

- 官方文件：[NVIDIA Approximate State Preparation using MPS Sequential Encoding（0.13.0）](https://nvidia.github.io/cuda-quantum/0.13.0/applications/python/mps_encoding.html)。
- Day 13 於 2026-09-07 查閱：statevector 到電路的準備工作不同於 simulator buffer 初始化。本日未使用 MPS 或宣稱一般高維 gate 成本；另核對 D3 的 state buffer 順序與 D5 的 normalization 定義，實際執行 CUDA-Q 0.15.1。

### D14｜Preprocessing 與 Data Leakage

- 官方文件：[scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)。
- Day 15 於 2026-09-07 查閱：資料先切分，preprocessing 只在 train fit，test 不參與模型選擇。本日以 NumPy 實作，沒有安裝 scikit-learn。

### D15｜Hybrid QNN 教學（歷史版本）

- 官方文件：[NVIDIA Hybrid Quantum Neural Networks，CUDA-Q 0.8.0](https://nvidia.github.io/cuda-quantum/0.8.0/examples/python/tutorials/hybrid_qnns.html)。
- Day 16 於 2026-09-07 查閱，僅支持 classical layers 與 quantum expectation 可整合的例子；不以該歷史教學的 API／梯度程式作本日正確性依據。Day 16 以 NumPy MLP 與既有 CUDA-Q 0.15.1 PQC 實作。

### P3｜解析量子梯度

Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, Nathan Killoran. “Evaluating analytic gradients on quantum hardware.” *Physical Review A* 99, 032331 (2019). DOI: [10.1103/PhysRevA.99.032331](https://doi.org/10.1103/PhysRevA.99.032331)。[arXiv:1811.11184](https://arxiv.org/abs/1811.11184) 為 2018 preprint。

- Day 17 於 2026-09-07 核對出版商的作者、題名、正式年份、DOI 與特定條件下兩次 shifted execution 取得導數的主張。
- 本日只將標準兩點規則用於每個 weight 獨立出現一次的 RY；loss chain rule、共享參數反例與矩陣 reference 另以程式驗證。

### D16｜Logistic Regression 的標準 Loss

- [scikit-learn log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)。
- Day 18 於 2026-09-07 查閱：標準 logistic regression 使用 negative log-likelihood；本文 logistic-link 以共同 Brier loss 訓練，明確不冒稱標準 LogisticRegression solver benchmark。另查閱 D14 的資料洩漏邊界。
