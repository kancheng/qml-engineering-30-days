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

### P4｜Data Re-uploading

Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I. Latorre. “Data re-uploading for a universal quantum classifier.” *Quantum* 4, 226 (2020). DOI: [10.22331/q-2020-02-06-226](https://doi.org/10.22331/q-2020-02-06-226).

- 類型：Peer-reviewed research article；[出版頁](https://quantum-journal.org/papers/q-2020-02-06-226/)。
- Day22 於 2026-09-07 核對作者、題名、發表日 2020-02-06、volume4／article226。出版頁對應 arXiv v2，另提示較新的 v3；本日引用期刊版本。
- 支持範圍：重複資料編碼與 trainable operations、qubit／layer 取捨。本日 RY／CNOT／ZZ 與 Iris 配對實驗是受限示範，不聲稱論文完整復現、universal approximation 或量子優勢。

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

### D17｜Iris Dataset

- R. A. Fisher, Iris, UCI Machine Learning Repository；[官方頁面](https://archive.ics.uci.edu/dataset/53/iris)，DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76)。
- Day 20 於 2026-09-07 核對 150 instances、4 features、3 classes、CC BY 4.0，下載原始 iris.data 並記錄 SHA-256。
- 本日只使用 versicolor／virginica，去除完全重複 feature＋label 後為 99 筆，再做 train-only standardization／PCA2；不宣稱完整三分類 Iris benchmark。

### D18｜PCA 與 Explained Variance

- 官方文件：[scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)。
- Day21 於 2026-09-07 查閱：以 train-only SVD 實作 PCA2；variance ratio 不視為分類資訊保留率。未安裝 scikit-learn。

### D19｜Feature Selection

- 官方文件：[scikit-learn Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)。
- Day21 於 2026-09-07 查閱：參照單變量 ranking 流程，自行用 NumPy 實作 binary label 的絕對 point-biserial correlation top2；不聲稱使用 SelectKBest 或做統計顯著性檢定。資料洩漏規則另見 D14。

### P5｜Quantum Feature Space 與 Kernel Estimation

Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta. “Supervised learning with quantum-enhanced feature spaces.” *Nature* 567, 209–212 (2019). DOI: [10.1038/s41586-019-0980-2](https://www.nature.com/articles/s41586-019-0980-2)。

- Day23 於2026-09-07查閱Nature出版索引：發表日2019-03-13、作者與頁碼。正文頁存取受網站重新導向限制，未宣稱閱讀全文。
- 支持範圍：quantum feature spaces／kernel estimation概念；本日自訂RY／CNOT map與kernel ridge，不是論文電路／SVM完整復現，不宣稱量子優勢。

### D20｜Kernel Ridge Regression

- [scikit-learn官方文件](https://scikit-learn.org/stable/modules/kernel_ridge.html)，Day23於2026-09-07查閱。
- KRR結合squared loss、regularization與kernel trick；本日用NumPy solve實作，沒有安裝scikit-learn。Clipped score僅用於本日validation／評分，不宣稱calibrated probability。

### P6｜Barren Plateaus

Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven. “Barren plateaus in quantum neural network training landscapes.” *Nature Communications* 9, 4812 (2018). DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)。[arXiv:1803.11173](https://arxiv.org/abs/1803.11173)。

- Day24於2026-09-07核對arXiv作者、題名與journal reference。arXiv v1提交於2018-03-29；期刊為2018年，兩者版本不混稱。
- 支持範圍：隨機初始化電路的梯度集中與2-design關係。本日n≤8、32seeds是小規模診斷，不宣稱證明此CZ ansatz的漸近scaling。

### P7｜Cost Locality 與 Barren Plateaus

M. Cerezo, Akira Sone, Tyler Volkoff, Lukasz Cincio, and Patrick J. Coles. “Cost function dependent barren plateaus in shallow parametrized quantum circuits.” *Nature Communications* 12, 1791 (2021). DOI: [10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w)。[arXiv:2001.00550v3](https://arxiv.org/abs/2001.00550v3)。

- Day24於2026-09-07核對出版索引與arXiv：期刊2021-03-19，v3於2021-03-20更新並標示published version；原始preprint為2020年。
- 論文結果有local 2-design blocks等假設；locality與depth共同影響梯度。本日Z0／global parity比較不是完整定理復現，也不將更換cost當成保持原任務的保證。

### D21｜UCI Wine Dataset

S. Aeberhard and M. Forina (1992). Wine [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)。[官方頁面](https://archive.ics.uci.edu/dataset/109/wine)。

- Day25於2026-09-07查閱：178筆、13個features、三種cultivars，CC BY4.0。資料來源不是Wine Quality。
- 實驗預先取class2／3、119筆、無完全重複features＋label；原始wine.data不修改，來源SHA-256與欄位順序保存於protocol。
- 支持範圍：來源、欄位、類別與授權；本日binary PCA2模型的結果不宣稱完整Wine三分類表現。

### D22｜CUDA-Q NoiseModel

- [NVIDIA Noisy Simulation](https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html)，2026-09-07查閱。
- Day26使用本機0.15.1 NoiseModel.add_channel與density-matrix-cpu；channel慣例另依本機Kraus docstring與解析測試核對。DepolarizationChannel的p是非identity Pauli總機率，完全混合在p=.75。

### D23｜CUDA-Q Noisy Backends

- [NVIDIA Noisy Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/noisy.html)，2026-09-07查閱。
- density-matrix-cpu提供density matrix simulation；GPU nvidia支援trajectory noisy sampling。本日不宣稱GPU exact density matrix或QPU error calibration。

### D24｜Statevector Backend與Precision

- [NVIDIA State Vector Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html)，Day27於2026-09-07查閱。
- nvidia預設fp32，可透過option='fp64'指定雙精度；Day27將CPU／GPU fp64作主配對，fp32另列。時間結論只依本機實測，不由文件推論固定speedup。

### D25｜Multi-GPU Statevector Simulation

- [NVIDIA State Vector Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html)，Day28於2026-09-08查閱。
- 支持mgpu,fp64、MPI啟動形式與資源條件。另核對本機0.15.1 targets/nvidia.yml的配置；latest文件的完整預設值不視為本機版本保證。Day28容量與延遲數字來自自訂假設模型，沒有multi-GPU實測。

### D26｜Multi-QPU Task Parallelism

- [NVIDIA Multiple QPUs](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/mqpusims.html)、[Multi-GPU Workflows](https://nvidia.github.io/cuda-quantum/latest/using/examples/multi_gpu_workflows.html)，Day28於2026-09-08查閱。
- 支持mqpu模擬多個QPU、observe_async與qpu_id派送；任務平行與單一statevector分散必須區分。官方範例效能不移植為本機結論。

### D27｜Hardware Target與Local Emulation

- [NVIDIA Ion Trap Backends](https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware/iontrap.html)、[Quantum Hardware](https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware.html)，Day29於2026-09-08查閱。
- 支持ionq emulate=True為本地無noise emulation、provider與實際裝置的區分、遠端帳號條件。本日0.15.1本地驗證成功；未提交cloud simulator或physical QPU，不固定裝置可用性與費用。

### D28｜Sampling與Asynchronous Hardware Workflows

- [Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)、[Using Quantum Hardware Providers](https://nvidia.github.io/cuda-quantum/latest/using/examples/hardware_providers.html)，Day29於2026-09-08查閱。
- 支持明確terminal measurement、sample_async／get與job reference取得流程。Day29以本地counts驗證輸出契約，沒有遠端future／queue實測。

### D29｜IonQ Job生命週期與Metadata

- [IonQ Jobs](https://docs.ionq.com/user-manual/jobs)、[API v0.4 Get Job](https://docs.ionq.com/api-reference/v0.4/jobs/get-job)，Day29於2026-09-08查閱。
- 區分排隊、執行、完成、失敗與取消；版本間started／running名稱有差異。Day29僅以文件說明工作管理，沒有直接呼叫IonQ API；本地wall time不能當QPU execution time。

## Day30 結論的專案證據入口

- [Day30證據報告](results/day30/README.md)與[來源SHA-256](results/day30/evidence.json)：從Day20／23／25保存預測核算metrics，重算Day26 Wine noise metrics與Day27 fp64計時比值，核對Day28模型與Day29本地預演範圍。
- 這是本專案結果彙整，不是新增外部文獻、硬體benchmark或量子優勢證明。既有官方文件查閱日期與版本條件維持原紀錄。
