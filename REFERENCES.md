# 文獻與資源索引

本頁保存系列使用的論文、教材、資料集與官方文件。每筆來源列出用途與既有核對紀錄，方便從章節結論回查原始資料。正式題名與作者保留原文；中文說明解釋來源與實驗的關係。

This index preserves the papers, textbooks, datasets, and official documentation used throughout the series. Original titles and author names remain unchanged. The notes describe each source's role and the scope of previously recorded checks; editing these notes does not constitute a new source review.

## 如何閱讀書目與版本

| 標記或用語 | 意義 |
|---|---|
| R | 綜述或調查論文，整理既有研究方向 |
| F | 基礎教材，建立概念與數學定義 |
| P | 原始研究，提出方法、推導或實驗結果 |
| D | 技術文件或資料來源 |
| DOI | 數位物件識別碼，用來定位特定論文或資料集 |
| arXiv／preprint | 預印本平台與預印本；上架不等於已完成期刊審查 |
| Version of record | 出版社正式發布的版本，可能與預印本不同 |
| Peer-reviewed | 經同行審查，由相關領域研究者評閱 |
| 卷、頁碼、文章編號 | 出版物中的定位資訊；有些期刊使用文章編號代替頁碼 |
| API／SDK | 程式互動介面／軟體開發工具組 |
| `latest` | 隨網站更新的文件版本，不保證與本機安裝版本相同 |

例如章節中的 `[R2]` 對應下方 R2 書目。綜述適合建立研究地圖，特定定理或效果則需連同原始研究的假設判讀。書目保存作者順序、出版資訊、版本與查閱日期；某筆新增核對不表示其他來源同時更新。

## 常見概念

| 用語 | 白話說明 |
|---|---|
| QML／NISQ | 量子機器學習／仍有噪聲且規模有限的量子裝置階段 |
| 狀態向量、振幅、Dirac 記號 | 用複數表示量子狀態；振幅絕對值平方決定機率；`\|ψ⟩` 是表示狀態的符號 |
| 量子閘、RX／RY／RZ | 改變狀態的操作；後三者分別繞 X、Y、Z 軸旋轉 |
| H、CNOT、Pauli 操作 | H 轉換常用量測基底；CNOT 依控制位元翻轉另一位元；Pauli X、Y、Z 是基本單位元操作 |
| Bell／GHZ 態 | 常見糾纏態；糾纏表示多位元狀態不能各自獨立描述 |
| 密度矩陣、混合態、約化狀態 | 矩陣可描述多種狀態的機率混合；只觀察部分系統時使用約化狀態 |
| 可分離態 | 能寫成各子系統乘積態的機率混合，沒有糾纏 |
| 量測基底、期望值、shots | 區分狀態的方向、按機率加權的量測平均、重複準備與量測次數 |
| 特徵映射、編碼、Ansatz | 將資料轉成量子表示、實際轉換方法、預先選定的可調電路模板 |
| 梯度、可訓練性、貧瘠高原 | 參數改動時目標的變化率、是否容易有效更新、某些條件下梯度集中近零的現象 |
| MLP／PQC／QNN | 傳統多層神經網路／可調參數量子電路／量子神經網路 |
| 核函數、核嶺迴歸 | 計算樣本相似度的函數、由相似度表求預測係數的方法 |
| CPU／GPU／QPU | 中央處理器／圖形處理器／量子處理器；前兩者在此可用於模擬 |
| CUDA／CUDA-Q／cuQuantum | NVIDIA 運算平台／量子程式工具／量子模擬加速函式庫 |
| 驅動程式／工具包／執行環境 | 與裝置溝通的軟體／開發工具集合／程式執行所需元件 |
| 後端、編譯、非同步 | 執行計算的工具、將程式轉成可執行形式、提交後先繼續其他工作再取結果 |

NumPy 是 Python 數值運算套件。Bloch 向量用三個實數描述單一量子位元，保真度衡量狀態相似程度。解析式是可直接推導答案的公式，變異數衡量抽樣結果的分散程度。鏈式法則連結多層轉換的導數，最佳化器決定參數如何更新。量子程式中的 kernel 與計算樣本相似度的核函數是不同概念。

## 查閱與實作紀錄

原有書目查閱日期：**2026-09-01**。Day 04 於 **2026-09-06** 另查閱 D1、D3、D5 與 F2；此更新不代表其他論文已重新核對。

Day 05 同日查閱 D3 與新增的 D6；本日流程的解析期望值與抽樣變異數由代碼和測試核對，未新增 QML 效果或量子優勢宣稱。

Day 06 同日查閱 D1、D4，新增 D7–D9；驅動程式／工具包／執行環境版本分層記錄，實際可執行性由 [環境診斷報告](results/day06/environment.json) 核對。

Day 07 同日查閱 D5，新增 D10–D11；量子位元配置、參數、迴圈、分支與基底標籤行為由 CPU／GPU 測試確認。

Day 08 同日查閱 D3、D5，沿用 D11 的量子電路組合；實測 CUDA-Q 0.15.1 的抽樣計數、run 逐次整數回傳、observe 有限次量測與 shots_count=-1。以 cos(theta)／sin(theta) 核對 Z0／X0X1，不將精確模擬值視為 QPU 的直接讀出能力。

Day 09 同日查閱 D10 中使用 `list[float]`（浮點數串列）作參數的量子程式，沿用 D3／D5 的 observe 與狀態核對方式。教學 Ansatz 的資料／權重分離、零權重解析式與參數週期性由 CPU／GPU 測試確認；沒有訓練或研究效果宣稱。

Day 11 於 2026-09-06 查閱 D5 的量子資料嵌入（將輸入轉成量子狀態）定義；本日使用 NumPy 比較基底／振幅表示，實際 CUDA-Q 電路沿用 Day 9 RY 特徵映射。資料縮放與編碼／讀出碰撞由解析式及 CPU／GPU 測試核對。

Day 12 於 2026-09-06 查閱 D5 的角度編碼與 Pauli 旋轉的定義慣例；以明寫 RX／RY／RZ 量子閘實作，未依賴額外的輔助函式。三種角度映射、H→RZ 準備順序、Bloch 向量的符號與乘積態保真度由 NumPy／CPU／GPU 核對。

Day 14 於 2026-09-07 查閱 D11 的量子電路組合與 qview 介面；重用 Day 9、12、13 量子程式和 Day 10 最佳化器，CPU／GPU 核對組合模型及短訓練，未新增研究成效宣稱。

Day 19 於 2026-09-07 重新查閱 D15 的混合模型結構示例；未移植歷史梯度程式，以 Day 17 參數位移規則和 NumPy 完整損失函數差分核對編碼層、量子權重與輸出層的鏈式法則。

## QML 入門綜述與專書

### R1｜經典 QML 綜述

Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, and Seth Lloyd. “Quantum machine learning.” *Nature* 549, 195–202 (2017). DOI: [10.1038/nature23474](https://doi.org/10.1038/nature23474).

- 類型：綜述論文。
- 用途：QML 的歷史脈絡、演算法分類，以及早期軟硬體挑戰。
- 核對：Nature 出版頁列出 2017-09-14 發表、卷 549、頁 195–202。

### R2｜近期挑戰與機會

M. Cerezo, Guillaume Verdon, Hsin-Yuan Huang, Lukasz Cincio, and Patrick J. Coles. “Challenges and opportunities in quantum machine learning.” *Nature Computational Science* 2, 567–576 (2022). DOI: [10.1038/s43588-022-00311-3](https://doi.org/10.1038/s43588-022-00311-3).

- 類型：綜述論文。
- 用途：NISQ 時代的可訓練性、量子優勢與 QML 研究限制。
- 核對：Nature 出版頁列出 2022-09-15 正式出版版本。

### R3｜工程導向調查論文（預印本）

Kamila Zaman, Alberto Marchisio, Muhammad Abdullah Hanif, and Muhammad Shafique. “A Survey on Quantum Machine Learning: Current Trends, Challenges, Opportunities, and the Road Ahead.” arXiv:2310.10315 (2023). [arXiv record](https://arxiv.org/abs/2310.10315).

- 類型：arXiv 調查論文；目前在本系列中以預印本身分引用。
- 用途：QML 演算法、硬體、軟體工具與應用的廣泛索引。
- 注意：它適合找研究分支，不單獨作為強效果宣稱的證據。

### R4｜監督式 QML 專書

監督式學習使用帶有正確答案的資料來訓練模型。ISBN 是書籍識別碼，eBook 表示電子書。

Maria Schuld and Francesco Petruccione. *Supervised Learning with Quantum Computers*. Springer, 2018. DOI: [10.1007/978-3-319-96424-9](https://doi.org/10.1007/978-3-319-96424-9).

- 類型：研究專書。
- 用途：資訊編碼、量子模型求值與訓練與監督式量子機器學習的系統化入口。
- 核對：Springer 書目頁列出 eBook 於 2018-08-30 出版，ISBN 978-3-319-96424-9。

## 量子計算基礎

### F1｜標準教材

Michael A. Nielsen and Isaac L. Chuang. *Quantum Computation and Quantum Information: 10th Anniversary Edition*. Cambridge University Press, 2010. [Publisher page](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE).

- 用途：狀態向量、Dirac 記號、量測、量子閘與量子資訊的基礎定義。
- 核對：Cambridge 出版頁列出作者與 2010 年 10th Anniversary Edition。

### F2｜混合態、可分離性與約化狀態

IBM Quantum Learning, “Multiple systems and reduced states,” *General formulation of quantum information*.
[官方教材](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/multiple-systems)。

- 類型：官方教學教材，非原始研究論文。
- 用途：Day 04 的經典相關混合態、可分離性與 Bell 態約化狀態。
- 查閱：2026-09-06；教材以乘積態的機率混合說明可分離態，並指出 Bell 態的局部約化狀態為完全混合態。
- 本日 Z／X 基底機率表另由 NumPy 密度矩陣計算與自動測試核對。

## 資料編碼與量子模型的原始研究

### P1｜資料編碼與表達能力

表達能力描述模型能表示哪些輸入與輸出關係，頻譜則有助於分析可表示的變化頻率。

Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer. “Effect of data encoding on the expressive power of variational quantum-machine-learning models.” *Physical Review A* 103, 032430 (2021). DOI: [10.1103/PhysRevA.103.032430](https://doi.org/10.1103/PhysRevA.103.032430).

- 類型：經同行審查的研究論文。
- 用途：Day 12–14 的資料編碼、資料重複編碼與傅立葉頻譜（將函數拆成不同頻率成分） 討論。
- 核對：APS 頁列出作者、文章編號 032430，發表日 2021-03-24。

### P2｜量子電路學習

K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii. “Quantum circuit learning.” *Physical Review A* 98, 032309 (2018). DOI: [10.1103/PhysRevA.98.032309](https://doi.org/10.1103/PhysRevA.98.032309).

- 類型：經同行審查的研究論文。
- 用途：Day 3 的可調角度旋轉與 QML 連接，以及 Day 9–10 的傳統與量子交替運算的訓練流程。
- 核對：APS 正式出版版本列出四位作者、卷 98、文章編號 032309，發表日 2018-09-10。
- 支持範圍：論文提出以一般電腦迭代調整淺層可調量子電路的傳統與量子共同學習的架構；本系列不由此延伸宣稱已獲得量子優勢。

### P4｜資料重複編碼

通用近似討論在指定條件與足夠資源下，能否逼近一類函數，不等於小型實驗已達到相同能力。ZZ 將兩位元相同結果記 +1、不同記 −1，再取平均。

Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I. Latorre. “Data re-uploading for a universal quantum classifier.” *Quantum* 4, 226 (2020). DOI: [10.22331/q-2020-02-06-226](https://doi.org/10.22331/q-2020-02-06-226).

- 類型：經同行審查的研究論文；[出版頁](https://quantum-journal.org/papers/q-2020-02-06-226/)。
- Day22 於 2026-09-07 核對作者、題名、發表日 2020-02-06、卷 4／文章編號 226。出版頁對應 arXiv v2，另提示較新的 v3；本日引用期刊版本。
- 支持範圍：重複資料編碼與可訓練操作、量子位元與層數取捨。本日 RY／CNOT／ZZ 與 Iris 配對實驗是受限示範，不聲稱論文完整復現、通用近似或量子優勢。

## 官方技術文件

### D1｜CUDA-Q Quick Start

[NVIDIA CUDA-Q Quick Start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)

- 用途：安裝、第一個量子程式、Bell／GHZ 態與 `cudaq.sample`。
- 注意：`latest` 文件會更新；Day06 的實際安裝版本已保存在環境紀錄。

### D2｜CUDA-Q Local Installation

WSL2 是 Windows 上的 Linux 執行環境，ARM64 與 Ampere 分別是處理器與 GPU 架構名稱；compute capability 是 NVIDIA 標示 GPU 功能能力的版本。這裡保留當時查閱條件，不代表目前所有版本的安裝要求。

[NVIDIA CUDA-Q Local Installation](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html)

- 用途：作業系統、Python、CUDA、驅動程式與 GPU 模擬條件。
- 2026-09-01 查閱結果：官方列出 Linux、macOS ARM64、Windows via WSL2；GPU 架構包含 Ampere，最低 compute capability 7.5。實際環境仍以安裝當天頁面為準。

### D3｜CUDA-Q Kernel Execution

`sample` 回傳各量測結果的次數，`run` 取得程式逐次回傳值，`observe` 計算期望值，`get_state` 取得模擬狀態。直接從狀態計算與有限次抽樣是不同的求值方式。

[Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)

- 用途：`sample`、`run`、`observe`、`get_state` 與非同步版本。

### D4｜cuQuantum

張量網路將大型狀態拆成互相連接的多維陣列，成本取決於狀態與連接結構。

[NVIDIA cuQuantum Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/index.html)

- 用途：以 GPU 加速狀態向量與張量網路模擬；用來區分 CUDA-Q 與 cuQuantum 的角色。

### D5｜CUDA-Q Python API

[NVIDIA CUDA-Q Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)

- 查閱日期：2026-09-06。
- 用途：Day 04 的 `sample(..., explicit_measurements=True)` 與量測輸出順序。
- 同日查閱 D1 的 Bell 態電路／安裝提醒與 D3 的 sample／get_state。Day 04 實際安裝 `cuda-quantum-cu12==0.15.1`，完整依賴見 [requirements-day04-lock.txt](requirements-day04-lock.txt)。`latest` 文件會更新，實際行為另由本專案 CPU／GPU 測試確認。

### D6｜Notebook 程式化執行

Notebook 是結合文字、程式與輸出的筆記本；儲存格是其中一段內容。nbclient 負責執行，nbformat 處理檔案格式，ipykernel 提供 Python 執行核心。

[nbclient — Executing notebooks](https://nbclient.readthedocs.io/en/latest/client.html)

- 類型：Jupyter nbclient 官方文件。
- 查閱日期：2026-09-06。
- 用途：Day 05 以 `NotebookClient` 執行所有程式儲存格，成功後保存輸出；遇到儲存格執行錯誤時失敗。
- 實際環境：nbclient 0.11.0、nbformat 5.11.1、ipykernel 7.3.0；完整版本見 [requirements-day05-lock.txt](requirements-day05-lock.txt)。

### D7｜CUDA 驅動程式、工具包與硬體架構

[NVIDIA CUDA Toolkit, Driver, and Architecture Matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)

- 查閱日期：2026-09-06。
- 用途：區分驅動程式的 CUDA 支援資訊與實際工具包版本；`nvidia-smi` 的 CUDA Version 不代表已安裝工具包的清單。

### D8｜CUDA 次版本相容性

主版本是版本號較前面的主要編號，次版本表示同系列內的更新。PTX 是 GPU 程式的中間表示，nvcc 是 CUDA 編譯工具；虛擬環境中的執行套件版本可能與系統工具不同。

[NVIDIA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

- 查閱日期：2026-09-06。
- 用途：說明 CUDA 主版本系列內的次版本相容性，以及新功能、PTX 等限制；不把主版本相同當成所有功能皆可運作的保證。
- Day 06 分別保存系統 nvcc 12.8、venv 執行環境套件 12.9.79 與 Hello Quantum 執行結果。

### D9｜CUDA 平台官方入口

[NVIDIA CUDA Platform](https://developer.nvidia.com/cuda)

- 查閱日期：2026-09-06；CUDA Zone 入口重新導向此頁。
- 用途：CUDA 平台、GPU 開發工具與執行環境的基本分工；配合 D4 的 cuQuantum 與 D1 的 CUDA-Q 定位。

### D10｜量子電路建構

`qubit` 表示單一量子位元，`qvector` 管理一組位元。配置是建立這些位元，型別化參數則明確指定輸入是整數、角度或其他資料。

[NVIDIA CUDA-Q Building Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html)

- 查閱日期：2026-09-06。
- 用途：Day 07 的 qubit／qvector 配置、型別化參數與基本電路建構。

### D11｜量子程式規格

`qview` 是查看一部分量子位元的介面；非對稱測試用不同位元結果辨認排序，避免只看 00 或 11 而漏掉順序問題。

[NVIDIA CUDA-Q Quantum Kernels Specification](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/kernels.html)

- 查閱日期：2026-09-06。
- 用途：主機端與量子程式的分界、可用型別與一般程式的條件與迴圈。本日實作輸入參數驅動的 if／for，未驗證依量測結果控制後續操作。
- 同日 D5 核對 `State.amplitude(bitstring)` 與明確指定的量測，避免以 GHZ 對稱字串推定所有位元順序均正確。CUDA-Q 0.15.1 CPU／GPU 均測試非對稱案例。

## 延伸方法與資料來源

### D12｜量子演算法基本操作

座標搜尋每次嘗試增減一個參數，依目標值選擇更新；最佳化器是決定更新方式的演算法。

- 官方文件：[NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/algorithmic_primitives.html)。
- Day 10 於 2026-09-06 查閱：`observe` 計算的期望值可作為傳統目標函數最佳化的核心；本日使用自訂座標搜尋，未使用 CUDA-Q 最佳化器 API。

### D13｜狀態準備與 MPS 範例

MPS（矩陣乘積態）以相連的小矩陣描述狀態，是張量網路的一種。將陣列直接放入模擬器記憶體，與建立可在硬體執行的準備電路，成本不同。正規化使振幅絕對值平方總和為 1。

- 官方文件：[NVIDIA Approximate State Preparation using MPS Sequential Encoding（0.13.0）](https://nvidia.github.io/cuda-quantum/0.13.0/applications/python/mps_encoding.html)。
- Day 13 於 2026-09-07 查閱：狀態向量到電路的準備工作不同於模擬器狀態緩衝區初始化。本日未使用 MPS 或宣稱一般高維量子閘成本；另核對 D3 的狀態緩衝區順序與 D5 的正規化定義，實際執行 CUDA-Q 0.15.1。

### D14｜前處理與資料洩漏

資料洩漏是評估資料提前影響模型或轉換規則，可能使成績過度樂觀。訓練資料建立規則，驗證資料選模型，測試資料最後評分。scikit-learn 是傳統機器學習套件。

- 官方文件：[scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)。
- Day 15 於 2026-09-07 查閱：資料先切分，前處理只以訓練資料建立規則，測試資料不參與模型選擇。本日以 NumPy 實作，沒有安裝 scikit-learn。

### D15｜混合量子神經網路教學（歷史版本）

- 官方文件：[NVIDIA Hybrid Quantum Neural Networks，CUDA-Q 0.8.0](https://nvidia.github.io/cuda-quantum/0.8.0/examples/python/tutorials/hybrid_qnns.html)。
- Day 16 於 2026-09-07 查閱，僅支持傳統運算層與量子量測期望值可整合的例子；不以該歷史教學的 API／梯度程式作本日正確性依據。Day 16 以 NumPy MLP 與既有 CUDA-Q 0.15.1 PQC 實作。

### P3｜解析量子梯度

鏈式法則將多層轉換的導數連起來；參數位移法以特定角度變化求導數，有限差分則用微小改動近似導數。

Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, Nathan Killoran. “Evaluating analytic gradients on quantum hardware.” *Physical Review A* 99, 032331 (2019). DOI: [10.1103/PhysRevA.99.032331](https://doi.org/10.1103/PhysRevA.99.032331)。[arXiv:1811.11184](https://arxiv.org/abs/1811.11184) 為 2018 預印本。

- Day 17 於 2026-09-07 核對出版商的作者、題名、正式年份、DOI 與特定條件下兩次參數位移後的執行取得導數的主張。
- 本日只將標準兩點規則用於每個權重獨立出現一次的 RY；損失函數的鏈式法則、共享參數反例與矩陣參考計算另以程式驗證。

### D16｜邏輯斯迴歸的標準損失

負對數概似對模型賦予正確答案的機率取負對數，與此分類情境的交叉熵相關。Brier 損失是輸出與 0／1 標籤的平均平方誤差；logistic-link 使用 sigmoid 將分數轉到 0 與 1 之間。

- [scikit-learn log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)。
- Day 18 於 2026-09-07 查閱：標準邏輯斯迴歸使用負對數概似；本文的 logistic-link 使用共同 Brier 損失訓練，與標準邏輯斯迴歸求解器的比較不同。另查閱 D14 的資料洩漏邊界。

### D17｜Iris 鳶尾花資料

Iris 是鳶尾花資料，versicolor 與 virginica 是本章選用的兩類。特徵是描述樣本的數值，標籤是正確類別；SHA-256 是由檔案內容計算的摘要。CC BY 4.0 表示需依授權條件署名。

- R. A. Fisher, Iris, UCI Machine Learning Repository；[官方頁面](https://archive.ics.uci.edu/dataset/53/iris)，DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76)。
- Day 20 於 2026-09-07 核對 150 樣本、4 特徵、3 類別、CC BY 4.0，下載原始 `iris.data` 並記錄 SHA-256。
- 本日只使用 versicolor／virginica，去除完全重複特徵＋標籤後為 99 筆，再做僅用訓練資料建立的標準化與兩主成分表示；不宣稱完整三分類 Iris 比較實驗。

### D18｜PCA 與解釋變異比例

PCA 將欄位組合成較少的新座標；SVD（奇異值分解）是找出主要方向的矩陣分解方法。PCA2 保留兩個主成分，解釋變異比例描述保留多少數值變化，不代表相同比例的分類訊息。

- 官方文件：[scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)。
- Day21 於 2026-09-07 查閱：以僅使用訓練資料的 SVD 實作 PCA2；變異比例不視為分類資訊保留率。未安裝 scikit-learn。

### D19｜特徵選擇

點二系列相關係數衡量連續欄位與二分類標籤的關聯。SelectKBest 是套件的特徵選擇工具名稱；統計顯著性檢定則評估差異能否僅由隨機波動解釋。

- 官方文件：[scikit-learn Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)。
- Day21 於 2026-09-07 查閱：參照逐一評分各欄位的排序流程，自行用 NumPy 計算欄位與二分類標籤的點二系列相關係數，取絕對值最高的兩個欄位；不聲稱使用 SelectKBest 或做統計顯著性檢定。資料洩漏規則另見 D14。

### P5｜量子特徵空間與核估計

SVM（支援向量機）是另一種分類方法，本系列的核嶺迴歸並非同一個求解方式。

Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta. “Supervised learning with quantum-enhanced feature spaces.” *Nature* 567, 209–212 (2019). DOI: [10.1038/s41586-019-0980-2](https://www.nature.com/articles/s41586-019-0980-2)。

- Day23 於 2026-09-07 查閱 Nature 出版索引：發表日 2019-03-13、作者與頁碼。正文頁存取受網站重新導向限制，未宣稱閱讀全文。
- 支持範圍：量子特徵空間／核函數估計概念；本日自訂 RY／CNOT 編碼與核嶺迴歸，不是論文電路／SVM 完整復現，不宣稱量子優勢。

### D20｜核嶺迴歸

KRR 是核嶺迴歸的縮寫；正則化限制預測函數的大小，核技巧透過相似度運算，免去列出所有轉換後特徵。`solve` 解線性方程組；將分數截斷到 [0, 1] 不保證輸出 0.8 就對應約八成實際發生率。

- [scikit-learn 官方文件](https://scikit-learn.org/stable/modules/kernel_ridge.html)，Day23 於 2026-09-07 查閱。
- KRR 結合平方誤差、正則化與核技巧；本日用 NumPy solve 實作，沒有安裝 scikit-learn。截斷後分數僅用於本日驗證資料／評分，不宣稱經校準的機率。

### P6｜貧瘠高原

2-design 是指定低階統計量與理想均勻隨機操作一致的條件，不能只憑電路看似隨機就認定成立。初始化是設定起始權重，漸近行為描述規模持續增加時的趨勢。CZ 是受控相位操作。

Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven. “Barren plateaus in quantum neural network training landscapes.” *Nature Communications* 9, 4812 (2018). DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)。[arXiv:1803.11173](https://arxiv.org/abs/1803.11173)。

- Day24 於 2026-09-07 核對 arXiv 作者、題名與期刊書目。arXiv v1 提交於 2018-03-29；期刊為 2018 年，兩者版本不混稱。
- 支持範圍：隨機初始化電路的梯度集中與 2-design 關係。本日 n≤8、32 個隨機種子是小規模診斷，不宣稱證明此 CZ 電路模板 的漸近規模成長趨勢。

### P7｜目標作用範圍與貧瘠高原

局部目標只讀取少數位元，全域目標涉及全部位元。深度計算必須依序完成的操作層數。

M. Cerezo, Akira Sone, Tyler Volkoff, Lukasz Cincio, and Patrick J. Coles. “Cost function dependent barren plateaus in shallow parametrized quantum circuits.” *Nature Communications* 12, 1791 (2021). DOI: [10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w)。[arXiv:2001.00550v3](https://arxiv.org/abs/2001.00550v3)。

- Day24 於 2026-09-07 核對出版索引與 arXiv：期刊 2021-03-19，v3 於 2021-03-20 更新並標示已出版版本；原始預印本為 2020 年。
- 論文結果有局部 2-design 區塊等假設；目標作用範圍與電路深度共同影響梯度。本日 Z0／全域奇偶性比較不是完整定理復現，也不將更換目標函數當成保持原任務的保證。

### D21｜UCI Wine 資料

Wine 以化學特徵辨認葡萄栽培品種；Wine Quality 是另一個資料集，任務不同。

S. Aeberhard and M. Forina (1992). Wine [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)。[官方頁面](https://archive.ics.uci.edu/dataset/109/wine)。

- Day25 於 2026-09-07 查閱：178 筆、13 個特徵、三種栽培品種，CC BY 4.0。資料來源不是 Wine Quality。
- 實驗預先取類別 2／3、119 筆、無完全重複特徵＋標籤；原始 `wine.data`不修改，來源 SHA-256 與欄位順序保存於實驗設定。
- 支持範圍：來源、欄位、類別與授權；本日二分類 PCA2 模型的結果不宣稱完整 Wine 三分類表現。

### D22｜CUDA-Q 噪聲模型

Kraus 算符是一組描述噪聲作用的矩陣。去極化通道混合 X、Y、Z 操作；完全混合的單位元狀態對任何量測方向都給出各半機率。

- [NVIDIA Noisy Simulation](https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html)，2026-09-07 查閱。
- Day26 使用本機 0.15.1 NoiseModel.add_channel 與 density-matrix-cpu；通道慣例另依本機 Kraus 函式說明與解析測試核對。DepolarizationChannel 的 p 是執行 X、Y 或 Z 操作的合計機率，完全混合在 p=.75。

### D23｜CUDA-Q 含噪聲模擬器

軌跡是某次隨機噪聲作用下的狀態演化，多次抽樣反映整體分布；這與直接保存完整密度矩陣不同。

- [NVIDIA Noisy Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/noisy.html)，2026-09-07 查閱。
- density-matrix-cpu 提供密度矩陣模擬；GPU nvidia 支援含噪聲軌跡抽樣。本日不宣稱 GPU 直接計算完整密度矩陣或 QPU 誤差校準。

### D24｜狀態向量後端與精度

fp32／fp64 分別以 32／64 位元保存實數，後者精度與儲存需求較高。

- [NVIDIA State Vector Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html)，Day27 於 2026-09-07 查閱。
- nvidia 預設 fp32，可透過 option='fp64'指定雙精度；Day27 將 CPU／GPU fp64 作主配對，fp32 另列。時間結論只依本機實測，不由文件推論固定加速比。

### D25｜多 GPU 狀態向量模擬

`mgpu` 將單一狀態向量分散到多張卡。MPI 是多程序交換訊息的介面，容量表示能容納多大狀態，延遲表示取得結果前的等待時間。

- [NVIDIA State Vector Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html)，Day28 於 2026-09-08 查閱。
- 支持 mgpu,fp64、MPI 啟動形式與資源條件。另核對本機 0.15.1 targets/nvidia.yml 的配置；latest 文件的完整預設值不視為本機版本保證。Day28 容量與延遲數字來自自訂假設模型，沒有多 GPU 實測。

### D26｜多個模擬 QPU 的任務分工

`mqpu` 讓各卡處理獨立任務；`observe_async` 提交非同步期望值計算，`qpu_id` 指定任務目的地。

- [NVIDIA Multiple QPUs](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/mqpusims.html)、[Multi-GPU Workflows](https://nvidia.github.io/cuda-quantum/latest/using/examples/multi_gpu_workflows.html)，Day28 於 2026-09-08 查閱。
- 支持 mqpu 模擬多個 QPU、observe_async 與 qpu_id 派送；任務平行與單一狀態向量分散必須區分。官方範例效能不移植為本機結論。

### D27｜硬體後端與本地預演

IonQ 是量子運算供應商，本地預演只在本機模擬指定後端的流程，不等於供應商已接收遠端工作。

- [NVIDIA Ion Trap Backends](https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware/iontrap.html)、[Quantum Hardware](https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware.html)，Day29 於 2026-09-08 查閱。
- 支持 ionq emulate=True 為本地無噪聲預演、供應商與實際裝置的區分、遠端帳號條件。本日 0.15.1 本地驗證成功；未提交雲端模擬器或實體 QPU，不固定裝置可用性與費用。

### D28｜抽樣與非同步硬體流程

future 代表尚待取得的結果，`get()` 等待完成；工作識別資訊用來追蹤遠端工作。末端量測明確指定最後需要讀出的位元。

- [Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html)、[Using Quantum Hardware Providers](https://nvidia.github.io/cuda-quantum/latest/using/examples/hardware_providers.html)，Day29 於 2026-09-08 查閱。
- 支持明確末端量測、sample_async／get 與工作識別資訊取得流程。Day29 以本地計數核對輸出格式與意義，沒有遠端 future／排隊實測。

### D29｜IonQ 工作狀態與附加資訊

附加資訊可包含裝置、時間與工作狀態。started／running 都可用來標示執行中，但應保留實際介面版本的原始名稱。

- [IonQ Jobs](https://docs.ionq.com/user-manual/jobs)、[API v0.4 Get Job](https://docs.ionq.com/api-reference/v0.4/jobs/get-job)，Day29 於 2026-09-08 查閱。
- 區分排隊、執行、完成、失敗與取消；版本間 started／running 名稱有差異。Day29 僅以文件說明工作管理，沒有直接呼叫 IonQ API；本地總等待時間不能當 QPU 執行時間。

## Day30 結論的專案證據入口

- [Day30 證據報告](results/day30/README.md)與[來源 SHA-256](results/day30/evidence.json)：從 Day20／23／25 保存預測核算評估指標，重算 Day26 Wine 噪聲評估指標與 Day27 fp64 計時比值，核對 Day28 模型與 Day29 本地預演範圍。
- 這是本專案結果彙整，不是新增外部文獻、硬體比較實驗或量子優勢證明。既有官方文件查閱日期與版本條件維持原紀錄。
