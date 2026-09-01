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

目前書目查閱日期：**2026-09-01**。

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

## 引用格式範本

文章內第一次出現時使用：

```text
作者指出的實際主張。[R2]
```

文章末尾列出「本日文獻」，提供題名、出版資訊與可解析連結。Repository 內以 `R`（review）、`F`（foundation）、`P`（primary research）、`D`（documentation）作為穩定索引，但投稿稿件會轉換成目標期刊要求的引用格式。
