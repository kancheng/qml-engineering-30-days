# 從 AI 工程到量子機器學習：30 天 CUDA-Q 實作

這個系列以 Python 與 NVIDIA CUDA-Q 建立量子機器學習（Quantum Machine Learning，QML）的實驗流程，從量子狀態、資料編碼與模型訓練，逐步走到效能比較與硬體執行準備。QML 探討如何將量子計算用於資料學習；CUDA-Q 是編寫與執行量子程式的工具。系列排程從 **2026-09-15** 開始，共 30 篇。

每章都有概念說明，Day03 起附可執行程式或可重現實驗。現有成果包含小型模型、數值核對與保存結果；多 GPU 部分是情境模型，QPU（真實量子處理器）部分是本地預演，尚無多卡或真實 QPU 實測，也未證明量子方法優於適當的傳統方法。

This 30-part series uses Python and NVIDIA CUDA-Q to explore quantum machine learning, from quantum states and data encoding to model training, simulation performance, and hardware preparation. Articles include explanations and, from Day03 onward, executable examples or reproducible experiments. Multi-GPU results are scenario models, and QPU work is a local rehearsal; neither constitutes physical multi-GPU or QPU measurements.

## 閱讀與實驗方式

前兩章建立方向與基本數學語言，後續將概念接到程式、測試與結果。CUDA-Q 是主要工具，NumPy 提供一般電腦上的數值參考計算；PennyLane 與 Qiskit 是其他量子程式工具，在需要比較時介紹。

模型比較同時保留傳統方法作為基準，並記錄資料、訓練預算與結果。模擬器是用一般電腦計算量子狀態的程式，模擬變快不等於量子模型分類更準，也不代表整段訓練更快。各章的具體限制與執行日期保存在結果報告中。

## 設備與環境

[Day03 的環境與示範](articles/day03/README.md) 及 [設備紀錄](articles/day03/ENVIRONMENT.md) 說明專案設定。`.venv` 是獨立保存 Python 套件的虛擬環境，固定版本有助於重現相同執行條件。

| 設備 | 專案用途 |
|---|---|
| Windows Surface Pro 7 | 文章、版本管理、資料整理與小型 CPU 數值運算 |
| Ubuntu 與 NVIDIA RTX 3060 筆電 | CUDA-Q 電路、GPU 模擬、模型實驗、噪聲與計時 |

CPU 是中央處理器，GPU 是擅長平行運算的圖形處理器。Ubuntu 是 Linux 作業系統。實際執行條件以各日環境紀錄為準；安裝與硬體支援的歷史查閱結果見 [D2](REFERENCES.md)。Day20 與 Day25 的訓練由 NumPy 在 CPU 完成，CUDA-Q CPU／GPU 用於核對固定權重的輸出。

## 30 天章節目錄

以下為系列排程與已完成文章，日期範圍為 2026-09-15 至 2026-10-14。文章完成不表示所有提到的硬體實驗都已執行，具體範圍見各日報告。

| 日期 | Day | 章節 | 實作與重點 |
|---|---:|---|---|
| 09/15 | 01 | [AI 工程師為什麼要理解量子機器學習？](articles/day01/README.md) | 專案方向、研究問題與文獻入口 |
| 09/16 | 02 | [從 Bit 到 Qubit：量子狀態與量測機率](articles/day02/README.md) | 以數值表示量子狀態，從振幅計算量測機率 |
| 09/17 | 03 | [量子閘如何改變量子狀態？](articles/day03/README.md) | 用矩陣實作量子操作，保存測試與結果 |
| 09/18 | 04 | [疊加、糾纏與量測：建立第一個貝爾態](articles/day04/README.md) | 疊加、糾纏與不同量測方向的實驗 |
| 09/19 | 05 | [從一般機器學習流程看懂 QML 流程](articles/day05/README.md) | 從輸入資料到量測輸出的完整流程 |
| 09/20 | 06 | [CUDA-Q、CUDA、cuQuantum 有何不同？](articles/day06/README.md) | 工具分工、環境診斷與 CPU／GPU 基本驗證 |
| 09/21 | 07 | [第一個 CUDA-Q 量子核心程式：量子位元、參數與流程控制](articles/day07/README.md) | 量子位元配置、參數、迴圈與條件分支 |
| 09/22 | 08 | [`sample`、`run`、`observe`：同一電路的三種執行方式](articles/day08/README.md) | 比較抽樣計數、逐次回傳值與期望值 |
| 09/23 | 09 | [參數化量子電路：把資料與可訓練參數分開](articles/day09/README.md) | 分開資料與權重，驗證可調電路的反應 |
| 09/24 | 10 | [第一個 CUDA-Q 訓練迴圈：依預測誤差更新權重](articles/day10/README.md) | 以平均平方誤差與座標搜尋更新參數 |
| 09/25 | 11 | [一般數值資料如何轉成量子態？](articles/day11/README.md) | 只用訓練資料建立轉換規則，檢查編碼差異 |
| 09/26 | 12 | [角度編碼：角度範圍、旋轉軸與量測方式](articles/day12/README.md) | 比較資料轉成旋轉角度的範圍與方式 |
| 09/27 | 13 | [振幅編碼：從正規化向量到狀態準備電路](articles/day13/README.md) | 將資料轉成狀態振幅，檢查準備成本 |
| 09/28 | 14 | [資料編碼與可調電路：組成可訓練的量子模型](articles/day14/README.md) | 組合資料編碼、可調電路與訓練流程 |
| 09/29 | 15 | [第一個 CUDA-Q 量子分類器：訓練、評分與模型保存](articles/day15/README.md) | 二分類模型、保存權重、訓練曲線與分類邊界 |
| 09/30 | 16 | [量子神經網路與一般神經網路：參數、運算與輸出的差別](articles/day16/README.md) | 比較量子與傳統神經網路的參數和輸出 |
| 10/01 | 17 | [量子模型的梯度：從電路輸出到訓練損失](articles/day17/README.md) | 以參數位移、微小差分與矩陣導數核對梯度 |
| 10/02 | 18 | [一般機器學習與量子模型：固定條件下的比較](articles/day18/README.md) | 固定資料與評估預算，比較傳統與量子模型 |
| 10/03 | 19 | [混合神經網路：讓一般計算層與量子電路一起學習](articles/day19/README.md) | 結合傳統與量子運算，核對完整梯度 |
| 10/04 | 20 | [Iris 鳶尾花分類：一般、量子與混合模型的比較](articles/day20/README.md) | Iris 鳶尾花二分類、壓縮輸入與成本紀錄 |
| 10/05 | 21 | [特徵多、量子位元少：比較三種資料壓縮方式](articles/day21/README.md) | 比較特徵選擇、主成分分析與可訓練壓縮 |
| 10/06 | 22 | [資料重複編碼：讓資料再次進入量子電路](articles/day22/README.md) | 單次與重複編碼、分段輸入與電路成本 |
| 10/07 | 23 | [量子核方法：從資料相似度建立分類模型](articles/day23/README.md) | 從樣本相似度建立模型，核對完整矩陣 |
| 10/08 | 24 | [Barren Plateau：為什麼 QNN 學不動？](articles/day24/README.md) | 比較不同初始化、電路規模與目標的梯度 |
| 10/09 | 25 | [第二個真實資料集：Wine 分類實驗](articles/day25/README.md) | Wine 二分類、十三維對照與固定模型驗證 |
| 10/10 | 26 | [量子噪聲：模擬干擾如何改變量測與預測](articles/day26/README.md) | 噪聲通道、理論分布與有限次量測的差別 |
| 10/11 | 27 | [CPU 與 GPU：如何比較量子模擬效能](articles/day27/README.md) | 相同精度的單 GPU 計時與數值核對 |
| 10/12 | 28 | [從單 GPU 到多 GPU：容量與任務分工](articles/day28/README.md) | 單一狀態分散與獨立任務分工的假設模型 |
| 10/13 | 29 | [從模擬器到 QPU：量測預算與遠端工作追蹤](articles/day29/README.md) | 本地硬體後端預演、量測預算與工作追蹤 |
| 10/14 | 30 | [30 天 CUDA-Q 與 QML 實作回顧：成果、限制與後續方向](articles/day30/README.md) | 從保存資料核算指標，整理成果與後續實驗 |

## 六個階段的成果

| 範圍 | 成果與入口 |
|---|---|
| Day01–05 | 量子基礎、[實驗筆記本](notebooks/day05_quantum_fundamentals.ipynb) 與 [流程圖](figures/day05_pipeline.svg) |
| Day06–10 | 電路參數、量測與 [第一個訓練流程](articles/day10/README.md) |
| Day11–15 | 資料編碼與 [二分類模型](articles/day15/README.md) |
| Day16–20 | 模型比較與 [Iris 實驗](articles/day20/README.md) |
| Day21–25 | 輸入壓縮、核方法與 [Wine 實驗](articles/day25/README.md) |
| Day26–30 | 噪聲、效能、硬體預演與 [證據報告](results/day30/README.md) |

各階段的銜接與用語說明見 [學習路線](ROADMAP.md)，文獻原名、用途與版本紀錄見 [參考索引](REFERENCES.md)。

## 核對保存結果

以下指令不需要 GPU，會檢查保存資料與彙整結果的一致性：

```bash
python3 articles/day30/evidence.py --check
python3 -m unittest discover -s articles/day30 -p 'test_*.py'
```

這不會重跑所有歷史訓練。完整重跑方式見各章文章及對應的 `requirements-dayXX.txt` 套件清單。

## 專案目錄

```text
.
├── README.md             # 專案說明與章節目錄
├── ROADMAP.md            # 階段安排與學習重點
├── REFERENCES.md         # 文獻、官方資源與版本紀錄
├── articles/             # 每日文章、Python 程式與測試
├── notebooks/            # 結合文字、程式與輸出的實驗筆記本
├── data/                 # 保存的資料、來源與授權
├── figures/              # 文章圖表
└── results/              # 結果與執行環境紀錄
```

## 官方工具入口

- [CUDA-Q 入門](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)：第一個量子程式。
- [CUDA-Q 範例](https://nvidia.github.io/cuda-quantum/latest/using/examples/examples.html)：各種電路與執行方式。
- [CUDA-Q 本地安裝](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html)：安裝條件與環境設定。
- [cuQuantum 文件](https://docs.nvidia.com/cuda/cuquantum/latest/index.html)：加速量子模擬的 GPU 函式庫，與編寫量子程式的 CUDA-Q 分工不同。
