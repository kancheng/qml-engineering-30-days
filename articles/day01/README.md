# Day 01｜AI 工程師為什麼要理解量子機器學習？

量子機器學習（Quantum Machine Learning，QML）研究如何將量子計算用於機器學習任務，例如根據資料判斷類別。本系列從一般數值資料出發：先把資料轉成量子電路能處理的形式，再透過電路運算與量測取得預測所需的數值；一般電腦則負責整理資料與調整模型。本章介紹這個分工，以及學習過程中最重要的判斷方式：**程式能執行、預測較準、完成任務的成本較低，是三件需要分別驗證的事。** 因此，實作量子模型的同時，也需要保留解決相同任務的一般機器學習方法作為比較基準，才能知道結果的意義。

Quantum Machine Learning (QML) studies how quantum computation can be used for machine learning tasks, such as classifying data. This series starts with ordinary numerical data: the data is converted into a form a quantum circuit can process, and measurements produce numerical outputs for predictions. A conventional computer prepares the data and adjusts the model. This chapter introduces that division of work and a central distinction: **running successfully, making better predictions, and completing a task at lower cost are three separate outcomes that require separate evidence.** Comparing a quantum model with a conventional machine learning method solving the same task helps establish what the results mean.

---

人工智慧（Artificial Intelligence，AI）工程的日常工作，是把資料、模型與運算資源組合成能完成任務的系統。加入量子計算後，仍然需要回答熟悉的問題：資料如何處理、模型如何學習、結果如何驗證，以及一次實驗要花多少時間。

第一天先建立整體流程，再逐步認識其中的量子概念。

## 1. QML 要解決什麼問題？

機器學習（Machine Learning，ML）利用資料調整模型，使模型能對新的資料做出預測。例如，根據花瓣與花萼的長度判斷花的種類。

QML 探討能否讓量子計算參與這個過程。量子計算使用量子位元（qubit）描述與處理資訊；量子位元的狀態除了量測得到 0 或 1 的機率，還包含會影響後續運算的相位關係。這些概念會從 Day 02 開始展開。

本系列的路線是：

```text
一般數值資料 → 量子狀態的準備 → 量子電路運算 → 量測結果
             → 計算預測誤差 → 調整參數 → 再次執行
```

這條流程可以先在一般電腦上的模擬器執行。模擬器是用數值運算模仿量子系統的程式，並不是真正的量子處理器。

## 2. 從熟悉的機器學習流程開始

一般機器學習可以拆成六個步驟：

| 步驟 | 名稱 | 用途 |
|---|---|---|
| 收集樣本 | 資料集（dataset） | 整理用來學習與評估的資料，例如花的尺寸與種類 |
| 整理輸入 | 特徵處理（feature engineering） | 選擇或轉換提供給模型的資訊，例如將尺寸換成一致的尺度 |
| 執行計算 | 模型（model） | 依輸入與可調整的參數產生輸出 |
| 取得答案 | 預測（prediction） | 輸出類別、數值或各類別的預測機率 |
| 衡量誤差 | 損失函數（loss function） | 將預測與正確答案的差距轉成可比較的數值 |
| 調整模型 | 最佳化器（optimizer） | 根據損失或其變化方向更新參數，嘗試降低誤差 |

QML 可以將其中一部分模型計算交給量子電路。量子電路（quantum circuit）是一連串作用在量子位元上的操作；其中的基本操作稱為量子閘（quantum gate），作用類似組成運算流程的小步驟。

將數值資料轉成量子狀態的過程稱為資料編碼（encoding）。量子狀態（quantum state）是用來計算量測機率與後續演化的數學描述。完成電路運算後，量測（measurement）產生一般電腦可以讀取的結果；多次量測的統計可用來估計模型輸出。

若電路含有可調整的角度等參數，就稱為參數化量子電路（parameterized quantum circuit）。一般電腦負責調整參數，量子電路負責部分運算，這種分工稱為量子與經典混合計算（hybrid quantum-classical computing）。此處的「經典」是指一般電腦採用的計算方式。

## 3. 量子計算、QML 與 Quantum AI 的關係

- **量子計算（quantum computing）**：研究如何利用量子系統處理資訊，範圍包含基本操作、演算法與硬體。
- **量子機器學習（QML）**：聚焦量子計算如何建立或協助機器學習方法。
- **量子人工智慧（Quantum AI）**：較寬泛的用語，使用時需要確認具體指的是機器學習、搜尋、最佳化，還是其他 AI 任務。

本系列以一般數值資料的機器學習為主，逐步補上實作所需的量子計算基礎。

## 4. 量子模型多了哪些需要考慮的事？

一般神經網路（neural network）由多層可調整的計算組成，每層將輸入轉換後交給下一層。深度學習（deep learning）使用多層神經網路學習資料的表示方式。量子模型則可能先將輸入編碼，再經過量子電路，最後由量測取得輸出。

這個改變帶來幾項額外成本：

| 概念 | 直觀說明 | 為什麼會影響實驗？ |
|---|---|---|
| 量子位元數 | 電路使用多少個量子位元 | 影響可表示的狀態空間，也影響模擬所需記憶體 |
| 硬體連接性 | 哪些量子位元能直接一起操作 | 原本的一步運算可能需要拆成更多步驟 |
| 電路深度（circuit depth） | 考慮可同時執行的操作後，仍需依序完成的層數 | 影響執行成本，也可能增加受到雜訊干擾的機會 |
| 狀態準備（state preparation） | 從初始狀態建立所需量子狀態的操作 | 將資料放入電路本身就需要成本 |
| 量測次數（shots） | 重複準備、執行並量測電路的次數 | 次數越多通常能降低抽樣波動，但也增加成本 |
| 雜訊（noise） | 使實際演化或讀出偏離理想情況的干擾 | 可能改變量測分布，增加 shots 不會自動消除這種偏差 |
| 梯度（gradient） | 描述損失隨各個參數改變的方向與幅度 | 求梯度可能需要額外執行多次電路 |
| 量子處理器（Quantum Processing Unit，QPU） | 實際執行量子操作的硬體 | 與模擬器的條件不同，還需考慮裝置限制與遠端執行流程 |

## 5. 這個系列會實作哪些模型？

第一條路是**變分量子分類器（Variational Quantum Classifier，VQC）**。「變分」在這裡表示反覆調整電路參數，嘗試降低預測誤差；「分類器」則是判斷資料所屬類別的模型。

第二條路是**量子核方法（quantum kernel method）**。核函數可以理解成比較兩筆資料相似程度的規則；量子核使用量子狀態之間的關係建立這種比較，再交給一般電腦上的學習方法處理。

第三條路是**混合模型（hybrid model）**：先由一般神經網路整理輸入，再讓量子電路處理部分資訊，最後將輸出轉成預測。含有可訓練量子電路的模型也常稱為量子神經網路（Quantum Neural Network，QNN），但名稱不代表運算方式與一般神經網路相同。

實驗也會比較資料編碼與電路結構。角度編碼（angle encoding）將數值用作量子操作的角度；振幅編碼（amplitude encoding）則將正規化後的資料放進量子狀態的振幅，也就是描述狀態的複數係數。振幅決定量測機率，但不是可以直接一次讀出的資料表。這些方法的限制會在後續章節展開。

固定的電路排列方式常稱為電路模板（ansatz）；將輸入映射到量子狀態的規則則可稱為特徵映射（feature map）。名稱不同，關心的都是資料如何進入模型，以及模型能產生哪些輸出。

## 6. 如何判斷結果有沒有意義？

比較基準（baseline）是用來回答同一問題的參考方法。例如，量子分類器可以和多層感知器（Multilayer Perceptron，MLP）比較；MLP 是由多層一般神經網路計算組成的模型。

公平比較需要記錄資料切分、輸入資訊、參數量與訓練成本。資料切分是將樣本分成訓練、驗證與測試用途：訓練資料用來調整模型，驗證資料協助選擇設定，測試資料用來評估選定模型。

也需要保留隨機種子（seed），也就是控制程式產生隨機序列的起始設定，方便在相同環境重做實驗。不同種子可能得到不同結果，因此單次的準確率（accuracy，即分類正確的比例）不足以描述模型表現。

量子優勢（quantum advantage）需要說明量子方法在什麼任務、品質要求與成本條件下優於適當的經典方法。寫出電路或得到較高的一次準確率，都還不足以支持這個結論。

2017 年的 QML 綜述整理了這個領域的研究方向與挑戰。[R1] 綜述是彙整多項研究的文章，適合建立全貌。2022 年的綜述進一步討論資料、可訓練性與量子優勢的條件；可訓練性指模型是否能在可負擔的資源下有效調整參數。[R2]

## 7. 模擬變快與模型變好是不同問題

圖形處理器（Graphics Processing Unit，GPU）擅長同時執行大量適合平行處理的運算，因此可以用來加速某些量子模擬。中央處理器（Central Processing Unit，CPU）則是一般電腦執行程式與控制流程的主要處理器。

CPU 和 GPU 都是經典硬體。GPU 更快地模擬量子電路，表示某個模擬工作在指定設定下更快，不能直接推論真實量子電腦更快，也不能推論模型預測更準。

效能評測（benchmark）需要固定工作內容與計時範圍。例如，單次預測的時間與包含反覆調整參數的完整訓練時間，應分別記錄。

## 8. CUDA-Q 在流程中負責什麼？

CUDA-Q 是本系列使用的量子程式開發工具，用來描述電路、執行模擬與取得量測結果。量子核心程式（quantum kernel）指描述量子運算的程式區塊；這裡的 kernel 與前面的「核函數」是不同概念。

取樣（sampling）是重複執行電路取得量測結果。期望值（expectation value）則是依各種結果的機率計算的平均值；在有限次量測中，可以用樣本平均估計。

執行後端（backend）指定程式交給哪個模擬器或硬體執行。切換後端方便沿用電路描述，但仍需重新確認精度、量測方式與裝置限制。

### 設備分工

| 設備 | 本系列規劃用途 |
|---|---|
| Surface Pro 7 | 寫作、文獻整理，以及使用 Python 數值運算套件 NumPy 執行小型計算 |
| Ubuntu 主機與 RTX 3060 顯示卡 | 建立 CUDA-Q 環境，執行 GPU 模擬與效能實驗 |

Ubuntu 是 Linux 作業系統的一種。實際能否執行 GPU 模擬，仍需確認驅動程式與軟體版本；Day 06 會記錄環境檢查結果。安裝文件入口見 [D2]。

## 9. 30 天的六個階段

1. **Day 01–05：量子計算基礎。** 認識量子位元、基本操作與量測，建立可以用數值核對的小電路。
2. **Day 06–10：CUDA-Q 與參數調整。** 將電路執行、誤差計算與參數更新接成反覆運作的流程。
3. **Day 11–15：資料編碼與分類。** 將一般資料放入量子模型，觀察模型如何學習分類。
4. **Day 16–20：梯度與模型比較。** 比較一般神經網路、量子模型與混合模型的品質及成本。
5. **Day 21–25：資料表示與訓練限制。** 研究降維、量子核與梯度變小的問題。降維是將輸入轉成較少的特徵；貧瘠高原（barren plateau）則描述特定條件下梯度隨規模快速縮小、使訓練困難的現象。
6. **Day 26–30：雜訊、效能與硬體準備。** 區分模擬、模型估算與實際執行證據，整理後續方向。

完整題目與成果見 [30 天學習路線](../../ROADMAP.md)。

## 10. 實驗紀錄如何累積？

所有文章、程式與結果都放在同一個專案儲存庫（repository），也就是集中保存專案檔案與修改紀錄的地方。

重要實驗保留軟體環境、隨機種子、執行後端、耗時與評估指標。評估指標（metrics）是用來描述表現的數值，例如準確率或預測誤差。這些紀錄讓結果能回查，也方便在相同條件下重新執行。

閱讀時可以區分三種內容：用來建立直覺的簡化解釋、指定條件下可重現的實驗結果，以及需要更完整證據支持的研究結論。程式與檔案存在，並不代表所有結論都已獲得驗證。

## 11. 研究資源怎麼讀？

基礎概念可以從 Nielsen 與 Chuang 的教材查起，集中理解狀態、量測與量子閘。[F1] 接著閱讀 QML 綜述，了解不同方法在解決什麼問題，以及各自受到哪些限制。[R1][R2]

若聚焦有正確答案可供訓練的監督式學習（supervised learning），Schuld 與 Petruccione 的專書可協助串起資料編碼、訓練與推論；推論（inference）就是使用已選定的模型產生預測。[R4] 進入特定主題後，再閱讀提出方法或分析的原始論文，例如資料編碼與模型表達能力的研究。[P1] 表達能力指模型結構能表示哪些輸入與輸出關係，不等於訓練一定能找到好的參數。

完整書目與使用目的見 [文獻與資源索引](../../REFERENCES.md)。閱讀文獻時，也要區分預印本與正式審查後的出版版本：arXiv 是常見的預印本平台，文章上傳至該平台不代表已通過同行審查；同行審查是由同領域研究者評閱研究的方法與論述。

## 12. 本日文獻

- [R1] Biamonte et al., “Quantum machine learning,” *Nature* 549, 195–202 (2017), [DOI](https://doi.org/10.1038/nature23474).
- [R2] Cerezo et al., “Challenges and opportunities in quantum machine learning,” *Nature Computational Science* 2, 567–576 (2022), [DOI](https://doi.org/10.1038/s43588-022-00311-3).
- [R4] Schuld and Petruccione, *Supervised Learning with Quantum Computers*, Springer (2018), [DOI](https://doi.org/10.1007/978-3-319-96424-9).
- [F1] Nielsen and Chuang, *Quantum Computation and Quantum Information*, 10th Anniversary Edition, Cambridge University Press (2010), [publisher page](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE).
- [P1] Schuld, Sweke, and Meyer, “Effect of data encoding on the expressive power of variational quantum-machine-learning models,” *Physical Review A* 103, 032430 (2021), [DOI](https://doi.org/10.1103/PhysRevA.103.032430).
- [D2] [NVIDIA CUDA-Q Local Installation](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html), accessed 2026-09-01.

## 13. 下一篇

[Day 02｜從 Bit 到 Qubit：工程師需要懂多少量子力學？](../day02/README.md)

下一篇從只能取 0 或 1 的位元（bit）開始，介紹量子位元如何表示狀態，以及為什麼量測結果需要用機率描述，再逐步建立閱讀量子電路所需的數學符號。
