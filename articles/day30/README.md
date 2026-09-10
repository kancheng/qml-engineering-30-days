# Day 30｜30 天 CUDA-Q 與 QML 實作回顧：成果、限制與後續方向

[Day29](../day29/README.md) 透過本地理想抽樣、硬體後端的本地預演與合成雜訊，核對小型電路的輸出，並整理尚未提交的 QPU 工作規格。Day30 將這些驗證連同前面各章的模型、資料與效能紀錄一起回顧，整理系列成果、結論的適用範圍，以及後續需要補足的實驗。

**每項實驗結果都需要連同資料、模型與執行條件一起判讀。** 完成量子電路與訓練程式之後，還需要確認每項結果回答了什麼問題，才能判斷下一輪實驗應該改進資料、模型還是執行方式。本章沿著量子基礎、CUDA-Q 電路、QML 建模與硬體執行準備，整理已建立的工程流程，並從保存的預測與計時資料重新計算指標，讓結論能追溯到原始紀錄。不同證據需要分開解讀：模型比較需要檢查輸入資訊、資料切分與訓練預算；分類準確率之外，也要觀察衡量預測機率誤差的 Brier score；GPU 加快特定電路的模擬，不代表完整 QML 訓練已加速，也不等於量子方法優於經典方法。含雜訊模擬、多 GPU 假設模型與 QPU 本地預演，也各有不同的驗證範圍，不能合併當成真實硬體成果。本章的實作是核對既有資料與彙整證據，並未重跑全部歷史實驗。現有小型實驗尚不足以證明量子優勢。下一輪需要新的保留資料、公平對照、完整成本紀錄，以及事先訂定的判讀標準，才能進一步評估模型的實際用途。

[Day29](../day29/README.md) checked a small circuit through local ideal sampling, hardware-target emulation, and synthetic noise, and prepared an unsubmitted QPU job specification. Day30 reviews those checks alongside the earlier model, data, and performance records to organize the series' outcomes, the scope of its conclusions, and the experiments still needed.

**Experimental results need to be interpreted alongside their data, models, and execution conditions.** After building circuits and training code, identifying the question each result answers helps determine whether the next experiment should improve the data, model, or execution method. The chapter reviews the engineering workflow from quantum fundamentals and CUDA-Q circuits to QML modeling and hardware preparation, recalculating metrics from saved predictions and timings so that conclusions remain connected to their records. The key is to interpret different types of evidence separately. Model comparisons require attention to input information, data splits, and training budgets; classification accuracy should be considered alongside the Brier score, which measures probability prediction error. Faster GPU simulation of a particular circuit establishes neither faster complete QML training nor an advantage of quantum methods over classical methods. Noisy simulation, multi-GPU scenario models, and local QPU emulation likewise have distinct validation scopes and do not collectively establish physical-hardware results. This chapter checks existing data and consolidates evidence rather than rerunning every historical experiment. The small experiments do not establish quantum advantage. Further evaluation requires new held-out data, fair comparisons, complete cost records, and interpretation criteria defined in advance.

---

**本章整理 30 天 CUDA-Q 與 QML 實作的成果、適用範圍與後續實驗方向。系列已建立可訓練、可核對、可重現的工程流程；現有資料與實驗條件尚不足以證明量子優勢。**

Day29 以本地預演走到硬體後端的編譯與抽樣介面，但沒有提交 實體 QPU。今天把整個專案收束成一份可以回查的結論：哪些數字來自實測、哪些來自模型、哪些問題仍未回答。

本日實作是[證據彙整程式](evidence.py)與[測試](test_evidence.py)。它從既有 JSON 重新核算分類指標、模型選擇與計時比例，產生[系列證據報告](../../results/day30/README.md)及[來源雜湊清單](../../results/day30/evidence.json)。本章使用保存的候選與選定結果，未追加訓練。

CUDA-Q 是執行量子程式的工具，QML 是量子機器學習。CPU（中央處理器）與 GPU（圖形處理器）在本系列用來模擬量子電路；QPU 則是真實量子處理器。後端是實際執行計算的工具，本地預演只驗證編譯與模擬流程，並非硬體執行。量子優勢需要在明確任務與條件下，證明量子方法相對於適當傳統方法的優勢，不能只由 GPU 模擬加速推論。

## 1. 30 天完成的是哪條工程路徑？

| 階段 | 實作內容 | 現在能做的事 |
|---|---|---|
| Day01–05 | NumPy 量子閘、Bell 狀態、基礎實驗筆記本 | 從狀態與量測核對電路 |
| Day06–10 | CUDA-Q 量子電路程式、sample／observe、反覆更新參數的訓練流程 | 把參數、量測與最佳化器串起來 |
| Day11–15 | 僅使用訓練資料的前處理、資料編碼、XOR 分類器 | 保存訓練曲線、模型保存檔與分類邊界 |
| Day16–20 | 梯度、MLP／VQC／Hybrid、Iris 二分類 | 在固定資料與預算下比較模型 |
| Day21–25 | 壓縮層、資料重複編碼、核函數、可訓練性、Wine | 檢查表達方式、成本與資料任務的關係 |
| Day26–30 | 噪聲、單 GPU 計時、多 GPU 模型、QPU 預演、證據彙整 | 分辨模型假設、實測範圍與部署缺口 |

NumPy 是 Python 數值運算套件。量子閘是改變量子狀態的基本操作；Bell 態具有糾纏，表示兩個量子位元無法各自獨立描述。`sample` 回傳量測抽樣計數，`observe` 計算量測結果依機率加權的平均，也就是期望值。XOR 任務在兩個輸入不同時標為 1、相同時標為 0；分類邊界則是模型區分不同類別的分界。

資料編碼將輸入轉成量子操作，壓縮層將多個特徵合併成較少數值，資料重複編碼讓同一輸入在電路中再次影響操作。最佳化器是根據損失或梯度決定如何更新權重的演算法；可訓練性描述這種更新是否容易有效改善目標。

關鍵轉變是開始能追問：「這個結果是怎麼得到的？」每個準確率都可以追溯到資料切分、前處理規則、參數選擇、後端與輸出格式與意義。

完整 30 篇文章與每日可執行檔清單在[彙整報告](../../results/day30/README.md)。檔案齊全代表程式與紀錄存在，不表示今天重跑過全部歷史實驗。

## 2. 相同輸入下的模型比較

Day20 的 Iris 二分類使用 99 筆去重資料，每組資料切分的測試資料只有 21 筆；Day25 的 Wine 二分類使用 119 筆，每組資料切分的測試資料為 26 筆。兩天各有兩組資料切分，不能把它們當大量獨立重複。詳見[Day20 報告](../../results/day20/README.md)與[Day25 報告](../../results/day25/README.md)。

訓練資料用來建立前處理規則與更新權重，驗證資料用來選擇候選模型，測試資料在選定後評分。隨機種子控制資料打亂或起始權重的產生，方便重現同一組設定。

PCA（主成分分析）將原始特徵組合成較少的新座標，PCA2 表示保留兩個主成分。MLP（多層感知器）是傳統神經網路；VQC（變分量子分類器）透過調整電路角度學習；Hybrid 是結合傳統運算與量子電路的混合模型。

Brier 分數是預測值與 0／1 標籤的平均平方誤差，越小越接近標籤。準確率只看分類是否正確；混淆矩陣則分別記錄各真實類別被判成哪些類別。

下面只比較同一天、同資料切分、同 PCA2 輸入的 MLP／VQC／Hybrid。數字是選定模型的測試資料 Brier，越小越好：

| 任務／資料切分 | MLP | VQC | Hybrid |
|---|---:|---:|---:|
| Iris 2028 | 0.070884 | 0.105396 | 0.076867 |
| Iris 2029 | 0.088543 | 0.102869 | 0.081691 |
| Wine 2030 | 0.028478 | 0.032404 | 0.034820 |
| Wine 2031 | 0.041693 | 0.057050 | 0.041495 |

VQC 在這四個配對的 Brier 都比 MLP 高。Hybrid 有兩個配對較低、兩個較高；Wine 2031 差值只有約−0.000198。這些是有限資料上的描述，不足以做穩定優勢或顯著性主張。[重新核算表](../../results/day30/README.md)

![相同輸入的 Brier 差值](../../results/day30/paired_brier.png)

顯著性主張需要評估差異是否超出隨機波動，本章表格只是保存結果的比較，未完成這類推論。目標函數是訓練時希望降低的誤差，收斂則表示繼續更新後結果趨於穩定。

不同模型有不同參數量；相同目標函數評估預算也不等於相同更新步數、時間或充分收斂。因此這份對照支持「在這套固定協定下得到這些結果」，不能推出某個模型族已達最佳表現。

Logistic13 將十三個標準化特徵加權後，以 sigmoid 函數轉成 0 與 1 之間的值。標準化先減去訓練平均值，再除以描述分散程度的標準差。電路深度是必須依序完成的操作層數。

Day25 還保留看到全部 13 維特徵的 logistic13：Wine 2031 的測試資料 Brier 為 0.032224。它是實用完整輸入比較基準，與 PCA2 模型的輸入資訊不同，不能用來單獨歸因量子層效果。這項對照也說明，分析模型表現時，應先檢查降維是否丟失任務所需的資訊，再評估電路深度。

## 3. 量子核方法的比較條件

量子特徵映射將每筆輸入轉成量子狀態，量子核函數計算兩筆狀態的相似度。相似度矩陣保存樣本兩兩比較的數值；核嶺迴歸利用這張矩陣求出樣本係數，再加權組合成預測。λ 是正則化強度，用來控制限制預測函數大小的懲罰，降低過度迎合訓練資料的風險。

Day23 改用固定量子特徵映射與傳統核嶺迴歸。它沒有訓練量子權重，仍然需要檢查相似度矩陣、λ選擇及完整前處理。

線性核依輸入內積衡量相似度；本章引用的版本另含常數項。內積是將對應特徵相乘後相加。RBF（徑向基底函數）則依資料間距離計算相似度，距離越遠，相似度越低。

兩個資料切分的量子核的測試 Brier 分別為 0.066907、0.071703；線性核 為 0.041345、0.047579；RBF 為 0.067819、0.028753。量子核方法在第一組略低於 RBF，第二組較高；線性核 在兩組都較低。[Day23 報告](../../results/day23/README.md)

這三者使用相同四維縮放輸入；Day20 則是 PCA2。即使來自 Iris，也不能把跨日結果合併成隔離單一因素的比較。核嶺迴歸將超出 [0, 1] 的分數截斷到邊界，也不保證機率已校準；例如輸出 0.8，不一定對應約八成的實際發生率。

本系列完成兩種 QML 建模路徑：可訓練電路與固定量子核函數。兩條路都需要傳統模型比較基準、資料切分與完整成本紀錄。

## 4. GPU 讓模擬變快，回答的是哪個問題？

fp64 是以 64 位元保存實數的雙精度格式。暖機是在正式計時前先執行數次，降低首次準備工作的影響；延遲是呼叫開始到結果可用的等待時間。以下速度比均為 CPU 時間中位數除以 GPU 時間中位數，中位數是多次結果排序後的中間值。

Day27 依序量測相同電路、相同權重與 fp64 條件下的 CPU／GPU 暖機後延遲。小電路 4 量子位元／4 區塊的 CPU/GPU 約 0.613，當時 CPU 較快；16 量子位元／12 區塊約 1112.885，當時 GPU 較快。[Day27 原始報告](../../results/day27/README.md)

OpenMP 是 CPU 平行運算機制，此處只用一個執行緒，也就是一個可排程的工作單位。RY、RZ 分別繞 Y 與 Z 軸旋轉，CZ 是受控相位閘；全域 Z 量測將所有位元的 ±1 記分相乘後取平均。

這個比值有明確的比較條件：`qpp-cpu`、單一 OpenMP 執行緒、特定 RY／RZ／CZ 電路、全域 Z 量測量的直接期望值計算。它不是對最佳傳統機器學習系統的比較，也不是 GPU 普遍快千倍。Day30 重新核算了全部 8 組 fp64 配對，而非只留下最大的比例。

還有一個容易混淆的事實：**Day20／25 共 28 次訓練都是 NumPy CPU 參考計算；這兩天的 CUDA-Q 訓練呼叫次數皆為 0。** CUDA-Q CPU／GPU 核對的是凍結模型的電路求值。不能拿 NumPy 訓練時間除以 GPU 求值時間，稱為 QML 訓練加速。

早期 Day10 確實執行了 CUDA-Q 反覆更新參數的訓練流程；兩件事並不衝突。每個實驗都需要分別記錄訓練使用的計算工具，才能正確解讀訓練與驗證的成本。

## 5. 噪聲下的準確率與機率誤差

凍結模型表示保持已訓練的權重與資料處理規則不變。位元翻轉以機率 p 交換 0 與 1；q0 是第 0 個量子位元。ZZ 對兩位元結果相同記 +1、不同記 −1，再取平均。

Day26 對凍結 Wine VQC 的末端 q0 加入位元翻轉。資料切分 2030 在 p=0 時 Brier 約 0.032404，p=.3 時約 0.124883；準確率卻同為 96.15%。[Day26 報告](../../results/day26/README.md)

原因是這個特定噪聲位置將 ZZ 期望值乘上正的縮放係數，分數往 0.5 靠近但未必越過分類門檻。分類標籤可以不變，預測品質仍然變差。只看準確率，會漏掉這種變化。

一次 shot 是一次狀態準備與量測；抽樣變異是有限次量測造成的統計波動。校準資料則記錄真實裝置當時的操作與誤差特性。

Day29 用另一個可解析電路確認，更多 shots 會降低抽樣變異，卻不會將合成噪聲下的 ZZ 期望值 0.84 恢復成理想值 1。三模式的 90 組計數全部在本地執行，這不是硬體校準資料。[Day29 報告](../../results/day29/README.md)

梯度描述參數小幅改動時目標函數的變化率；初始化是設定起始權重。

Day24 也讓「梯度小」這句話更具體：需要一起看初始化、量子位元數、深度與目標涵蓋的量子位元範圍。那一天的 768 個初始化、1,536 個單參數梯度是小規模診斷，沒有證明所有量子神經網路（QNN）在規模持續增加時是否仍容易訓練，更不能直接推論分類準確率。[Day24 報告](../../results/day24/README.md)

## 6. 硬體驗證的範圍與待補證據

| 問題 | 本系列證據 | 尚缺的證據 |
|---|---|---|
| 單 GPU 是否能加速此模擬？ | Day27 特定設定實測 | 其他 CPU 調校、電路與規模 |
| 多 GPU 能提高多少容量／速度？ | Day28 可重現假設模型 | 真實多卡、互連、分散與最高記憶體用量量測 |
| 硬體後端能否編譯此小電路？ | Day29 IonQ 本地預演 | 遠端裝置接收與實際執行 |
| 真實 QPU 是否有實用 QML 優勢？ | 本系列未測量 | 同任務品質、端到端成本與可重複硬體結果 |

互連是 GPU 之間交換資料的連線，分散運算則將同一工作拆給多個裝置。端到端成本包括資料處理、訓練、量測、等待與取得結果的整段流程。

Day28 的 16 組容量、12 組延遲情境有公式與比較基準來源，但沒有多 GPU 實測。Day29 的 2048-shot 工作規格仍為`not_submitted`，實體 QPU 工作數為 0。排隊與費用是未知值，不是零。

因此，本系列既沒有證明量子優勢，也沒有證明 QML 不可能有用。它將下一步需要驗證的問題縮小到能實際設計實驗的程度。

## 7. 從保存資料重新核對結論：讓結論也能重跑

[evidence.py](evidence.py)不用啟動 CUDA-Q，直接讀保存的資料並執行以下核對：

1. 對 Day20／23／25 共 20 個選定模型，依標籤與保存預測重算 60 組資料分組 Brier、準確率與混淆矩陣；檢查隨機種子／λ由驗證資料選定、參數與候選模型一致。
2. 核對每組資料切分內的資料列編號沒有交疊；重算 Day26 共 18 組 Wine 噪聲評估指標。
3. 由 Day27 原始 7 次暖機後時間重算 8 組 fp64 中位數與比值，確認實驗設定、權重摘要與輸出誤差。
4. 檢查 Day28 引用的單 GPU 比較基準 沒有過期；彙整 Day29 的本地執行、計數總 shots 與未提交狀態。
5. 保存 26 份來源 JSON 的 SHA-256，建立 30 篇文章與 Day03–30 可執行檔索引。

JSON 是保存結構化資料的文字格式；SHA-256 是由檔案內容計算的摘要，用來確認來源版本是否一致。

輸出 [evidence.json](../../results/day30/evidence.json)、[報告](../../results/day30/README.md)與 Brier 差值圖。`--check`會比較當前來源、彙整 JSON 及報告，來源或結果變動時要求重新產生，避免舊結論悄悄沿用新實驗。

`.venv` 是專案獨立保存套件的虛擬環境，Matplotlib 是繪圖套件；`MPLCONFIGDIR` 指定其設定與快取目錄。

```bash
# 產生彙整與圖表；沿用既有環境，不需 GPU 或帳號。
MPLCONFIGDIR=/tmp/day30-matplotlib .venv/bin/python articles/day30/evidence.py

# 純標準函式庫核對，沒有 GPU 或 Matplotlib 也能執行。
python3 articles/day30/evidence.py --check

# 彙整邏輯、損壞資料與保存結果測試。
python3 -m unittest discover -s articles/day30 -p 'test_*.py'
```

依賴入口：[requirements-day30.txt](../../requirements-day30.txt)。重跑只覆寫`results/day30/`的本日產物。這是保存資料的一致性核對，不是重新計算所有模型電路求值、證明全部前處理無誤或重跑 30 天 GPU 測試；舊實驗的時間戳記與限制仍保留在原日報告。

## 8. 下一輪研究：先訂什麼結果會改變判斷

以下是本專案的後續實驗規格，尚未執行。各項實驗需要先訂定足以影響模型選擇的品質與成本條件。

| 問題 | 下一個實驗 | 事先約定的判讀方式 |
|---|---|---|
| 模型差異是否穩定？ | 用新的評估設計與保留資料，比較相同輸入的傳統模型／VQC／Hybrid | 先定最小實用 Brier 改善幅度、重複規格與成本上限，再看測試資料 |
| 瓶頸是不是資訊壓縮？ | 配對完整輸入比較基準、PCA、可訓練壓縮層與資料重複編碼 | 分開報告表達方式、輸入資訊與最佳化器預算，不只比參數量 |
| 有限 shots 會否改變選模？ | 凍結候選模型，對固定量測次數預算重複評估驗證資料 | 檢查排名與預測不確定性；不以測試資料反覆選 shot 數 |
| GPU 收益能否延伸？ | 固定精度與電路，增加合理 CPU 執行緒對照和記憶體量測 | 保留全部設定與原始時間；容量、延遲、吞吐量分別報告 |
| QPU 硬體落差有多大？ | 選定可用裝置後，先執行 Day29 的小型已核對工作 | 保存工作、校準、shots、編譯、排隊、費用，再決定是否搬模型 |

容量表示能容納多大的狀態，延遲表示單次工作等待多久，吞吐量則是每秒完成多少工作。三者需要分開量測。保留資料是建模與選參數時不查看、留待最後評估的資料。

目前 Iris／Wine 結果已在開發過程中被反覆看過；即使每一天遵守驗證資料選參數，也不應把這些測試資料當成永遠未見的資料。下一輪若要做更強主張，需要新的保留評估設計。

如果 QML 沒有達到事先定義的實用品質或成本條件，就保留較簡單的傳統模型方案。若在充分對照後出現可重複差異，再投入更大的模擬或硬體預算。這是一個能被結果改變的決策流程。

## 9. 系列成果與後續延伸

QML 的實作流程涵蓋資料、數值計算、量測、權重更新與執行成本。將這些資訊保存在同一套可檢查的紀錄中，才能追查結果如何產生。

本系列累積了量子模型實作、數值驗證、比較實驗與成本紀錄，並標示各項結果的適用範圍。後續若要評估 QML 在特定系統中的用途，仍需依任務需求設計完整的品質與成本對照。

這些程式、資料與實驗紀錄，構成後續擴充資料集、比較模型與進行硬體驗證的基礎。

## 來源與閱讀入口

本章數值取自專案保存的實驗紀錄。數值入口為[Day20](../../results/day20/README.md)、[Day23](../../results/day23/README.md)、[Day24](../../results/day24/README.md)、[Day25](../../results/day25/README.md)、[Day26](../../results/day26/README.md)、[Day27](../../results/day27/README.md)、[Day28](../../results/day28/README.md)、[Day29](../../results/day29/README.md)。

方法與官方後端文件見 [REFERENCES.md](../../REFERENCES.md)，實作行為以各章固定的軟體版本為準。[完整 30 天目錄](../../README.md)、[Roadmap](../../ROADMAP.md)。

## 延伸研究

[N1] Shreeya Sanjeev Gokhale et al. “A review of quantum machine learning algorithms, applications, and emerging advantages.” Discover Computing 29, 226 (2026)；綜述論文。[原始來源](https://doi.org/10.1007/s10791-026-10085-1)；[完整書目](../../REFERENCES.md#n1)。

本章彙整品質、成本與限制；這篇綜述可銜接後續研究方向。外部文獻提供比較脈絡，本章數值仍以專案保存的實驗紀錄為準。
