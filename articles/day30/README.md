# Day 30｜寫了30天CUDA-Q與QML，我還相信QML嗎？

**我仍願意研究QML，也更清楚自己需要什麼證據才會採用它。這30天完成了可訓練、可核對、可重現的工程流程；在本系列的資料與實驗條件下，我們還沒有證明量子優勢。**

Day29以本地emulation走到硬體target的編譯與抽樣介面，但沒有提交physical QPU。今天把整個專案收束成一份可以回查的結論：哪些數字來自實測、哪些來自模型、哪些問題仍未回答。

本日實作是[證據彙整程式](evidence.py)與[測試](test_evidence.py)。它從既有JSON重新核算分類指標、模型選擇與計時比例，產生[系列證據報告](../../results/day30/README.md)及[來源雜湊清單](../../results/day30/evidence.json)。今天沒有追加訓練或挑選更好看的seed。

## 1. 30天完成的是哪條工程路徑？

| 階段 | 交付成果 | 現在能做的事 |
|---|---|---|
| Day01–05 | NumPy gate、Bell state、fundamentals notebook | 從state與measurement核對電路 |
| Day06–10 | CUDA-Q kernel、sample／observe、optimization loop | 把參數、量測與optimizer串起來 |
| Day11–15 | train-only preprocessing、encoding、XOR classifier | 保存訓練曲線、checkpoint與decision boundary |
| Day16–20 | gradient、MLP／VQC／Hybrid、binary Iris | 在固定資料與預算下比較模型 |
| Day21–25 | bottleneck、re-uploading、kernel、trainability、Wine | 檢查表達方式、成本與資料任務的關係 |
| Day26–30 | noise、單GPU計時、多GPU模型、QPU預演、證據彙整 | 分辨模型假設、實測範圍與部署缺口 |

關鍵轉變是開始能追問：「這個結果是怎麼得到的？」一個漂亮的accuracy，現在可以一路追回split、preprocessor、參數選擇、backend與輸出契約。

完整30篇文章與每日可執行檔清單在[彙整報告](../../results/day30/README.md)。檔案齊全代表交付物存在，不表示今天重跑過全部歷史實驗。

## 2. QML模型有學到東西嗎？有，但還要看對照

Day20的binary Iris使用99筆去重資料，每split的test只有21筆；Day25的binary Wine使用119筆，每split的test為26筆。兩天各有兩組split，不能把它們當大量獨立重複。詳見[Day20報告](../../results/day20/README.md)與[Day25報告](../../results/day25/README.md)。

下面只比較同一天、同split、同PCA2輸入的MLP／VQC／Hybrid。數字是選定模型的test Brier，越小越好：

| 任務／Split | MLP | VQC | Hybrid |
|---|---:|---:|---:|
| Iris 2028 | 0.070884 | 0.105396 | 0.076867 |
| Iris 2029 | 0.088543 | 0.102869 | 0.081691 |
| Wine 2030 | 0.028478 | 0.032404 | 0.034820 |
| Wine 2031 | 0.041693 | 0.057050 | 0.041495 |

VQC在這四個配對的Brier都比MLP高。Hybrid有兩個配對較低、兩個較高；Wine2031差值只有約−0.000198。這些是有限資料上的描述，不足以做穩定優勢或顯著性主張。[重新核算表](../../results/day30/README.md)

![相同輸入的Brier差值](../../results/day30/paired_brier.png)

不同模型有不同參數量；相同objective evaluation預算也不等於相同更新步數、時間或充分收斂。因此這份對照支持「在這套固定協定下得到這些結果」，不能推出某個模型族已達最佳表現。

Day25還保留看到全部13維特徵的logistic13：Wine2031的test Brier為0.032224。它是實用完整輸入baseline，與PCA2模型的輸入資訊不同，不能用來單獨歸因量子層效果。但它提醒我們：先問資料是否被降維丟掉，再問量子電路是否不夠深。

## 3. Quantum Kernel沒有免除公平比較的責任

Day23改用固定量子feature map與classical kernel ridge。它沒有訓練量子weights，仍然需要檢查Gram matrix、λ選擇及完整前處理。

兩個split的quantum kernel test Brier分別為0.066907、0.071703；linear為0.041345、0.047579；RBF為0.067819、0.028753。Quantum kernel在第一組略低於RBF，第二組較高；linear在兩組都較低。[Day23報告](../../results/day23/README.md)

這三者使用相同四維縮放輸入；Day20則是PCA2。即使來自Iris，也不能把跨日結果合併成隔離單一因素的比較。Ridge的clipped score也不是經校準的機率。

我們完成的是兩種QML建模路徑：可訓練電路與固定量子kernel。兩條路都需要classical baseline、資料切分與完整成本紀錄。

## 4. GPU讓模擬變快，回答的是哪個問題？

Day27才依序量測相同電路、相同weights與fp64條件下的CPU／GPU warm latency。小電路4qubits／4blocks的CPU/GPU約0.613，當時CPU較快；16qubits／12blocks約1112.885，當時GPU較快。[Day27原始報告](../../results/day27/README.md)

這個大比例有明確分母：`qpp-cpu`、單一OpenMP thread、特定RY／RZ／CZ電路、global-Z exact observe。它不是對最佳classical ML系統的比較，也不是GPU普遍快千倍。Day30重新核算了全部8組fp64配對，而非只留下最大的比例。

還有一個容易混淆的事實：**Day20／25共28次訓練都是NumPy CPU reference；這兩天的CUDA-Q training calls皆為0。** CUDA-Q CPU／GPU核對的是凍結模型的forward。不能拿NumPy fit time除以GPU inference time，稱為QML訓練加速。

早期Day10確實執行了CUDA-Q optimization loop；兩件事並不衝突。我們需要逐個實驗說清楚training engine，而不是替整個repository貼一張「GPU訓練」標籤。

## 5. Noise讓我重新看待Accuracy

Day26對凍結Wine VQC的末端q0加入bit flip。Split2030在p=0時Brier約0.032404，p=.3時約0.124883；accuracy卻同為96.15%。[Day26報告](../../results/day26/README.md)

原因是這個特定noise位置把ZZ乘上正縮放，分數往0.5靠近但未必越過分類threshold。分類標籤可以不變，預測品質仍然變差。只看accuracy，會漏掉這種變化。

Day29用另一個可解析電路確認，更多shots會降低抽樣變異，卻不會將合成noise下的ZZ期望值0.84恢復成理想值1。三模式的90組counts全部在本地執行，這不是硬體校準資料。[Day29報告](../../results/day29/README.md)

Day24也讓「梯度小」這句話更具體：需要一起看初始化、qubit數、depth與cost locality。那一天的768個初始化、1,536個單參數梯度是小規模診斷，沒有證明所有QNN的漸近trainability，更不能直接推論分類準確率。[Day24報告](../../results/day24/README.md)

## 6. 我們還沒有跨過的硬體邊界

| 問題 | 本系列證據 | 尚缺的證據 |
|---|---|---|
| 單GPU是否能加速此模擬？ | Day27特定設定實測 | 其他CPU調校、電路與規模 |
| 多GPU能提高多少容量／速度？ | Day28可重現假設模型 | 真實多卡、互連、分散與peak memory量測 |
| 硬體target能否編譯此小電路？ | Day29 IonQ本地emulation | 遠端裝置接收與實際執行 |
| 真實QPU是否有實用QML優勢？ | 本系列未測量 | 同任務品質、端到端成本與可重複硬體結果 |

Day28的16組容量、12組延遲情境有公式與baseline來源，但沒有multi-GPU measurements。Day29的2048-shot工作規格仍為`not_submitted`，physical QPU jobs為0。queue與費用是未知值，不是零。

因此，本系列既沒有證明量子優勢，也沒有證明QML不可能有用。它將下一步需要驗證的問題縮小到能實際設計實驗的程度。

## 7. 最後一天的實作：讓結論也能重跑

[evidence.py](evidence.py)不用啟動CUDA-Q，直接讀保存的資料並執行以下核對：

1. 對Day20／23／25共20個選定模型，依labels與保存預測重算60組partition Brier、accuracy與confusion matrix；檢查seed／λ由validation選定、參數與candidate一致。
2. 核對每split內的row IDs沒有交疊；重算Day26共18組Wine noise metrics。
3. 由Day27原始7次warm時間重算8組fp64 median與比值，確認protocol、weights hash與輸出誤差。
4. 檢查Day28引用的單GPUbaseline沒有過期；彙整Day29的local execution、counts總shots與未提交狀態。
5. 保存26份來源JSON的SHA-256，建立30篇文章與Day03–30可執行檔索引。

輸出[evidence.json](../../results/day30/evidence.json)、[報告](../../results/day30/README.md)與Brier差值圖。`--check`會比較當前來源、彙整JSON及報告，來源或結果變動時要求重新產生，避免舊結論悄悄沿用新實驗。

```bash
# 產生彙整與圖表；沿用既有環境，不需GPU或帳號。
MPLCONFIGDIR=/tmp/day30-matplotlib .venv/bin/python articles/day30/evidence.py

# 純標準函式庫核對，沒有GPU或Matplotlib也能執行。
python3 articles/day30/evidence.py --check

# 彙整邏輯、損壞資料與保存結果測試。
python3 -m unittest discover -s articles/day30 -p 'test_*.py'
```

依賴入口：[requirements-day30.txt](../../requirements-day30.txt)。重跑只覆寫`results/day30/`的本日產物。這是保存資料的一致性audit，不是重新計算所有模型forward、證明全部前處理無誤或重跑30天GPU測試；舊實驗的timestamp與限制仍保留在原日報告。

## 8. 下一輪研究：先訂什麼結果會改變判斷

以下是本專案的後續實驗規格，尚未執行。目標是讓新證據足以改變選型，而不是把既有結果換一張更大的圖。

| 問題 | 下一個實驗 | 事先約定的判讀方式 |
|---|---|---|
| 模型差異是否穩定？ | 用新的評估設計與保留資料，比較相同輸入的classical／VQC／Hybrid | 先定最小實用Brier改善幅度、重複規格與成本上限，再看test |
| 瓶頸是不是資訊壓縮？ | 配對完整輸入baseline、PCA、learned bottleneck與re-uploading | 分開報告表達方式、輸入資訊與optimizer預算，不只比參數量 |
| 有限shots會否改變選模？ | 凍結候選模型，對固定shot budgets重複評估validation | 檢查排名與預測不確定性；不以test反覆選shot數 |
| GPU收益能否延伸？ | 固定精度與電路，增加合理CPU thread對照和記憶體量測 | 保留全部case與原始times；容量、latency、throughput分別報告 |
| QPU硬體落差有多大？ | 選定可用裝置後，先執行Day29的小型已核對工作 | 保存job、校準、shots、編譯、queue、費用，再決定是否搬模型 |

目前Iris／Wine結果已在開發過程中被反覆看過；即使每一天遵守validation選參數，也不應把這些test當成永遠未見的資料。下一輪若要做更強主張，需要新的保留評估設計。

如果QML沒有達到事先定義的實用品質或成本條件，就保留較簡單的classical方案。若在充分對照後出現可重複差異，再投入更大的模擬或硬體預算。這是一個能被結果改變的決策流程。

## 9. 回答第一天的問題

從AI Engineer走到QML，不只是學一套gate語法。真正花時間的是把資料、數值、量測、optimizer與執行成本放在同一份可檢查的紀錄裡。

我仍然相信這段學習值得做：它讓我能寫出量子模型，也能指出自己的證據在哪裡停止。至於QML是否值得放進某個實際系統，答案要由那個任務的完整對照實驗決定。

30天的終點，是一個可以繼續被驗證、修正，甚至被否定的專案。

## 來源與閱讀入口

本文結論以repository保存的實驗為依據，沒有新增外部市場、硬體可用性或產業效能主張。數值入口為[Day20](../../results/day20/README.md)、[Day23](../../results/day23/README.md)、[Day24](../../results/day24/README.md)、[Day25](../../results/day25/README.md)、[Day26](../../results/day26/README.md)、[Day27](../../results/day27/README.md)、[Day28](../../results/day28/README.md)、[Day29](../../results/day29/README.md)。

方法與官方backend文件沿用[REFERENCES.md](../../REFERENCES.md)的既有核對紀錄；查閱日期與版本限制不因本日彙整而更新。[完整30天目錄](../../README.md)、[Roadmap](../../ROADMAP.md)。
