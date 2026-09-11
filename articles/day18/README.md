# Day 18｜一般機器學習與量子模型：固定條件下的比較

Day 17 把梯度算清楚了。今天換問題：**若要把「一般機器學習」和「量子模型」放在同一張成績單上，事前要先釘死哪些規則？**

想像一場考試：同一份考卷、同一套計分、同一時間預算，但考生人數與答題策略不同。成績可以比較「這次考試誰答對較多」；不能直接推論「這個族群永遠比較強」。本章的「公平」指**條件公開且一致**，不是宣稱一次實驗就能替整個模型家族下結論。

比較對象：logistic、MLP、VQC，外加「永遠輸出訓練標籤平均值」的常數對照。三者共用訓練集縮放、Brier 損失、不需梯度的座標搜尋；**沒用** Day 17 的參數位移。程式：[benchmark.py](benchmark.py)、[plot_results.py](plot_results.py)。

Day18 benchmarks logistic, MLP, and a variational quantum classifier on one small XOR split under a shared protocol: train-only scaling, Brier loss, coordinate search with 109 objective evaluations per fit, and validation-only seed selection. Results are recorded with costs and do not establish model-family rankings or quantum advantage.

---

## 1. 先把比較規則寫進檔案

**QML**＝量子機器學習；**VQC**＝變分量子分類器（調電路參數來分類）；**MLP**＝多層感知器。任務仍是 XOR：兩個座標異號為一類、同號為另一類。

訓練開始前寫入 `protocol.json`、`dataset.json`、`scaler.json`：

| 項目 | 固定設定 |
|---|---|
| 資料 | Day 15 XOR 規則，新種子＝2027 |
| 切分 | 訓練／驗證／測試各 12 筆，每類各 6 |
| 前處理 | 同一把「只看訓練集」的最小／最大值尺；越界截斷 |
| 損失 | Brier：`mean((p1−y)²)` |
| 最佳化器 | 同一份座標搜尋（不需梯度） |
| 每次訓練預算 | **109** 次目標函數評估 |
| 初始化 | `normal(0,0.2)`，每模型種子 42／43 |
| 選模 | 各模型內選最終驗證 Brier 較低的種子 |
| 分類門檻 | `p1≥0.5` → 類別 1 |
| 測試 | 選定後才評分 |

訓練改權重、驗證選設定、測試最後拆封。資料內容的 SHA-256 寫進協定，CPU／GPU 兩次執行必須一致。用新切分是為了不把已看過的 Day 15 測試集再當盲測；它仍是小型合成 XOR，不是獨立真實資料。

先切分再縮放，避免資料洩漏。[D14] 看完本次測試後，不再改預算或權重。

## 2. 三個模型＋一個常數基準

| 模型 | 怎麼算 | 可訓練參數 |
|---|---|---:|
| logistic | `sigmoid(x·w + b)` | 3 |
| MLP | `2 → tanh(2) → sigmoid(1)`（含偏差） | 9 |
| VQC | 兩位元、`positive_half` 編碼、一層模板、`p1=(1−ZZ)/2` | 4 |
| constant | 訓練標籤平均值＝0.5 | 不訓練 |

**重要：**本日 logistic 用的是 **Brier**，不是標準邏輯斯迴歸常見的對數損失（最大概似）。兩者目標不同。[D16] 這裡保留 logistic 的輸出形狀，刻意統一損失，方便控制實驗——**不能**把它說成 scikit-learn `LogisticRegression` 的官方效能基準。

XOR 需要非線性：logistic 在 0.5 門檻下邊界是直線，很難拆開對角兩類；MLP／VQC 可以彎。因此 logistic 差，**不代表**「一般機器學習整體輸給量子」。MLP 的容量與調參也沒有廣泛搜尋。

參考測試成績（種子 2027 切分）：常數與 logistic 約 50%；MLP 約 58%；VQC 約 100%、Brier 約 0.034。12 筆測試每錯一筆約差 8.3 個百分點——數字敏感，不宜做家族排名。

## 3. 相同「輪數」≠ 相同工作量

座標搜尋一次只試改一個權重：加減一步，變差就退回。若三個模型跑「相同輪數」，九參數 MLP 會比三參數 logistic 多很多候選評估。

本日改釘：**固定 109 次目標函數**＝1 次初始損失＋108 次＝54 對正負候選。只接受完整配對，不超預算；每對之後存已接受的權重與損失。一整輪都沒改善才把步長 0.4 減半；預算用完就停。

| 模型 | 完整輪次 | 最後未完成整輪的更新 |
|---|---:|---|
| logistic（3 參數） | 18 | 無 |
| MLP（9 參數） | 6 | 無 |
| VQC（4 參數） | 13 | 再處理兩個座標 |

未完成整輪的更新不觸發步長減半。大家候選評估次數相同，但參數多的模型「每個旋鈕被試到」的機會較少；座標順序也可能影響結果。這是本日資源規則的取捨，不是唯一公平定義。

## 4. 相同目標函數次數 ≠ 相同成本

每次目標函數都看同一批 12 筆訓練資料：

- logistic／MLP：NumPy 批次矩陣
- VQC：逐筆精算 `observe` → 每次訓練 `109×12＝1,308` 次電路呼叫；兩種子共 2,616 次／後端（驗證／測試／參考另計）

選共同、不需梯度的最佳化器，是為了避免拿不同梯度估計成本假裝「一樣努力」；Day 17 位移法**沒有**在此呼叫。這也不表示座標搜尋對每個模型都最合適。

另記 `fit_seconds_including_first_compilation`（含訓練內首次編譯與主控端成本，不含後續驗證／測試）。CPU／GPU 可能並行、後續種子可能吃到快取，未做隔離暖機或重複計時。因此時間只供工程追蹤，**不能**推論 GPU 加速幅度或演算法複雜度。GPU 執行裡，一般模型仍在 NumPy CPU 上算；VQC 單次 fit 約數十秒，遠高於 logistic／MLP 的毫秒級。

## 5. 選模與報告：全部候選都留著

各模型內只依**最終驗證 Brier**選種子；平手取較小種子。不挑「某一輪驗證最好」，也不用測試集挑哪個模型家族勝出。

報告訓練／驗證／測試的 Brier、準確率與混淆矩陣（列＝真實、欄＝預測）。常數 `p1=0.5` 在本門檻下全預測 1 → 均衡測試集準確率 50%、Brier 0.25。

`candidates.json` 保存所有種子的驗證、初始／最終權重、每對候選後的訓練紀錄、預算與時間；`selected.json` 保存選定權重與各切分機率／指標。原始資料、縮放器、協定分開存放，方便稽核。

## 6. 怎麼讀圖與怎麼不要過度解讀

![Benchmark](../../results/day18/qpp-cpu/benchmark.png)

橫軸是**目標函數評估次數**，不是完整輪次，也不是秒。右圖測試 Brier 只描述這 12 筆。

兩個初始化種子 ≠ 兩個獨立資料切分；CPU／GPU 同資料重跑 ≠ 額外統計樣本。本章沒有做顯著性檢定或信賴區間，也**不宣稱**量子方法優於一般方法。若要推廣，需要更大資料、多切分、各模型合理調參預算與更完整的經典基準；Day 20／25 會繼續擴充任務。

## 7. 重跑、推論與測試

沿用 `.venv`；固定依賴見 [requirements-day18.txt](../../requirements-day18.txt)。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day18/benchmark.py
OMP_NUM_THREADS=1 python articles/day18/benchmark.py --backend nvidia
OMP_NUM_THREADS=1 python articles/day18/plot_results.py
OMP_NUM_THREADS=1 python articles/day18/plot_results.py --backend nvidia

OMP_NUM_THREADS=1 python -m unittest discover -s articles/day18 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY18_TARGET=nvidia python -m unittest discover -s articles/day18 -p 'test_*.py' -v
```

預設寫入 `results/day18/<backend>/`；比較可用 `--output-dir /tmp/day18-check`。單筆推論（不更新權重）：

```bash
OMP_NUM_THREADS=1 python articles/day18/demo.py --features -0.7 0.7
```

四個測試檢查：三種參數量都剛好用完 109 次、損失不增加、輸入不被改、logistic 解析輸出、切分／縮放邊界與 VQC NumPy 參考。實驗通過條件**不要求**某個模型擊敗另一個。

全部 VQC 結果為無雜訊精算模擬器；沒有有限 shots 或 QPU 比較。本日留下可稽核的比較協定（資料雜湊、預算定義、選模規則、完整候選留存），並同時報告準確率、機率誤差與電路呼叫成本。它是固定條件下的控制實驗，不是模型家族排名或量子優勢證明。

下一篇 [Day 19](../day19/README.md) 談一般層與量子層串成的混合網路。

## 8. 來源

- [D14] [scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：訓練集-only 前處理與資料洩漏。
- [D16] [scikit-learn log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)：標準邏輯斯對數損失；與本日 Brier 控制實驗區分。

查閱日期 2026-09-07。本日 NumPy 自訂實作，未安裝 scikit-learn；索引見 [REFERENCES.md](../../REFERENCES.md)。

## 延伸研究

[N7] Jan Schnabel and Marco Roth. “Quantum kernel methods under scrutiny: a benchmarking study.” Quantum Machine Intelligence 7, 58 (2025)；研究論文。[原始來源](https://doi.org/10.1007/s42484-025-00273-5)；[完整書目](../../REFERENCES.md#n7)。

本章建立公平比較規則；該研究可補充資料集、編碼與超參數對基準的影響。其對象是核方法，不能直接當成本章所有模型的排名。
