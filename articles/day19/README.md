# Day 19｜混合神經網路：讓一般計算層與量子電路一起學習

[Day18](../day18/README.md) 在固定資料、損失函數與評估預算下，比較 Classical ML 與 QML，釐清公平比較的條件與結論限制。Day19 接著將經典層與量子層串接成單一模型，檢查誤差如何傳回各部分，讓三組參數共同更新。

**量子電路放在模型中間時，不只自身權重要更新，前後的一般計算層也需要取得正確的梯度。** 混合神經網路（hybrid neural network）將這些運算接成同一條訓練流程。本章先用 classical encoder（經典編碼層）把兩個輸入特徵轉成電路角度，再由 quantum layer（量子層）計算 `ZZ` 期望值，最後透過 classical head（經典輸出層）產生介於 0 與 1 的分數。與固定編碼不同，這裡產生角度的權重也會接受訓練，整個模型共有十二個參數。核心重點是量子層除了提供自身權重的導數，也必須提供輸入角度的導數，前面的編碼層才能透過 chain rule（鏈式法則）取得正確梯度。本章沿用參數位移法計算量子部分的局部導數，再接上經典計算的導數，並用完整模型的有限差分逐一核對。實作以三筆資料執行短程聯合更新，保存中間值、梯度與模型參數。資料向前產生預測，梯度則依各步驟的導數傳回參數。初始化可能阻斷這條路徑，而訓練誤差下降、分類正確與新資料表現，仍是不同的驗證問題。

[Day18](../day18/README.md) compared classical ML and QML under fixed data, loss, and evaluation budgets, clarifying the conditions for fair comparison and the limits of the conclusions. Day19 connects classical and quantum layers into one model and checks how errors propagate to update all three parameter groups jointly.

**When a quantum circuit sits inside a model, both its own weights and the surrounding classical layers need correct gradients.** A hybrid neural network connects these computations into one training workflow. A classical encoder converts two input features into circuit angles, a quantum layer computes the `ZZ` expectation, and a classical output head produces a score between 0 and 1. Unlike fixed encoding, the parameters that generate the angles are also trained, giving twelve parameters in total. The central requirement is that the quantum layer provide derivatives with respect to both its own weights and its input angles, allowing the preceding encoder to receive correct gradients through the chain rule. The chapter uses parameter-shift for the quantum layer's local derivatives, combines these with classical derivatives, and checks every parameter against finite differences of the full model loss. A short joint-update experiment on three samples saves intermediate values, gradients, and model parameters. Data flows forward to predictions, while derivatives connect the loss back to each parameter. Initialization can block that path, and decreasing loss remains distinct from correct classification or performance on new data.

---

Day 18 比較不同模型。今天建立一個真正串接的模型：**一般編碼層產生電路角度，量子層回傳期望值，一般輸出層產生機率，三組參數一起更新。**

完整程式：[hybrid.py](hybrid.py)、[experiment.py](experiment.py)、[demo.py](demo.py)。本日手寫鏈式法則，沿用 NumPy／CUDA-Q，不新增 PyTorch；重點是驗證梯度穿過整個模型，不是再做分類效能比較。

## 1. 十二個參數的資料流

編碼層（encoder）把輸入整理成量子電路的角度；量子層執行電路並提供數值；輸出層（head）將這個數值轉成最後分數。特徵是描述樣本的數值，例如長度與寬度。

仿射轉換（affine transformation）是「乘上權重，再加偏差」。權重控制輸入影響的大小，偏差是可調整的加法常數。`tanh` 將值壓到 −1 與 1 之間，`sigmoid(u)=1/(1+exp(−u))` 將值轉到 0 與 1 之間；`exp` 是指數函數。

```mermaid
flowchart LR
    X[兩個輸入特徵] --> E[一般編碼層：乘上 W，再加 b]
    E --> T[tanh，再乘 π]
    T --> Q[兩量子位元：RY 編碼與可調電路]
    Q --> F[ZZ 期望值 f]
    F --> H[輸出層：a f＋c]
    H --> P[sigmoid 分數]
    P --> L[Brier 損失]
```

| 部分 | 公式 | 參數數量 | 攤平向量索引 |
|---|---|---:|---|
| 一般編碼層 | z=xW+b，h=tanh(z)，θ=πh | 4＋2=6 | W=`[0:4]`、b=`[4:6]` |
| 可調量子電路 | f=⟨ZZ⟩ | 4 | `[6:10]` |
| 一般輸出層 | u=af+c，p=sigmoid(u) | 2 | a=`[10]`、c=`[11]` |

下式 `x` 是輸入，`W`、`b` 是編碼層參數，`h` 是經 tanh 轉換後的值，`θ` 是電路角度；`a`、`c` 是輸出層倍率與偏差。`f` 是量子期望值，`u` 是 sigmoid 前的原始分數，`p` 是最終輸出。

切片索引採 Python 的左含右不含規則。W 重新排列為 2×2；一批 B 筆資料的形狀依序為 B×2、B×2、B、B。

量子核心程式（quantum kernel）是描述量子操作的函式。RY 以角度旋轉單一量子位元的狀態；可調電路模板（ansatz）則組合多個操作，提供訓練權重。ZZ 是 `Z0Z1`，將兩位元相同記為 +1、不同記為 −1，期望值是依機率計算的平均值。

量子核心程式重用 Day 9 的一層模型，但不走其 `features→πx` 主控端外層函式，因為一般編碼層已直接產生弧度角度。NumPy 是 Python 的數值運算套件，CPU 是一般電腦的中央處理器，GPU 是擅長平行運算的圖形處理器。前後的一般計算都在 NumPy CPU；`--backend nvidia` 只切換量子模擬器。

## 2. 與固定資料編碼電路有何不同？

Day 14／15 的角度映射由固定縮放規則決定。本日 W、b 也接受最佳化器更新，因此角度會隨訓練改變：

```python
hidden = np.tanh(x @ parameters[:4].reshape(2, 2) + parameters[4:6])
angles = np.pi * hidden
```

程式的 `@` 是矩陣乘法，`reshape(2,2)` 將四個數排成兩列兩欄，`parameters[4:6]` 取索引 4 與 5。弧度是角度單位，`π` 弧度等於半圈。

tanh 將輸出限制在 [−1,1]，再映射至 [−π,π]。沒有以硬截斷截斷這一層，但 tanh 接近上下限時會變得平坦，這稱為飽和，此時導數可能很小。導數描述輸入微小改變時輸出的變化率。這不是最佳角度範圍的結論，也不保證所有特徵都保持可區分。

API 是程式呼叫功能的介面，B 是樣本數，B×2 表示每筆有兩個特徵。縮放器保存調整資料尺度的規則，應只由訓練資料決定，避免評估資料提前影響模型。

示範特徵已是小尺度的二維數值，API 接受有限 B×2 矩陣，沒有在模型內建立縮放規則。真實資料仍需獨立的僅由訓練資料決定的前處理，且應與模型檢查點一起保存；本日沒有隱藏的縮放器。

## 3. 量子層必須回傳兩種導數

梯度將損失對每個參數的導數排成向量，供最佳化器決定更新方向。雅可比矩陣（Jacobian）則整理輸出對各輸入或參數的導數。

若只求量子權重的梯度，一般編碼層就無法得到正確的更新方向。本日的量子層雅可比矩陣包含：

```text
[df/dθ0, df/dθ1, df/dw0, df/dw1, df/dw2, df/dw3]
```

局部偏導數表示暫時固定其他變數，只看某個角度的影響；鏈式法則再將相連步驟的導數相乘。

六個角度各自控制一個獨立 RY，因此可以使用 Day 17 的標準兩點參數位移法：

```text
df/dα = [f(α+π/2) − f(α−π/2)]/2
```

位移 θ 時固定量子權重；位移權重時固定 θ。這裡求的是量子層的局部偏導數，接著才乘一般鏈式法則的係數。不能直接對 W 加 π/2 然後套用同一公式，因為 W 經仿射轉換、tanh、π 才進入多筆資料的電路。

## 4. 完整鏈式法則

Brier 損失是預測分數與 0／1 標籤的平均平方誤差，標籤是正確答案。`mean` 表示對樣本取平均，`i` 是樣本編號，`B` 是樣本數，`δ_i` 是第 i 筆損失對原始分數的導數。

Brier 損失為 `L=mean((p−y)²)`。定義每筆資料對輸出層原始分數的導數：

```text
δ_i = 2(p_i−y_i) p_i(1−p_i) / B
```

輸出層梯度：

```text
dL/da = Σ_i δ_i f_i
dL/dc = Σ_i δ_i
```

量子權重梯度：

```text
dL/dw_j = Σ_i δ_i a (df_i/dw_j)
```

編碼層活化前梯度：

```text
g_i = δ_i a (df_i/dθ) ⊙ π(1−h_i²)
dL/dW = Σ_i outer(x_i,g_i)
dL/db = Σ_i g_i
```

`Σ` 表示對樣本加總，`⊙` 表示對應元素相乘；外積 `outer(x_i,g_i)` 將兩個向量的分量逐一相乘，形成矩陣。`g_i` 是第 i 筆資料傳回編碼層、套用 tanh 之前的梯度。

程式的外積方向與 `x @ W` 一致。每個梯度分量都與 NumPy 完整模型損失的中央差分（h=1e-5）核對，包含六個編碼層、四個量子和兩個輸出層參數。

中央差分以參數增加與減少一小段距離後的損失差，除以兩點距離，近似導數；本例距離參數 `h=1e-5`，即 0.00001。

這是手寫的鏈式法則，尚未接到 PyTorch 的自動微分介面；自動微分工具會追蹤計算步驟，再依鏈式法則計算導數。若未來包成 自訂自動微分函式，仍須維持相同的輸入／權重的雅可比矩陣與批次加總與平均規則。

## 5. 輸出層初始化也會影響梯度

由公式可見，a=0 時編碼層與量子權重的梯度全部為 0，即使輸出層偏差的梯度非零。測試刻意把 a 設為 0，驗證這個阻斷效果。

初始化是指定訓練開始前的參數，隨機種子方便在相同環境重現序列。`normal(0,0.2)` 從平均值 0、標準差 0.2 的常態分布抽樣；常態分布呈鐘形，標準差描述分散尺度。

正式示範以隨機種子 42 的 `normal(0,0.2)` 初始化 12 個參數，再將 a 指定為 0.8。這是為了展示三組參數都能更新，不是經搜尋得到的最佳初始化。sigmoid 或 tanh 飽和也可能減弱梯度，貧瘠高原描述特定條件下梯度隨規模快速縮小的現象，不能將所有小梯度都歸為這個原因。

本日 p 是 `sigmoid(af+c)`，**不再是 Day 15 的直接同位性機率 `(1−ZZ)/2`**。它是一般輸出層的輸出分數，沒有機率校準宣稱；校準是檢查預測 80% 的樣本是否約有 80% 屬於該類別。

## 6. 三筆資料的聯合更新

```text
features = [[−0.6,0.4], [0.3,0.5], [0.7,−0.2]]
labels   = [1,0,1]
```

學習率是梯度更新的倍率，控制一次移動多遠。固定學習率為 0.4，執行八次 `parameters -= 0.4 * gradient`。記錄初始化與八次更新後的損失／梯度，共九筆紀錄。

CPU／GPU 的損失都從約 0.236563 降至 0.200478，三組參數均有改變。但最終三筆 p 都大於 0.5，並未把類別 0 樣本分類正確。**損失下降只驗證更新流程，不代表分類任務已解決。** 本日沒有訓練／測試資料切分、選模或準確率效能比較，也沒有將結果與 Day 18 排名混用。

![Hybrid training](../../results/day19/qpp-cpu/hybrid_training.png)

L2 長度是向量各分量平方相加後開平方根。右圖用它描述各參數群組梯度的整體大小，不同群組參數數量與單位不同，不能直接解讀為「哪一層更重要」。

## 7. 成本與可觀察中間值

前向計算是固定參數下由輸入算到輸出，不更新參數。每筆資料需一次原始前向計算，加上六個局部導數各兩次位移後的前向計算，共 13 次量子 observe。B=3、九次梯度評估：

```text
9 × 3 × 13 = 351 次訓練 observe
```

最終參考值核對與模型重新載入的評估另計，沒有算進 351。無噪聲表示未加入使運算偏離理想情況的干擾。這些期望值由模擬器保存的狀態直接計算，仍可能有浮點誤差，沒有有限次量測訓練。

`forward` 同時回傳機率和暫存紀錄：隱藏層輸出、角度、量子期望值、原始分數。它們是主控端保留的一般數值，不是從真實量子處理器（QPU）直接取得所有內部振幅；振幅是量子狀態的係數，其絕對值平方才是量測機率。

## 8. 執行、模型檢查點與測試

`.venv` 是專案獨立保存套件的虛擬環境；`OMP_NUM_THREADS=1` 將 CPU 平行工作的執行緒數設為 1。模型檢查點（checkpoint）保存可重新載入的設定與參數；JSON 是結構化文字資料格式，PNG 是圖片格式。

沿用 `.venv`，無新增依賴；[requirements-day19.txt](../../requirements-day19.txt)。在專案根目錄執行：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day19/experiment.py
OMP_NUM_THREADS=1 python articles/day19/experiment.py --backend nvidia
OMP_NUM_THREADS=1 python articles/day19/plot_results.py
OMP_NUM_THREADS=1 python articles/day19/plot_results.py --backend nvidia

# 讀取保存參數，印出各層中間值。
OMP_NUM_THREADS=1 python articles/day19/demo.py --features -0.6 0.4
```

輸出為 `results/day19/<backend>/history.json`、`summary.json`、`checkpoint.json`、`hybrid_training.png`。模型檢查點包含架構描述、參數分段、輸入契約、隨機種子與所有 12 個參數。實驗會重新載入 JSON，驗證輸出一致。示範僅供此固定架構，不是跨模型的通用載入器。

實驗可加 `--output-dir /tmp/day19-check` 另存；預設重跑更新目錄，繪圖／示範讀取預設執行後端路徑。

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day19 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY19_TARGET=nvidia python -m unittest discover -s articles/day19 -p 'test_*.py' -v
```

五個測試涵蓋前向計算參考值／輸入不變性、全部 12 個參數的鏈式法則、量子輸入與權重 Jacobian、輸出層倍率為零梯度阻斷、錯誤輸入。完整數字見 [結果紀錄](../../results/day19/README.md)。

## 9. 來源與下一篇

[D15] [NVIDIA Hybrid QNN 教學（0.8.0）](https://nvidia.github.io/cuda-quantum/0.8.0/examples/python/tutorials/hybrid_qnns.html) 示範一般計算層與量子電路層的整合。2026-09-07 重新查閱；本日未移植其歷史版本梯度程式，而是沿用 Day 17 的位移規則並獨立核對完整鏈式法則。來源索引：[REFERENCES.md](../../REFERENCES.md)。

下一篇 [Day 20](../day20/README.md) 將進入 Iris 鳶尾花資料集（以花的尺寸辨認種類）的一般／量子／混合模型比較。本日不作 QPU、速度或量子優勢宣稱。

## 延伸研究

[N5] Brian Coyle et al. “Training-efficient density quantum machine learning.” npj Quantum Information 11, 172 (2025)；研究論文。[原始來源](https://doi.org/10.1038/s41534-025-01099-6)；[完整書目](../../REFERENCES.md#n5)。

本章檢查混合模型各層的導數；這篇研究可延伸思考模型結構與梯度計算量的關係。本章仍以獨立數值方法核對既有模型。
