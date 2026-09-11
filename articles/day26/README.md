# Day 26｜量子噪聲：模擬干擾如何改變量測與預測

[Day25](../day25/README.md) 在 Wine 資料集上比較了無噪聲的傳統、量子與混合分類器。今天換成另一件事：**把干擾明確寫進電路**，看狀態、量測次數表，以及已訓練模型的預測會怎麼變。

想像收音機收到同一首歌，但天線偶爾翻錯頻道、相位漂一下，或信號被沖淡。多聽幾遍可以讓「這次聽到什麼」比較穩定，卻**消不掉**頻道本身已被改過的事實。量子噪聲也一樣：通道改的是理論機率；有限次量測（shots）只讓計數在那些機率附近晃。

程式：[noise.py](noise.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[測試](test_noise.py)、[plot_results.py](plot_results.py)。結果見 [實測報告](../../results/day26/README.md)。沿用 `.venv`（專案獨立套件目錄），無需新增套件。

Day26 adds explicit Pauli noise after state preparation on two-qubit examples and on a frozen Day25 Wine VQC. Exact density-matrix checks, finite-shot sampling on CPU and GPU, and Brier-versus-accuracy comparisons are recorded for specified channels and insertion points; the chapter does not claim a full hardware noise model or noise-aware training.

---

## 1. 從狀態向量到密度矩陣

**狀態向量**用一組振幅描述純態；振幅是複數，絕對值平方才是量測機率。純態不表示結果一定確定——疊加態也可以是純態。

噪聲若讓狀態隨機走不同路徑，整體可能變成**混合態**，要用**密度矩陣** ρ 表示。密度矩陣同時保存機率與相位關係；相位關係影響分量如何相長或相消。純態是特例：`ρ=|ψ⟩⟨ψ|`。

噪聲通道可用 **Kraus 算符**描述。每個 K_k 是一種作用分量；加總後得到變化後的狀態：

```text
ρ' = Σ_k K_k ρ K_k†
Σ_k K_k† K_k = I
```

`†`＝共軛轉置（轉置後對複數取共軛）；I＝單位矩陣。第二行是完備條件，讓總機率仍為 1。這是描述隨機作用的整體狀態，不是在振幅上任意加亂數。

NumPy 在此實作上述矩陣計算。測試核對：跡為 1、半正定（PSD）、完備條件。**跡**是對角線元素總和，此處代表總機率；半正定確保合法量測的機率非負。符合完全正性且保跡的通道稱為 CPTP。

n 個量子位元：狀態向量約 `2^n` 個複數，密度矩陣約 `4^n` 個——更耗記憶體。本章只用兩個量子位元。

## 2. 三種通道與噪聲機率 p

X、Y、Z 是 **Pauli** 操作：X 交換 0 與 1；Z 保留 0、把 1 的正負號翻掉（改相對相位）；Y 同時涉及位元與相位。以下 p 都在 0 與 1 之間：

| 通道 | 變化後的狀態 ρ′ | 直觀作用 |
|---|---|---|
| 位元翻轉（bit flip） | (1−p)ρ+pXρX | 以機率 p 套用 X |
| 相位翻轉（phase flip） | (1−p)ρ+pZρZ | 以機率 p 套用 Z |
| 去極化（depolarizing） | (1−p)ρ+(p/3)(XρX+YρY+ZρZ) | 合計機率 p，等機率套用 X、Y 或 Z |

以上對應本機 CUDA-Q 的 `BitFlipChannel`、`PhaseFlipChannel`、`DepolarizationChannel`，經實測核對。[D22] CUDA-Q 是執行量子程式的工具；同名通道在別的工具裡，p 的定義可能不同。

**Bloch 向量**用三個實數描述單一量子位元。此處去極化把向量乘上 `1−4p/3`：p=0.75 時變成零，得到完全混合態 `I/2`；p=1 時乘上 −1/3，**不是**完全混合。完全混合態對任何單位元量測方向都各半。位元／相位翻轉的 p=1 只表示「每次都做那個操作」，也不等於完全混合。

另一常見寫法 `(1−p)ρ+pI/2` 裡的 p 意義不同，不能直接代換。本章以解析式與實測核對端點。

## 3. 噪聲加在哪裡？

先準備 `|00⟩`、`|+0⟩` 或 Bell 態 Φ+，再**只在 q0** 加一次通道。`|+⟩=(|0⟩+|1⟩)/√2`；`Φ+=(|00⟩+|11⟩)/√2` 具有糾纏（無法拆成兩個獨立位元狀態）。

```python
noise = cudaq.NoiseModel()
noise.add_channel('rz', [0], cudaq.PhaseFlipChannel(0.3))
```

`NoiseModel` 保存設定；`add_channel` 把通道綁到指定閘與位元。電路末端執行 `rz(0.0,q[0])` 當注入點：角度 0 的 RZ 不改理想狀態，但綁定的噪聲仍會跑。狀態準備與量測基底轉換不用 RZ，所以通道只作用一次。測試確認通道有生效，沒被編譯最佳化或不支援的模擬器默默丟掉。

本章用已驗證的「閘綁定」介面，沒有用電路內的 `apply_noise`。這是刻意的末端噪聲來源，不是依真實 QPU 校準資料建的硬體模型，也沒有在每個閘後自動加噪聲。

## 4. 計數不變，狀態仍可能改變

**量測基底**＝用哪組方向區分狀態。Z 基底是平常的 0／1；X 基底區分 `|+⟩` 與 `|−⟩=(|0⟩−|1⟩)/√2`。換基底，才能看見原先次數表沒顯示的相位差異。

ZZ／XX：兩位元沿 Z 或 X 量測，各記 +1／−1 後相乘，再取期望值。理想 Bell 態兩者皆為 1。

- q0 位元翻轉：ZZ → `1−2p`，XX 不變。
- 相位翻轉：XX → `1−2p`，ZZ 不變。
- 去極化：兩者都乘 `1−4p/3`。

只看 Z 基底的 00／11 計數，可能漏掉相位噪聲。實驗同時存 Z 與 X 基底計數；X 量測前先對兩位元加 H（Hadamard，用來換 X／Z 基底）。基底轉換閘本身**未**加噪聲。

三種狀態 × 三種通道 × 五個 p（0、0.1、0.3、0.75、1）× 四種量測量（Z0、X0、ZZ、XX）＝**180** 筆精算結果。NumPy 參考與 CUDA-Q `density-matrix-cpu` 應在浮點誤差內一致；實測基礎最大誤差約 `4.441e-16`。

## 5. 理論機率、有限次量測與 GPU 軌跡抽樣

CPU＝中央處理器；GPU＝擅長平行運算的圖形處理器。**後端**＝實際跑模擬的工具。本章兩種方式：[D23]

- `density-matrix-cpu`：密度矩陣直接算期望值，也可有限次抽樣。
- `nvidia`：含噪聲的**軌跡**抽樣，不是一次算完整密度矩陣。軌跡＝一次隨機噪聲作用下的演化；多次軌跡共同反映噪聲分布。

`observe` 回傳期望值；`sample` 回傳計數。一次 shot＝一次準備加量測。每種狀態／通道／p，在 Z、X 基底各 **4,096** shots，種子 42／43／44 → **270** 筆紀錄、**1,105,920** shots／後端。相同種子不保證不同後端得到相同計數。

兩位元有 00、01、10、11 四種結果。實驗把頻率與 NumPy 理論機率比對，要求最大絕對差小於 `0.06`（工程煙霧測試門檻，不是信賴區間）。實測最大 bin 誤差：CPU 約 `0.015`、GPU 約 `0.017`，兩端皆通過。

**增加 shots 讓頻率更接近「含噪聲」的理論分布，不會讓分布回到無噪聲。**

## 6. 回到 Day25：固定 Wine 量子分類器

**VQC**（變分量子分類器）靠訓練調電路參數。本章載入 Day25 選定 VQC 的權重、前處理與測試資料，在完整電路後、ZZ 量測前，於 q0 加同一種通道。兩組切分各 26 筆測試 × 三種通道 × 三個 p（0、0.1、0.3）＝**18** 筆批次；加上前面 180 次，合計 **648** 次精算 `observe`。Wine 解析對照最大誤差約 `3.331e-16`。

若無噪聲時 `z=⟨ZZ⟩`，末端噪聲有：

```text
bit flip:     z'=(1−2p)z
phase flip:   z'=z
depolarizing: z'=(1−4p/3)z
p(label1)=(1−z')/2
```

權重、前處理與測試樣本全部固定，不重訓、不依噪聲挑種子。另沿用 Day25 常數對照（每筆輸出訓練集標籤 1 比例）；該傳統對照不經量子通道。

選定的 p 讓縮放因子仍為正，因此理論上不會把預測推過 0.5 門檻的另一側，但可能更靠近 0.5。**準確率**只看分類對錯，可能不變；**Brier**（預測與 0／1 標籤的平均平方誤差）仍可能變差。實測例如切分 2030、位元翻轉：Brier 由約 `0.032` 升到 p=0.3 時約 `0.125`，準確率仍為 **96.15%**；同位置的相位翻轉則 Brier 與準確率都不變。

末端相位翻轉與 ZZ **對易**（交換順序不改相關結果），所以這個插入點上看不到相位影響。若改插在 Ansatz 中間，後續閘可能讓相位差異進結果。這不能證明模型普遍免疫相位噪聲，也不是「訓練時就加噪聲」的實驗。

![噪聲對量測與分類輸出的影響](../../results/day26/noise_effects.png)

## 7. 重跑與示範

套件清單：[requirements-day26.txt](../../requirements-day26.txt)。完整實驗使用 Day25 已保存模型。`OMP_NUM_THREADS=1` 把 CPU 執行緒數固定為 1。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day26/experiment.py run
OMP_NUM_THREADS=1 python articles/day26/experiment.py sample
OMP_NUM_THREADS=1 python articles/day26/experiment.py sample --backend nvidia
OMP_NUM_THREADS=1 python articles/day26/plot_results.py

OMP_NUM_THREADS=1 python articles/day26/demo.py --channel phase_flip --probability 0.3
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day26 -p 'test_*.py' -v
```

示範預設用 CPU 密度矩陣，顯示 Bell 態的 ZZ／XX 與兩種基底計數，可調 shots 與種子。CPU 可獨立完成期望值、抽樣、示範與測試；完整圖表報告需要 GPU 抽樣摘要。

輸出在 `results/day26/`；重跑會覆寫同名檔。改設定或 Day25 模型後應全部重跑，避免舊計數與新設定混用。

五項測試涵蓋：通道與密度矩陣合法性、解析公式、相位翻轉的量測差異、去極化端點、結果確定時的抽樣，以及錯誤輸入。本章只模擬指定的末端 Pauli 通道，未加入振幅阻尼、讀出錯誤、相關噪聲，也未使用 QPU 校準資料。

Day27 比較 CPU／GPU 模擬效能。本章計時可能含 JIT、快取與軌跡抽樣成本，尚不能單獨歸因於處理器加速。

## 8. 來源

- [D22] [NVIDIA Noisy Simulation](https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html)：噪聲模型、依閘注入通道，以及期望值與抽樣。
- [D23] [NVIDIA Noisy Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/noisy.html)：CPU 密度矩陣與 GPU 軌跡模擬。

查閱日期 2026-09-07；以安裝的 CUDA-Q 0.15.1、解析式與實測共同核對通道定義。完整索引見 [REFERENCES.md](../../REFERENCES.md)。

已保存的 GPU 抽樣程序結束時可能出現 `cudaErrorCudartUnloading`；退出碼為 0 且分布檢查通過，但該結束訊息的根因尚未定位。

接續：[Day27｜CPU vs GPU Quantum Simulation](../day27/README.md)。

本日核對指定末端 Pauli 通道下的密度矩陣、有限次抽樣，以及凍結 Wine VQC 的輸出變化。準確率可維持不變而 Brier 變差；相位翻轉在此 ZZ 插入點上不改預測。結果限於所述通道、位置與模擬器，不是完整硬體噪聲模型，也不是噪聲感知訓練。

## 延伸研究

[N9] Antonio Anna Mele et al. “Noise-induced shallow circuits and the absence of barren plateaus.” Nature Physics 22, 751–756 (2026)；研究論文。[原始來源](https://doi.org/10.1038/s41567-026-03245-z)；[完整書目](../../REFERENCES.md#n9)。

本章觀察噪聲對結果的影響；該研究提醒噪聲類型與量測範圍會改變理論結論。本章的 Pauli 通道不能直接套用非保單位噪聲的結論。
