# Day 24｜Barren Plateau：為什麼 QNN 學不動？

[Day17](../day17/README.md) 確認gradient怎麼算；今天問另一件事：**計算正確的gradient，是否仍小到難以估計或更新？** 我們比較qubit數、電路block數、cost locality與initialization，不執行分類器訓練。

交付：[landscape.py](landscape.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[plot_results.py](plot_results.py)、[測試](test_landscape.py)、[實測報告](../../results/day24/README.md)。沿用獨立`.venv`與既有套件。

## 1. Loss沒下降，不等於Barren Plateau

Barren Plateau研究關注在指定的電路與初始化分布下，梯度集中在零附近，且梯度variance可隨qubit數呈指數下降。McClean等人的結果與隨機電路及2-design性質有關，並非宣稱所有QNN都無法訓練。[P6]

單次gradient=0還可能是stationary point、參數與observable對易、參數不在cost的因果範圍、程式漏接參數，或數值差分失準。Loss停滯也可能來自optimizer、預算、資料或loss設計。因此本日固定參數位置，在多個初始化樣本上看分布，而不是挑一條平坦training curve就下結論。

## 2. 固定Protocol

| 項目 | 本日設定 |
|---|---|
| Qubits n | 2、4、6、8 |
| Blocks L | 1、4、8 |
| 每個block | 每線RY→RZ，再CZ(0,1)、CZ(1,2)…CZ(n−2,n−1) |
| 量子參數 | 2nL，每個rotation獨立 |
| Initial states | 全部從零態開始 |
| 初始化 | Uniform(−π,π)；Normal(0,0.1) |
| Seeds | 每組0…31，32個樣本 |
| 梯度座標 | 第一個block、qubit0的RY，weights[0] |
| Local cost | C_local=⟨Z0⟩ |
| Global cost | C_global=⟨Z0 Z1 … Z(n−1)⟩ |
| Shots／noise | Exact，無噪聲 |

四種n×三種L×兩種初始化×32seeds，共**768個電路初始化**；每個算兩種cost的同一個梯度，共**1,536個gradient values**。沒有Iris、accuracy或train／test切分，本日是landscape診斷；classical可解析對照見第4節。

L是block數，不是compiled circuit depth。每block有2n個單qubit gates、n−1個CZ；statevector有2^n個complex amplitudes。GPU記憶體在本日n≤8並非瓶頸，也不做GPU加速結論。

兩個cost都在[-1,1]，避免把sum與mean的任意scale差異當成梯度消失；但它們代表不同objective。將global換成local可能改變任務，不能保證原本的解仍正確。[P7] 本日不是該論文local 2-design ansatz的完整復現。

## 3. 三條路徑核對Gradient

NumPy參考實作同步傳播state與單一tangent state：

```text
|ψ'> = G|ψ>
|dψ'> = G|dψ> + (dG/dθ)|ψ>  # 第二項只在weights[0]對應的gate加入
∂C/∂θ = 2 Re⟨dψ|O|ψ⟩
dRY/dθ = (-iY/2) RY
```

NumPy以tensor-axis操作套用單qubit gates，不建立大型2^n×2^n矩陣。Tensor排序把qubit0放在最高位；CUDA-Q直接使用spin.z(0)指定observable，避免把backend內部位元順序當成相同陣列排序。

CUDA-Q獨立使用parameter-shift：

```text
g = [C(θ0+π/2) − C(θ0−π/2)] / 2
```

此規則適用於本日獨立RY參數，不是把完整loss隨意位移。每個gradient兩次observe，兩backend各**3,072次observe**；保存兩個shifted expectations，不只保存相減後的值。測試與demo另以NumPy central finite difference h=1e-5核對。

## 4. 可解析對照：不用深電路也會有Global梯度縮小

考慮product state `⊗_j RY(θ_j)|0⟩`，θ_j獨立uniform於[−π,π]：

```text
C_local = cos θ0
C_global = ∏_j cos θ_j

g_local = −sin θ0
g_global = −sin θ0 × ∏_(j>0) cos θ_j
```

由E[sin θ]=0、E[sin² θ]=E[cos² θ]=1/2，可直接推導：

```text
E[g_local] = E[g_global] = 0
Var(g_local) = 1/2
Var(g_global) = (1/2)^n
```

這是此指定分布與cost的精確ensemble結論，不是用四個資料點擬合所得。模型可classically計算，卻已能展示global cost的梯度variance呈指數縮小。

![Product-state control](../../results/day24/product_control.png)

主實驗depth1的RZ與CZ都對Z讀出對易，因此對兩種cost也有相同解析形式。CZ可能改變state的糾纏，但此時不改變這些expectations；不要把depth1的下降歸因於「電路很深」。測試會直接核對depth1解析式。

對照的32個samples另用長度n的uniform向量；主實驗用長度2nL、交錯RY／RZ的向量，因此相同seed並非逐筆相同角度。兩者分布上的理論比較仍成立。

## 5. 如何讀Variance圖

對固定(n,L,initialization,cost)的32個gradient，報告mean、sample variance(ddof=1)、RMS、median absolute gradient，以及`|g|<0.01`比例。0.01只是描述性門檻，不是Barren Plateau判定標準。

![Gradient variance](../../results/day24/variance.png)

本次uniform組的global variance隨n下降；local反應不同。小角度組沒有呈現相同下降趨勢，但不能據此宣稱可擴展訓練已解決。每組僅32樣本、最大8qubits，沒有估confidence interval或做scaling fit；曲線連線只幫助閱讀，不表示證明單調或指數關係。

不同設定重用seed，因此曲線可能相關；它們不是大量獨立重複benchmark。固定index0不代表完整gradient norm，也不代表所有參數。Local observable的因果範圍與所選參數位置都會影響結果。

完整數值與驗證誤差見[結果報告](../../results/day24/README.md)，不只公布有下降的曲線。

## 6. 小角度初始化不是萬用解

本日`small`是Normal(0,0.1)，不是由inverse blocks構成的identity-block初始化。即使所有rotation為0，CZ仍是非identity unitary，只是它對初始全零態沒有作用。

在全零weights，兩個cost皆1且gradient=0；測試明確驗證此stationary point。若最小化本日cost，這也不是想要的最小值。因此「把weights設小」可能避開某些隨機分布的集中，也可能靠近stationary區域，不能只看variance較小就說更容易訓練。

實際排查可依序核對：參數是否影響輸出、parameter-shift與reference是否一致、cost與observable的scale／locality、跨初始化分布、qubit與depth依賴，最後再比較optimizer和初始化的訓練結果。本日只完成前述gradient診斷，不聲稱已驗證mitigation的最終accuracy。

## 7. Exact Gradient與Shot成本的差別

本日全部是exact simulator。若未來用獨立S次shots估每個Pauli expectation，parameter-shift估計的variance為：

```text
Var(g_hat) = [(1−C_plus²)+(1−C_minus²)] / (4S) ≤ 1/(2S)
```

這是在±1 outcomes與兩組獨立shots的假設下推導，不是本日實測。典型sampling error按1/√S縮小；gradient很小時，區分訊號與shot fluctuation可能需要更多shots。增加optimizer iterations本身不會改善單次gradient估計的解析度。

GPU fp32與CPU fp64會有浮點差異，故驗證用absolute tolerance=1e-5，不在接近0的gradient上算不穩定relative error。本日沒有noise-induced barren plateau實驗，不能把浮點誤差叫做量子noise。

## 8. 重跑與示範

沿用[requirements-day24.txt](../../requirements-day24.txt)，不新增套件：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day24/experiment.py run
OMP_NUM_THREADS=1 python articles/day24/experiment.py verify
OMP_NUM_THREADS=1 python articles/day24/experiment.py verify --backend nvidia
OMP_NUM_THREADS=1 python articles/day24/plot_results.py

OMP_NUM_THREADS=1 python articles/day24/demo.py --qubits 4 --depth 4 --initialization uniform
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day24 -p 'test_*.py' -v
OMP_NUM_THREADS=1 DAY24_TARGET=nvidia python -m unittest discover -s articles/day24 -p 'test_*.py' -v
```

Demo可加`--seed`、`--backend nvidia`，比較兩種cost的tangent／parameter-shift／finite difference。CPU可獨立執行run／verify／demo／tests；完整圖表報告需兩backend summaries。

輸出在`results/day24/`，重跑覆寫同名檔案；重新run後再verify與plot。artifact SHA-256綁定protocol與samples，防止引用舊的backend summary。GPU退出有`cudaErrorCudartUnloading`，但驗證與測試exit code=0，根因未定位。

## 9. 下一步與文獻

Day25將進入第二個真實資料集benchmark，把前面對表示、模型與成本的限制帶回分類任務。

- [P6] Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, Hartmut Neven. “Barren plateaus in quantum neural network training landscapes.” *Nature Communications* 9, 4812 (2018). [arXiv與期刊資訊](https://arxiv.org/abs/1803.11173)，DOI 10.1038/s41467-018-07090-4。
- [P7] M. Cerezo, Akira Sone, Tyler Volkoff, Lukasz Cincio, Patrick J. Coles. “Cost function dependent barren plateaus in shallow parametrized quantum circuits.” *Nature Communications* 12, 1791 (2021). [arXiv v3](https://arxiv.org/abs/2001.00550v3)，DOI 10.1038/s41467-021-21728-w。

查閱日期2026-09-07；[REFERENCES.md](../../REFERENCES.md)保存版本與支持範圍。本日僅引用原始研究的概念與條件，不宣稱此CZ電路滿足論文全部假設。

接續：[Day25｜第二個真實資料集：Wine](../day25/README.md)。
