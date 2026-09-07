# Day 26｜沒有完美 Qubit：用 CUDA-Q 模擬 Quantum Noise

Day25使用exact、無噪聲電路；今天加入bit flip、phase flip、depolarizing channel，分開比較**物理noise模型造成的分布改變**與**有限shots的抽樣波動**。

程式：[noise.py](noise.py)、[experiment.py](experiment.py)、[demo.py](demo.py)、[tests](test_noise.py)、[plot_results.py](plot_results.py)。結果見[實測報告](../../results/day26/README.md)。沿用`.venv`，無新增套件。

## 1. 從State Vector到Density Matrix

純態可寫為ρ=|ψ⟩⟨ψ|。Noise channel用Kraus operators描述：

```text
ρ' = Σ_k K_k ρ K_k†
Σ_k K_k† K_k = I
```

這是ensemble狀態，不是任意對state amplitudes加隨機數。NumPy實作Kraus演算，測試trace=1、PSD與completeness。Density matrix有4^n個complex entries，比statevector的2^n更耗記憶體；本日只有兩qubit。

## 2. 三種Channel與p的定義

| Channel | ρ' | 直觀作用 |
|---|---|---|
| Bit flip | (1−p)ρ+pXρX | 機率p套用X |
| Phase flip | (1−p)ρ+pZρZ | 機率p套用Z |
| Depolarizing | (1−p)ρ+(p/3)(XρX+YρY+ZρZ) | 合計p發生非identity Pauli |

以上為本機CUDA-Q `BitFlipChannel`、`PhaseFlipChannel`、`DepolarizationChannel`的實測慣例。[D22] Depolarizing將Bloch向量縮放為1−4p/3，所以p=.75時完全混合；p=1時縮放為−1/3，不是I/2。Bit／phase flip的p=1也代表確定套用Pauli，不是最大混合。

不可把另一份文件使用的`(1−p)ρ+pI/2`定義直接代入。本日用解析式及實測核對端點，沒有依名稱猜參數意義。

## 3. Noise放在哪裡很重要

本日先準備|00⟩、|+0⟩或Bell Φ+，再只在q0加入一次channel：

```python
noise = cudaq.NoiseModel()
noise.add_channel('rz', [0], cudaq.PhaseFlipChannel(0.3))
```

電路末端明確執行`rz(0.0,q[0])`作為注入位置；零角度gate不改變理想state，但對應的noise會執行。準備與量測基底轉換不使用RZ，因此只命中一次。測試確認noise確實生效，沒有被最佳化或不支援的target忽略。

採用本機已驗證的gate-based NoiseModel API。最新文件另有kernel內的`apply_noise`寫法；本日沒有依賴它。不要把這個人為末端channel叫做已校準的RTX／QPU硬體error model，也不是所有gates後自動加noise。

## 4. Z Counts不變，State也可能已改變

對Bell state，理想ZZ=XX=1。q0的bit flip使ZZ→1−2p但XX不變；phase flip使XX→1−2p但ZZ不變；depolarizing使兩者皆乘1−4p/3。

所以只觀察Z-basis的00／11分布，會漏看phase noise。實驗同時保存Z與X-basis的counts；X-basis透過量測前對兩線加H實作。Basis gates本身在本日是理想的。

三states×三channels×五個p(0,.1,.3,.75,1)×四個observables(Z0,X0,ZZ,XX)，共180筆exact核對。NumPy參考與CUDA-Q `density-matrix-cpu`應在浮點誤差內一致。

## 5. Exact、有限Shots與GPU Trajectories

官方列出density matrix CPU與GPU trajectory的noisy simulation方式。[D23] 本日使用：

- `density-matrix-cpu`：exact observe與finite-shot sample。
- `nvidia`：finite-shot noisy sample；不稱作GPU exact density matrix。

每個state／channel／p，在Z、X基底各用4096shots、seeds42／43／44，共270筆records／backend，1,105,920shots。比較四個bitstring頻率與NumPy精確機率，保存全部counts。

採用最大bin-frequency誤差<.06作smoke threshold；這是預先固定工程容差，不是統計confidence interval，也不保證每種隨機重跑都通過。相同seed不保證不同backend抽出相同counts。增加shots減少抽樣波動，不會消除noise channel造成的期望值偏移。

## 6. 回到Day25：凍結Wine VQC

讀取Day25選定VQC的weights、preprocessor與test samples，於完整VQC後、ZZ readout前加同一q0 channel。兩splits各26筆，三channels×p(0,.1,.3)，共18筆批次records、468次exact observe；加上基礎實驗，合計648次。

若理想z=⟨ZZ⟩，末端noise的解析式是：

```text
bit flip:     z'=(1−2p)z
phase flip:   z'=z
depolarizing: z'=(1−4p/3)z
p(label1)=(1−z')/2
```

所有參數與test資料凍結，不重訓、不依noise挑seed。保存Day25來源hash，並與解析式核對。Classical對照為NumPy density matrix；Wine另沿用保存的train-prior baseline，不讓classical分數通過量子channel。

本日選取的Wine noise強度使縮放因子仍為正，通常不改變0.5threshold的分類決策，Brier卻可能變差。Phase flip末端與ZZ對易，完全不改變此readout；若在Ansatz中間注入，結果可能不同。這不是模型對phase noise普遍免疫，也不是noise-aware training。

![Noise effects](../../results/day26/noise_effects.png)

## 7. 重跑與示範

依賴入口：[requirements-day26.txt](../../requirements-day26.txt)。完整實驗使用Day25已保存的模型檔：

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python articles/day26/experiment.py run
OMP_NUM_THREADS=1 python articles/day26/experiment.py sample
OMP_NUM_THREADS=1 python articles/day26/experiment.py sample --backend nvidia
OMP_NUM_THREADS=1 python articles/day26/plot_results.py

OMP_NUM_THREADS=1 python articles/day26/demo.py --channel phase_flip --probability 0.3
OMP_NUM_THREADS=1 python -m unittest discover -s articles/day26 -p 'test_*.py' -v
```

Demo預設density-matrix CPU，顯示Bell的ZZ／XX與兩種basis counts；可調shots與seed。CPU可獨立完成exact／sample／demo／tests，完整圖表報告需要GPU sample summary。輸出在`results/day26/`，重跑更新同名檔案；修改protocol或Day25模型後應全部重跑，避免舊counts與新設定混用。

五項測試涵蓋CPTP／density matrix、解析channel、phase flip讀出差異、depolarization端點、確定性抽樣與錯誤輸入。本日沒有amplitude damping、readout error、correlated noise或QPU校準；目前結果僅適用於指定末端Pauli channels。

Day27再做CPU／GPU simulation效能比較；本日時間含可能的JIT／cache與trajectory抽樣，不作隔離speedup結論。

## 8. 來源

- [D22] [NVIDIA Noisy Simulation](https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html)：NoiseModel、gate-based注入與observe／sample。
- [D23] [NVIDIA Noisy Simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/noisy.html)：density-matrix CPU與GPU trajectory方式。

查閱日期2026-09-07；本日以安裝的CUDA-Q0.15.1 docstrings、解析式及實際執行共同核對channel慣例。完整[REFERENCES.md](../../REFERENCES.md)。

本機GPU抽樣程序退出時有`cudaErrorCudartUnloading`訊息；exit code=0且全部分布檢查通過，根因未定位。

接續：[Day27｜CPU vs GPU Quantum Simulation](../day27/README.md)。
