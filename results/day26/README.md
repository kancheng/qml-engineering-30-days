# Day26 Noise 實測報告

![Noise effects](noise_effects.png)

## 精確驗證

Density-matrix CPU：180筆基礎observable records＋18筆Wine records，共648次observe。
基礎最大誤差=4.441e-16，Wine解析對照最大誤差=3.331e-16。

## 有限shots

| Backend | Records | Shots each | Total shots | Max bin-frequency error | Passed |
|---|---:|---:|---:|---:|---|
| density-matrix-cpu | 270 | 4096 | 1105920 | 0.015137 | True |
| nvidia | 270 | 4096 | 1105920 | 0.017090 | True |

每record對照四種bitstrings；固定absolute tolerance .06是工程smoke threshold，不是confidence interval。GPU是noisy finite-shot sampling，不是exact density matrix。CPU與GPU不要求同seed產生相同counts。

## 凍結Wine VQC

| Split | Channel | p | Test Brier | Accuracy |
|---|---|---:|---:|---:|
| 2030 | bit_flip | 0.0 | 0.032404 | 96.15% |
| 2030 | bit_flip | 0.1 | 0.050538 | 96.15% |
| 2030 | bit_flip | 0.3 | 0.124883 | 96.15% |
| 2030 | phase_flip | 0.0 | 0.032404 | 96.15% |
| 2030 | phase_flip | 0.1 | 0.032404 | 96.15% |
| 2030 | phase_flip | 0.3 | 0.032404 | 96.15% |
| 2030 | depolarizing | 0.0 | 0.032404 | 96.15% |
| 2030 | depolarizing | 0.1 | 0.043083 | 96.15% |
| 2030 | depolarizing | 0.3 | 0.081364 | 96.15% |
| 2031 | bit_flip | 0.0 | 0.057050 | 92.31% |
| 2031 | bit_flip | 0.1 | 0.077436 | 92.31% |
| 2031 | bit_flip | 0.3 | 0.145514 | 92.31% |
| 2031 | phase_flip | 0.0 | 0.057050 | 92.31% |
| 2031 | phase_flip | 0.1 | 0.057050 | 92.31% |
| 2031 | phase_flip | 0.3 | 0.057050 | 92.31% |
| 2031 | depolarizing | 0.0 | 0.057050 | 92.31% |
| 2031 | depolarizing | 0.1 | 0.069629 | 92.31% |
| 2031 | depolarizing | 0.3 | 0.106924 | 92.31% |

Noise只在末端q0注入，並非每個gate皆有error。Phase flip與ZZ對易，這裡不改變預測；不表示完整VQC對phase noise免疫。
本日bit flip p≤.3與depolarizing p≤.3使ZZ乘正縮放，因此0.5threshold的分類通常保持，Brier仍改變。這是特定插入位置的解析效果，不是noise-aware training。
Classical參考為NumPy Kraus density matrix與Day25保存的train-prior constant baseline；baseline不經量子channel。

## 紀錄

- [Protocol](protocol.json)、[Exact records](exact.json)、[Summary](summary.json)
- [Wine scores](wine.json)、[Wine source hashes](wine_source.json)
- [CPU counts](density-matrix-cpu/samples.json)、[CPU summary](density-matrix-cpu/summary.json)
- [GPU counts](nvidia/samples.json)、[GPU summary](nvidia/summary.json)
- [教學與重跑](../../articles/day26/README.md)
