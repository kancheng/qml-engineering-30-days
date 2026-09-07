# Day29｜Simulator → QPU 本地預演報告

**全部結果來自本地執行；未提交遠端工作，沒有QPU排隊、費用或硬體時間實測。**

CUDA-Q 0.15.1、Python 3.12.7；OMP_NUM_THREADS=1、CUDAQ_DEFAULT_SIMULATOR=qpp。

## 實驗設定

固定theta=π/3、2qubits；Z／X兩基底；128／1024／8192 shots；seeds42–46。每模式30次sample，共93,440 shots。
三模式合計90次sample、280,320 shots，另有三次32-shot位元順序probe與8次exact observe核對。

| Mode | 執行性質 | Noise | Exact reference |
|---|---|---|---|
| qpp-cpu | local_ideal_simulator | p=0.0 | 解析式＋exact observe |
| ionq-emulate | local_ideal_emulation | p=0.0 | 解析式 |
| density-bitflip | local_noisy_simulator | p=0.08 | 解析式＋exact observe |

![Shot comparison](shots.png)

## 五個Seeds的統計

RMSE相對各模式自己的期望值；不是相對理想電路的硬體誤差。每列只有5次重複，趨勢不保證單調。

| Mode | Shots | Observable | Mean | Reference | RMSE |
|---|---:|---|---:|---:|---:|
| qpp-cpu | 128 | Z0 | 0.46875 | 0.50000 | 0.08209 |
| qpp-cpu | 128 | ZZ | 1.00000 | 1.00000 | 0.00000 |
| qpp-cpu | 128 | X0 | -0.03125 | 0.00000 | 0.07057 |
| qpp-cpu | 128 | XX | 0.88750 | 0.86603 | 0.05241 |
| qpp-cpu | 1024 | Z0 | 0.48711 | 0.50000 | 0.03194 |
| qpp-cpu | 1024 | ZZ | 1.00000 | 1.00000 | 0.00000 |
| qpp-cpu | 1024 | X0 | -0.01094 | 0.00000 | 0.02698 |
| qpp-cpu | 1024 | XX | 0.87344 | 0.86603 | 0.01044 |
| qpp-cpu | 8192 | Z0 | 0.49360 | 0.50000 | 0.01308 |
| qpp-cpu | 8192 | ZZ | 1.00000 | 1.00000 | 0.00000 |
| qpp-cpu | 8192 | X0 | -0.00874 | 0.00000 | 0.01067 |
| qpp-cpu | 8192 | XX | 0.86558 | 0.86603 | 0.00494 |
| ionq-emulate | 128 | Z0 | 0.46875 | 0.50000 | 0.08209 |
| ionq-emulate | 128 | ZZ | 1.00000 | 1.00000 | 0.00000 |
| ionq-emulate | 128 | X0 | -0.03125 | 0.00000 | 0.07057 |
| ionq-emulate | 128 | XX | 0.88750 | 0.86603 | 0.05241 |
| ionq-emulate | 1024 | Z0 | 0.48711 | 0.50000 | 0.03194 |
| ionq-emulate | 1024 | ZZ | 1.00000 | 1.00000 | 0.00000 |
| ionq-emulate | 1024 | X0 | -0.01094 | 0.00000 | 0.02698 |
| ionq-emulate | 1024 | XX | 0.87344 | 0.86603 | 0.01044 |
| ionq-emulate | 8192 | Z0 | 0.49360 | 0.50000 | 0.01308 |
| ionq-emulate | 8192 | ZZ | 1.00000 | 1.00000 | 0.00000 |
| ionq-emulate | 8192 | X0 | -0.00874 | 0.00000 | 0.01067 |
| ionq-emulate | 8192 | XX | 0.86558 | 0.86603 | 0.00494 |
| density-bitflip | 128 | Z0 | 0.39375 | 0.42000 | 0.06128 |
| density-bitflip | 128 | ZZ | 0.86250 | 0.84000 | 0.05977 |
| density-bitflip | 128 | X0 | -0.03125 | 0.00000 | 0.07057 |
| density-bitflip | 128 | XX | 0.88750 | 0.86603 | 0.05241 |
| density-bitflip | 1024 | Z0 | 0.40586 | 0.42000 | 0.02256 |
| density-bitflip | 1024 | ZZ | 0.83594 | 0.84000 | 0.01629 |
| density-bitflip | 1024 | X0 | -0.01094 | 0.00000 | 0.02698 |
| density-bitflip | 1024 | XX | 0.87344 | 0.86603 | 0.01044 |
| density-bitflip | 8192 | Z0 | 0.41230 | 0.42000 | 0.00978 |
| density-bitflip | 8192 | ZZ | 0.84009 | 0.84000 | 0.00556 |
| density-bitflip | 8192 | X0 | -0.00874 | 0.00000 | 0.01067 |
| density-bitflip | 8192 | XX | 0.86558 | 0.86603 | 0.00494 |

## 驗證與限制

三模式的非對稱|10⟩ probe皆為10:32。CPU理想／density matrix合計8個exact expectations與解析式誤差<1e-10。
全部180個抽樣expectations通過預先指定的Hoeffding family-wise 99%誤差界；此寬鬆檢查用於抓明顯實作錯誤。
各原始JSON另存pointwise 95% Wilson區間；不是180組同時95%區間，也不包含校準漂移或noise模型不確定性。
每次sample的host wall time包含編譯／執行等成本，沒有分離暖機；不作backend效能比較。

## 原始資料與待提交規格

- [CPU counts](qpp-cpu.json)、[IonQ local emulation counts](ionq-emulate.json)、[Synthetic bit-flip counts](density-bitflip.json)
- [Summary](summary.json)、[未提交的工作規格](submission_plan.json)
- [文章與重跑](../../articles/day29/README.md)
