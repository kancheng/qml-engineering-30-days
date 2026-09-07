# Day27 CPU／GPU 實測報告

![Latency](latency.png)

同一個exact global-Z observe、同一weights，backend依序執行，不同時跑CPU與GPU。OMP_NUM_THREADS=1；每設定first1次、warmup2次、量測7次。

## 環境

CPU：11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz。
GPU run後快照：NVIDIA GeForce RTX 3060 Laptop GPU, 570.211.01, 6144 MiB, 53。
CUDA-Q：0.15.1；Python：3.12.7。

## First call與Warm latency

| Backend | Qubits | Blocks | First ms | Median ms | Q25 ms | Q75 ms |
|---|---:|---:|---:|---:|---:|---:|
| qpp-cpu | 4 | 4 | 40.3229 | 0.2180 | 0.2167 | 0.2274 |
| qpp-cpu | 4 | 12 | 0.6512 | 0.4298 | 0.4280 | 0.4422 |
| qpp-cpu | 8 | 4 | 2.6292 | 2.5186 | 2.4992 | 2.5472 |
| qpp-cpu | 8 | 12 | 7.5046 | 7.4470 | 7.4168 | 7.4535 |
| qpp-cpu | 12 | 4 | 80.8782 | 78.7158 | 77.6148 | 79.3277 |
| qpp-cpu | 12 | 12 | 233.6785 | 230.4207 | 229.7288 | 231.7110 |
| qpp-cpu | 16 | 4 | 2176.4321 | 2159.3767 | 2155.7428 | 2166.7009 |
| qpp-cpu | 16 | 12 | 6442.3800 | 6442.4541 | 6436.5393 | 6454.8383 |
| nvidia-fp64 | 4 | 4 | 50.3812 | 0.3557 | 0.3520 | 0.3598 |
| nvidia-fp64 | 4 | 12 | 1.0985 | 0.6104 | 0.6065 | 0.6324 |
| nvidia-fp64 | 8 | 4 | 3.2424 | 0.7380 | 0.7256 | 0.7426 |
| nvidia-fp64 | 8 | 12 | 2.1316 | 1.6859 | 1.6789 | 1.7238 |
| nvidia-fp64 | 12 | 4 | 3.6938 | 1.2231 | 1.1976 | 1.2640 |
| nvidia-fp64 | 12 | 12 | 3.1686 | 2.9851 | 2.9363 | 3.0589 |
| nvidia-fp64 | 16 | 4 | 3.1523 | 1.9608 | 1.9591 | 1.9938 |
| nvidia-fp64 | 16 | 12 | 6.0996 | 5.7890 | 5.7624 | 5.8124 |
| nvidia | 4 | 4 | 49.1471 | 0.3318 | 0.3274 | 0.3765 |
| nvidia | 4 | 12 | 0.8033 | 0.6085 | 0.5999 | 0.6211 |
| nvidia | 8 | 4 | 3.0495 | 0.7536 | 0.6955 | 0.8232 |
| nvidia | 8 | 12 | 2.0210 | 1.7243 | 1.6754 | 1.8340 |
| nvidia | 12 | 4 | 3.3293 | 1.2120 | 1.1776 | 1.3912 |
| nvidia | 12 | 12 | 3.8054 | 3.0867 | 2.8880 | 3.1855 |
| nvidia | 16 | 4 | 3.0546 | 1.8378 | 1.6724 | 1.9522 |
| nvidia | 16 | 12 | 5.2575 | 4.7608 | 4.4089 | 4.8294 |

First是該設定在此process的第一次呼叫，不保證cold compiler／disk cache；只有process首個case包含此process首次kernel執行。Process總wall time另存，包含Python import、輸出、setup與驗證。

## CPU median / GPU median

| Qubits | Blocks | GPU mode | Ratio (>1 GPU較快) | Max expectation error |
|---:|---:|---|---:|---:|
| 4 | 4 | nvidia-fp64 | 0.613 | 8.327e-17 |
| 4 | 4 | nvidia | 0.657 | 1.077e-07 |
| 4 | 12 | nvidia-fp64 | 0.704 | 4.996e-16 |
| 4 | 12 | nvidia | 0.706 | 1.250e-08 |
| 8 | 4 | nvidia-fp64 | 3.413 | 3.747e-16 |
| 8 | 4 | nvidia | 3.342 | 3.209e-08 |
| 8 | 12 | nvidia-fp64 | 4.417 | 0.000e+00 |
| 8 | 12 | nvidia | 4.319 | 5.180e-08 |
| 12 | 4 | nvidia-fp64 | 64.357 | 4.163e-17 |
| 12 | 4 | nvidia | 64.947 | 1.198e-08 |
| 12 | 12 | nvidia-fp64 | 77.190 | 8.674e-17 |
| 12 | 12 | nvidia | 74.649 | 1.546e-08 |
| 16 | 4 | nvidia-fp64 | 1101.300 | 3.990e-17 |
| 16 | 4 | nvidia | 1175.007 | 2.959e-09 |
| 16 | 12 | nvidia-fp64 | 1112.885 | 3.990e-17 |
| 16 | 12 | nvidia | 1353.236 | 7.259e-09 |

主比較為CPU fp64對GPU fp64；nvidia預設fp32是另一個precision條件，不把差異全歸因於GPU。
Ratio只適用本機、此電路、此thread設定與量測期間，不能當成通用加速倍數。IQR是7次重複的描述統計，不是confidence interval。

## 限制

最高16qubits的單份statevector下限為fp64 1MiB、fp32 0.5MiB，未量測peak RAM／VRAM，也未測試6GiB可容納的最大qubit數。
固定backend順序與固定case順序，未鎖頻、未控制OS背景負載或散熱；後段case可能受thermal／cache影響。CPU單thread不是最佳CPU調校。
相同參數反覆執行，不是QML training、batch throughput、noise trajectory或QPU benchmark；沒有手動關閉backend gate fusion。
GPU退出可能有cudaErrorCudartUnloading；請以保存的數值驗證與process exit status判斷本次執行，未定位該退出訊息根因。

## 原始紀錄

- [Protocol](protocol.json)、[Process wall times](process_times.json)、[Comparisons](comparison.json)
- [qpp-cpu records](qpp-cpu/records.json)、[qpp-cpu summary](qpp-cpu/summary.json)
- [nvidia-fp64 records](nvidia-fp64/records.json)、[nvidia-fp64 summary](nvidia-fp64/summary.json)
- [nvidia records](nvidia/records.json)、[nvidia summary](nvidia/summary.json)
- [教學與重跑](../../articles/day27/README.md)
