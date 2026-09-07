# Day 03 示範環境

本日使用 repository 內的 `.venv`，Python 3.12 與 NumPy 2.2.6。套件固定於 `requirements-day03.txt`，不使用 base 環境的 NumPy。實際執行版本由 `experiment.py` 寫入 `results/day03/summary.json`。

## 使用者提供的設備紀錄

來源：使用者於 2026-09-06 提供的 Ubuntu 終端輸出；此表不是實驗程式自動探測結果。

| 項目 | 值 |
|---|---|
| 作業系統 | Ubuntu（發行版本尚未提供） |
| GPU | NVIDIA GeForce RTX 3060 筆電 GPU |
| GPU 記憶體 | 6144 MiB |
| NVIDIA driver | 570.211.01 |
| `nvidia-smi` 顯示的 CUDA Version | 12.8 |
| `nvcc -V` | CUDA compilation tools 12.8，V12.8.93 |

## 本日實驗範圍

- Backend：NumPy CPU state-vector simulation。
- State：單量子位元、`complex128`。
- Measurement：精確機率與期望值，不使用有限 shots。
- Random seed：不適用；角度掃描沒有隨機性。
- GPU／CUDA-Q：本日程式未使用，這份紀錄不代表已完成 CUDA-Q backend 驗證。

## 重現指令

在 repository 根目錄執行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-day03.txt
python articles/day03/demo.py
python articles/day03/experiment.py
python -m unittest discover -s articles/day03 -p 'test_*.py' -v
```
