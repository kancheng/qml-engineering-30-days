# UCI Wine source data

Source: [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), DOI [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J).
Attribution: S. Aeberhard and M. Forina (1992), Wine [Dataset], UCI Machine Learning Repository.
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as listed by UCI.
Downloaded 2026-09-07 from [wine.data](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data).

`wine.data` is the unmodified source file: 178 rows, class identifier followed by 13 chemical measurements. This is Wine cultivar classification, not the separate Wine Quality dataset.
Day25 preselects original class 2 (label0) and class3 (label1), retaining119 rows. The loader checks exact feature+label duplicates; none are found. Original zero-based row IDs are preserved. The file SHA-256 and feature order are saved in `results/day25/protocol.json`.

Features, in source order: alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, proline. Measurements use source conventions; unlike Iris these are not all cm values.
