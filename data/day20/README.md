# Iris source data

Source: [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76).
Downloaded 2026-09-07 from [UCI iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).
UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution: R. A. Fisher, Iris dataset, UCI Machine Learning Repository.

`iris.data` is the unmodified downloaded file. The experiment records its SHA-256 in `results/day20/protocol.json`.
Day20 selects versicolor/virginica, removes identical feature-and-label rows before splitting, and preserves original zero-based row IDs. It does not apply the UCI page's setosa correction notes because setosa is excluded from this binary task.
