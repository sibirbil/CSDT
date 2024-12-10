# Custom Split Decision Tree (CSDT)

**CSDT** is a Python library designed for building and using decision trees with custom split criteria. It provides flexibility and control for machine learning projects, allowing users to define custom split logic and visualize the resulting trees.

---

## Features

- **Custom Split Logic**: Easily define custom split criteria for decision tree nodes.
- **Tree Visualization**: Generate high-quality tree visualizations using `graphviz`.
- **Flexible Splitting Criteria**: Works with user-defined splitting functions and evaluation metrics.
- **Seamless Integration**: Fully compatible with Python data science libraries like `pandas` and `scikit-learn`.

---

## Installation

You can install the library via pip:

```bash
pip install csdt
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone https://github.com/sibirbil/CSDT.git
cd Custom-Split-Decision-Tree
pip install -e .
```

---

## Usage

Here's an example of how to use **CSDT** to create and visualize a decision tree:

```python
import pandas as pd
from csdt import CSDT

# Sample data
data = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [5, 4, 3, 2, 1],
    "target": [1, 0, 1, 0, 1]
})

X = data[["feature1", "feature2"]]
y = data[["target"]]

# Initialize the tree
tree = CSDT(max_depth=3, min_samples_split=2, min_samples_leaf=1, verbose=True)

# Fit the tree
tree.fit(X, y)

# Visualize the tree
dot = tree.draw_tree()
dot.render("decision_tree", format="png")
```

This code will create a tree visualization and save it as `decision_tree.png`.

---

## Requirements

To use **CSDT**, make sure you have the following dependencies installed:

- `matplotlib==3.8.1`
- `numpy==2.1.0`
- `pandas==2.2.3`
- `scikit-learn==1.3.2`
- `graphviz`

If you use `conda`, you can create an environment with all required dependencies:

```bash
conda env create -f csdt.yml
conda activate csdt
```

---

## Conda Environment Setup

For Conda users, you can create an environment using the provided `csdt.yml` file:

1. Download or copy the `csdt.yml` file.
2. Create the environment:
   ```bash
   conda env create -f csdt.yml
   ```
3. Activate the environment:
   ```bash
   conda activate csdt
   ```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! If you'd like to contribute, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## Authors

- **Çağla Mıdıklı** - [GitHub](https://github.com/cagla0117)
- **İlker Birbil** - [GitHub](https://github.com/sibirbil)
- **Doğanay Özese** - [GitHub](https://github.com/dozese)


---

## Acknowledgments

Special thanks to all contributors and the open-source community for their support in making this project possible.
```

---
