# Custom Split Decision Tree (CSDT)

**CSDT** is a Python library designed for building and using decision trees with custom split criteria. It provides flexibility and control for machine learning projects, allowing users to define custom split logic and visualize the resulting trees.


Explore the project on the official website: [Custom Split Decision Tree](https://custom-split-decision-tree.github.io/)

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
cd CSDT
pip install -e .
```

---

## Usage

Here's an example of how to use **CSDT** to create and visualize a decision tree:

```python
import pandas as pd
from csdt import CSDT,split_criteria_with_methods
import numpy as np 

# Sample data
data = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [5, 4, 3, 2, 1],
    "target": [1, 0, 1, 0, 1]
})

X = data[["feature1", "feature2"]]
y = data[["target"]]
def return_mean(y, x):
        return y.mean(axis=0).astype(np.float64)  

def calculate_mse(y, predictions,initial_solutions):
    errors = y - predictions
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)
    return np.float64(mse)
        
split_criteria = lambda y, x,initial_solutions: split_criteria_with_methods(y, x,pred=return_mean, split_criteria= calculate_mse,initial_solutions=initial_solutions
            )
# Initialize the tree
tree = CSDT(max_depth=3, min_samples_split=2, min_samples_leaf=1, verbose=True,split_criteria=split_criteria,use_hashmaps=True)

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
- `gurobi=10.0.3`
- `pip=23.3.1`
- `python=3.11.6`
- `seaborn=0.12.2`
- `scipy=1.11.3`
- `setuptools=68.2.2`

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

