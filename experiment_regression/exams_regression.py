import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os
from sklearn import tree
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from csdt import CSDT, split_criteria_with_methods

SEED = 0
np.random.seed(SEED)

base_folder = os.getcwd()

from sklearn.metrics.pairwise import euclidean_distances

if __name__ == '__main__':
    csdt_min_samples_split = 10
    csdt_min_samples_leaf = 5
    verbose = False

    csdt_depth = 15
    features_list = ['gender', 'race/ethnicity', 'parental level of education',
                     'lunch', 'test preparation course']
    target_list = ['math score','reading score','writing score']

    df = pd.read_csv(os.path.join(base_folder, "datasets/constrained_exams.csv"))
    features_df = df[features_list]
    target_df = df[target_list]

    features_df = pd.get_dummies(features_df, columns=features_df.columns, drop_first=True, dtype=int)
    print("Number of features after one-hot encoding:", features_df.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        features_df.astype(np.float64),  
        target_df.astype(np.float64),   
        test_size=0.2,
        random_state=SEED
    )

    def return_mean(y, x):
        return y.mean(axis=0).astype(np.float64)  

    def calculate_mse(y, predictions,initial_solutions):
        mse = mean_squared_error(y, predictions)
        return np.float64(mse)  

    split_criteria = lambda y, x,initial_solutions: split_criteria_with_methods(
        y.astype(np.float64), x.astype(np.float64), pred=return_mean, split_criteria=calculate_mse
    )

    tree = CSDT(max_depth=csdt_depth, min_samples_leaf=csdt_min_samples_leaf, min_samples_split=csdt_min_samples_split,
                split_criteria=split_criteria,
                verbose=verbose, use_hashmaps=True,use_initial_solution=True)

    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, dtype=np.float64)  
    y_pred_df['leaf_id'] = tree.apply(X_test)
    y_pred_df = y_pred_df.drop_duplicates()

    csdt_mse = mean_squared_error(y_test, y_pred)
    print(f'CSDT MSE: {csdt_mse}')

    regressor = DecisionTreeRegressor(
        random_state=20,
        min_samples_leaf=csdt_min_samples_leaf,
        min_samples_split=csdt_min_samples_split,
        max_depth=csdt_depth
    )

    regressor.fit(X_train, y_train)
    y_pred_sklearn = regressor.predict(X_test)
    y_pred_sklearn_df = pd.DataFrame(y_pred_sklearn, columns=target_list, dtype=np.float64)  # Ensure 64-bit precision

    dt_mse = mean_squared_error(y_test, y_pred_sklearn)
    print(f'Sklearn DT MSE: {dt_mse}')
    output_folder = "results"

    os.makedirs(output_folder, exist_ok=True)

    csdt_output_path = os.path.join(output_folder, 'exams_csdt')
    dot = tree.draw_tree()
    dot.render(csdt_output_path, format='png', view=True)

    plt.figure(figsize=(15, 10))
    plot_tree(regressor, filled=True)

    sklearn_output_path = os.path.join(output_folder, 'exams_sklearn_dt.png')
    plt.savefig(sklearn_output_path, format='png', dpi=300, bbox_inches='tight')
