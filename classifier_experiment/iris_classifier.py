import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os
import sys
from sklearn import tree

from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from csdt import CSDT, split_criteria_with_methods


SEED = 0
np.random.seed(SEED)

base_folder = os.getcwd()

if __name__ == '__main__':
    csdt_min_samples_split = 10
    csdt_min_samples_leaf = 5
    number_of_folds = 5
    verbose = False

    csdt_depth = 3
    features_list = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    target_list = ['target']
    df = pd.read_csv(os.path.join(base_folder, "datasets/iris.csv"))
    features_df = df[features_list]
    target_df = df[target_list]
    
    X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=SEED)
   
    def calculate_gini(prediction, y,initial_solutions):
        if y.size == 0 or np.unique(y).size == 1:
            return 0

        if np.all(y == y[0]):
            return 0

        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def return_majority(y, x):
        if y.size == 0:
            print("Error: Empty Dataset")
            return None
        y_flat = np.ravel(y).astype(np.int64) 
        if np.bincount(y_flat).size == 0:
            print("Error: Dataset is not enough")
            return None
        y_new = np.bincount(y_flat).argmax()
        return [np.int64(y_new)]  
    split_criteria = lambda y, x,initial_solutions: split_criteria_with_methods(y, x, pred=return_majority, split_criteria=calculate_gini,initial_solutions=initial_solutions)
        
    csdt_tree = CSDT(max_depth=csdt_depth, min_samples_leaf=csdt_min_samples_leaf, min_samples_split=csdt_min_samples_split,
                     split_criteria=split_criteria, verbose=verbose,use_hashmaps=True)
    csdt_tree.fit(X_train, y_train)
    y_pred = csdt_tree.predict(X_test)
    
    csdt_gini = calculate_gini(y_test, y_pred,0)
    print(f'CSDT Gini: {csdt_gini}')


    classifier = DecisionTreeClassifier(random_state=20, min_samples_leaf=csdt_min_samples_leaf,
                                        min_samples_split=csdt_min_samples_split, max_depth=csdt_depth)
    
    classifier.fit(X_train, y_train)
    y_pred_sklearn = classifier.predict(X_test)
    output_folder = "results"

    os.makedirs(output_folder, exist_ok=True)
    sk_gini = calculate_gini(y_test, y_pred_sklearn,0)
    print(f'Sklearn Gini: {sk_gini}')
    csdt_output_path = os.path.join(output_folder, 'iris_csdt')
    dot = csdt_tree.draw_tree()
    dot.render(csdt_output_path, format='png', view=True)

    plt.figure(figsize=(15, 10))
    plot_tree(classifier, filled=True)
    sklearn_output_path = os.path.join(output_folder, 'iris_sklearn_dt.png')
    plt.savefig(sklearn_output_path, format='png')

    


