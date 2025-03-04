import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os
from sklearn import tree
import sys
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

    csdt_depth = 15
    features_list = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31']
    target_list = ['target']

    df = pd.read_csv(os.path.join(base_folder, "datasets/student.csv"))
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
            print("Hata: Boş veri seti üzerinde işlem yapılamaz.")
            return None
        y_flat = np.ravel(y).astype(np.int64)  
        if np.bincount(y_flat).size == 0:
            print("Hata: Frekans hesaplaması için yeterli veri yok.")
            return None
        y_new = np.bincount(y_flat).argmax()
        return [np.int64(y_new)]
    
    split_criteria = lambda y, x,initial_solutions: split_criteria_with_methods(y, x, pred=return_majority, split_criteria=calculate_gini,initial_solutions=initial_solutions)
    
    tree = CSDT(max_depth=csdt_depth, min_samples_leaf=csdt_min_samples_leaf, min_samples_split=csdt_min_samples_split,
                split_criteria=split_criteria, verbose=verbose,use_hashmaps=True,use_initial_solution=True)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df['leaf_id'] = tree.apply(X_test)
    y_pred_df = y_pred_df.drop_duplicates()
    
    ocdt_mse = calculate_gini(y_test, y_pred,0)
    print(f'CSDT Gini: {ocdt_mse}')

    classifier = DecisionTreeClassifier( min_samples_leaf=csdt_min_samples_leaf,
                                        min_samples_split=csdt_min_samples_split, max_depth=csdt_depth)
    
    classifier.fit(X_train, y_train)
    

    
    y_pred_sklearn = classifier.predict(X_test)
    y_pred_sklearn_df = pd.DataFrame(y_pred_sklearn, columns=target_list)
    dt_mse = calculate_gini(y_test, y_pred_sklearn,0)
    print(f'Sklearn DT Gini: {dt_mse}')
    

    output_folder = "results"

    os.makedirs(output_folder, exist_ok=True)

    csdt_output_path = os.path.join(output_folder, 'csdt_classifier')
    dot = tree.draw_tree()
    dot.render(csdt_output_path, format='png', view=True)

    plt.figure(figsize=(15, 10))
    plot_tree(classifier, filled=True)
    sklearn_output_path = os.path.join(output_folder, 'sklearn_classifier_dt.png')
    plt.savefig(sklearn_output_path, format='png')
