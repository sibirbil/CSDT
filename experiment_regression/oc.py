
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import gurobipy as gp
from gurobipy import GRB
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from IPython.display import display
from IPython.display import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from csdt import CSDT, split_criteria_with_methods
SEED = 0
np.random.seed(SEED)


base_folder = os.getcwd()
# Define hyperparameters
csdt_min_samples_split = 10
csdt_min_samples_leaf = 5
csdt_depth = 5
verbose = False
class_target_size = 7

# Define features and targets
features_list = ['EnrolledElectiveBefore', 'GradeAvgPrevElec', 'Grade', 'Major', 'Class', 'GradePerm']
target_list = [f'Course{id + 1}' for id in range(class_target_size)]

# Load dataset
df = pd.read_csv('datasets/class_df_size_500_targets_7.csv')
features_df = df[features_list]
target_df = df[target_list]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=SEED)
initial_solution = np.zeros(target_df.shape[1])

def formulate_and_solve_lp_class_data(y, x, lagrangian_multiplier=0, verbose=False,initial_solution=initial_solution):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create model
    model = gp.Model("Minimize SSE")

    # Set Gurobi parameters (disable console logging globally for this model)
    model.setParam("LogToConsole", 0)      # Disable general logging
    model.setParam("OutputFlag", 0)       # Disable detailed output
    model.setParam("MIPGap", 1e-4)        # Optional: set tighter convergence tolerance

    # Create decision variables
    predictions = model.addVars(num_targets, lb=0, ub=100, name="y")
    binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    sse = gp.quicksum(
        (predictions[j] - y[i][j]) * (predictions[j] - y[i][j])
        for i in range(num_instances) for j in range(num_targets)
    )

    # Add constraints
    model.addConstr(binary_vars.sum() <= 1, "one_prediction_constraint")
    for i in range(num_targets):
        model.addConstr(predictions[i] <= 110 * binary_vars[i], f"z_relationship_{i}")

    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)
    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}, Indicator = {binary_vars[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])
    return preds
    
def calculate_mse(y, predictions,initial_solution):
    return mean_squared_error(y, predictions)

use_hashmaps = True
use_initial_solution = True


split_criteria = lambda y, x,solution: split_criteria_with_methods(
    y, x, pred=formulate_and_solve_lp_class_data, split_criteria=calculate_mse , initial_solutions= solution
)
verbose = False
tree = CSDT(
    max_depth=csdt_depth,
    min_samples_leaf=csdt_min_samples_leaf,
    min_samples_split=csdt_min_samples_split,
    split_criteria=split_criteria,
    verbose=verbose,
    use_hashmaps= True,
    use_initial_solution= True
    
)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
csdt_mse = mean_squared_error(y_test, y_pred)
print(f"CSDT MSE: {csdt_mse}")

output_folder = "results"

os.makedirs(output_folder, exist_ok=True)

csdt_output_path = os.path.join(output_folder, 'class_csdt')
dot = tree.draw_tree()
dot.render(csdt_output_path, format='png', view=True)


