
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
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, RangeSet, Binary, SolverFactory, minimize

import sys
base_folder = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(base_folder, "..")))
from csdt import CSDT, split_criteria_with_methods

SEED = 0
np.random.seed(SEED)

base_folder = os.getcwd()
# Define hyperparameters
ocdt_min_samples_split = 10
ocdt_min_samples_leaf = 5
ocdt_depth = 5
verbose = False
class_target_size = 7

# Define features and targets
features_list = ['EnrolledElectiveBefore', 'GradeAvgPrevElec', 'Grade', 'Major', 'Class', 'GradePerm']
target_list = [f'Course{id + 1}' for id in range(class_target_size)]

# Load dataset
df = pd.read_csv('../datasets/class_df_size_500_targets_7.csv')
features_df = df[features_list]
target_df = df[target_list]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=SEED)
def formulate_and_solve_lp_class_data(y, x, lagrangian_multiplier=0, verbose=False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create Pyomo model
    model = ConcreteModel()
    model.targets = RangeSet(0, num_targets - 1)
    model.instances = RangeSet(0, num_instances - 1)

    # Decision variables
    model.predictions = Var(model.targets, bounds=(0, 100))
    model.binary_vars = Var(model.targets, within=Binary)

    # Objective function: Minimize Sum of Squared Errors (SSE)
    def objective_rule(model):
        return sum(
            (model.predictions[j] - y[i][j]) ** 2
            for i in model.instances for j in model.targets
        )
    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Constraints
    def one_prediction_constraint(model):
        return sum(model.binary_vars[j] for j in model.targets) <= 1
    model.one_prediction_constraint = Constraint(rule=one_prediction_constraint)

    def z_relationship_constraint(model, j):
        return model.predictions[j] <= 110 * model.binary_vars[j]
    model.z_relationship_constraints = Constraint(model.targets, rule=z_relationship_constraint)

    # Solve the model
    solver = SolverFactory('gurobi')  #'gurobi', 'cbc', 'ipopt' 
    results = solver.solve(model, tee=verbose)

    if verbose:
        print("Optimal Solution:")
        for j in model.targets:
            print(f"Target {j + 1}: Prediction = {model.predictions[j].value}, Indicator = {model.binary_vars[j].value}")
        print("Objective (Sum of Squared Errors):", model.objective())

    preds = np.array([model.predictions[j].value for j in model.targets])
    return preds

def calculate_mse(y, predictions):
    return mean_squared_error(y, predictions)

split_criteria = lambda y, x: split_criteria_with_methods(
    y, x, pred=formulate_and_solve_lp_class_data, split_criteria=calculate_mse
)
verbose = False
tree = CSDT(
    max_depth=ocdt_depth,
    min_samples_leaf=ocdt_min_samples_leaf,
    min_samples_split=ocdt_min_samples_split,
    split_criteria=split_criteria,
    verbose=verbose
)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
ocdt_mse = mean_squared_error(y_test, y_pred)
print(f"CSDT MSE: {ocdt_mse}")
