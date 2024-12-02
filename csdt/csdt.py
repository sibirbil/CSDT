
import time
import numpy as np
import pandas as pd
from graphviz import Digraph


class Node:
    
    def __init__(self):
        self.right = None
        self.left = None
        self.column = None
        self.column_name = None
        self.threshold = None
        self.id = None
        self.depth = None
        self.is_terminal = False
        self.prediction = None
        self.count = None
        self.split_details = None  
        self.class_counts = None  
        self.function = None
        self.error = None
        self.best_score= None
    """
    A node in the decision tree.

    Attributes:
        right (Node): The right child node.
        left (Node): The left child node.
        column (int): The column index of the feature used for splitting.
        column_name (str): The name of the feature used for splitting.
        threshold (float): The threshold value for the split.
        id (int): Unique ID of the node.
        depth (int): The depth of the node in the tree.
        is_terminal (bool): Whether the node is a terminal (leaf) node.
        prediction (any): The prediction value at this node (if terminal).
        count (int): The number of samples at this node.
        split_details (any): Additional details about the split.
        class_counts (any): Class counts at this node (for classification tasks).
        function (any): Custom function applied at the node.
        error (float): The error value at this node.
        best_score (float): The best score achieved at this node.
    """
        

class CSDT:
    """
    Custom Split Decision Tree (CSDT) implementation.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples required in a leaf node.
        min_samples_split (int): Minimum number of samples required to split a node.
        split_criteria (function): A function to evaluate the quality of a split.
        verbose (bool): Whether to display verbose output.
        Tree (Node): The root node of the tree.
    """
    
    def __init__(self, max_depth = 5,
                 min_samples_leaf = 5,
                 min_samples_split = 10,
                 split_criteria = None,
                 verbose = False):
        
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.split_criteria = split_criteria
        self.verbose = verbose
        self.Tree = None
    def buildDT(self, features, labels, node):
        """
        Build the complete decision tree recursively without pruning.

        Args:
            features (pd.DataFrame): The input features for the current node.
            labels (pd.DataFrame): The target labels for the current node.
            node (Node): The current node in the tree.

        Returns:
            None: Modifies the `node` in place to build the tree structure.
        """
        node.prediction, node.error = self.split_criteria(labels.to_numpy(), features.to_numpy())
        node.count = labels.shape[0]

        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        if features.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        split_info, split_gain, n_cuts, best_score = self.calcBestSplitCustom(features, labels)
        node.best_score = best_score 
        if n_cuts == 0:
            node.is_terminal = True
            return

        best_split_idx = np.argmin(split_gain[:, 0])
        splitCol = int(split_info[best_split_idx, 0])
        thresh = split_info[best_split_idx, 1]

        node.column = splitCol
        node.column_name = features.columns[splitCol]
        node.threshold = thresh

        labels_left = labels.loc[features.iloc[:, splitCol] <= thresh, :]
        labels_right = labels.loc[features.iloc[:, splitCol] > thresh, :]

        features_left = features.loc[features.iloc[:, splitCol] <= thresh]
        features_right = features.loc[features.iloc[:, splitCol] > thresh]

        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.id = 2 * node.id

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.id = 2 * node.id + 1

        self.buildDT(features_left, labels_left, node.left)
        self.buildDT(features_right, labels_right, node.right)


    def fit(self, features, labels):
        """
        Train the CSDT model on the given dataset.

        Args:
            features (pd.DataFrame): Feature matrix for training.
            labels (pd.DataFrame): Target values corresponding to the features.

        Returns:
            None: Builds the tree structure and stores predictions for each leaf node.
        """

        start = time.time()

        self.Tree = Node()
        self.Tree.depth = 0
        self.Tree.id = 1
        self.buildDT(features, labels, self.Tree)

        leaves = self.apply(features)
        leaf_predictions = {}
        for leaf_id in np.unique(leaves):
            leaf_indices = np.where(leaves == leaf_id)[0]
            leaf_labels = labels.iloc[leaf_indices].to_numpy()
            leaf_features = features.iloc[leaf_indices].to_numpy()
            leaf_predictions[leaf_id], _ = self.split_criteria(leaf_labels, leaf_features)

        self.leaf_predictions_df = pd.DataFrame(leaf_predictions)

        end = time.time()
        self.training_duration = end - start

    
    def apply(self, features):
        """
        Returns the node ID for each input object based on the tree traversal.

        Args:
            features (pd.DataFrame): The input features for multiple objects.

        Returns:
            np.ndarray: The predicted node IDs for each input object.
        """
        predicted_ids = [self.applySample(features.loc[i], self.max_depth, self.Tree) for i in features.index]
        predicted_ids = np.asarray(predicted_ids)
        return predicted_ids
    def applySample(self, features, depth, node):

        """
        Traverse the tree for a single sample and return the leaf node ID.

        Args:
            features (pd.Series): Input features for a single sample.
            depth (int): Maximum depth to traverse the tree.
            node (Node): Current node during traversal.

        Returns:
            int: Node ID of the leaf node where the sample ends up.
        """
        if node.is_terminal:
            return node.id

        if node.depth == depth:
            return node.id

        if features.iloc[node.column] > node.threshold:
            predicted = self.applySample(features, depth, node.right)
        else:
            predicted = self.applySample(features, depth, node.left)

        return predicted
    def predict(self, features):
        """
        Predict the target values for each input sample.

        Args:
            features (pd.DataFrame): Input feature matrix for prediction.

        Returns:
            np.ndarray: Predicted target values for each input sample.
        """
        leaves = self.apply(features)
        predictions = self.leaf_predictions_df[leaves].T

        return np.asarray(predictions)
    
    
    def calcBestSplitCustom(self, features, labels):
        """
        Find the best feature and threshold for splitting the data at the current node.

        Args:
            features (pd.DataFrame): Input features at the current node.
            labels (pd.DataFrame): Target values at the current node.

        Returns:
            tuple: 
                - split_info (np.ndarray): Information about each split.
                - split_gain (np.ndarray): Gain values for each split.
                - n_cuts (int): Number of possible splits.
                - best_score (float): The best penalty score achieved during split evaluation.
        """
        n = features.shape[0]
        cut_id = 0
        n_obj = 1
        split_perf = np.zeros((n * features.shape[1], n_obj))
        split_info = np.zeros((n * features.shape[1], 2))
        n_features = features.shape[1] 

        best_penalty = float('inf')
        best_feature = -1
        best_threshold = float('inf')
        N_t = len(labels)

        for k in range(features.shape[1]):
                
            x = features.iloc[:, k].to_numpy()
            y = labels.to_numpy()
            sort_idx = np.argsort(x)
            sort_x = x[sort_idx]
            sort_y = y[sort_idx, :]
            
            for i in range(self.min_samples_leaf -1 , n - self.min_samples_leaf  ):
                xi = (sort_x[i] + sort_x[i + 1]) / 2

                if sort_x[i] == sort_x[i + 1]:
     
                    continue

                left_yi = sort_y[:i+1, :]
                right_yi = sort_y[i+1:, :]

                left_xi = features.to_numpy()[sort_idx][:i+1]
                right_xi = features.to_numpy()[sort_idx][i+1:]

                left_instance_count = left_yi.shape[0]
                right_instance_count = right_yi.shape[0]

                left_prediction, left_perf = self.split_criteria(left_yi, left_xi)
                right_prediction, right_perf = self.split_criteria(right_yi, right_xi)

                N_t_L = len(left_yi)
                N_t_R = len(right_yi)
                if N_t_R == 0 or N_t_L == 0:
                    continue

                gain = N_t / n
                node_prediction,node_perf =  self.split_criteria(sort_y, sort_x)
                left_prediction, left_perf = self.split_criteria(left_yi, left_xi)
                right_prediction, right_perf = self.split_criteria(right_yi, right_xi)
                curr_score = (left_perf * left_instance_count + right_perf * right_instance_count) / n

                split_perf[cut_id, 0] = curr_score
                split_info[cut_id, 0] = k
                split_info[cut_id, 1] = xi

                if curr_score < best_penalty:
                    best_penalty = curr_score
                    best_feature = k
                    best_threshold = xi
                elif curr_score == best_penalty:
                    if k < best_feature or (k == best_feature and xi < best_threshold):
                        best_feature = k
                        best_threshold = xi

                cut_id += 1


        split_info = split_info[:cut_id, :]
        split_gain = split_perf[:cut_id, :]
        split_info = split_info[~np.isnan(split_gain).any(axis=1), :]
        split_gain = split_gain[~np.isnan(split_gain).any(axis=1), :]
        n_cuts = cut_id

        best_score = best_penalty  
        return split_info, split_gain, n_cuts, best_score

    def draw_tree(self):
        """
        Create a visual representation of the decision tree using Graphviz.

        Returns:
            Digraph: A Graphviz Digraph object representing the tree structure.
        """
        def add_nodes_edges(dot, node, parent=None, edge_label=""):
            if node is None:
                return

            if node.is_terminal:
                label = f"Pred: {node.prediction}\nCount: {node.count}\nError: {node.error:.3f}"
                dot.node(str(node.id), label, shape="box", width="0.7", height="0.4", fontsize="10")
            else:
                label = f"{node.column_name} <= {node.threshold:.3f}\nCount: {node.count}\nFeature ID: {node.column}\nError: {node.error:.3f}"
                dot.node(str(node.id), label, shape="ellipse", width="0.9", height="0.5", fontsize="10")

            if parent is not None:
                dot.edge(str(parent.id), str(node.id), label=edge_label)

            add_nodes_edges(dot, node.left, node, edge_label="Left")
            add_nodes_edges(dot, node.right, node, edge_label="Right")

        dot = Digraph(comment="Decision Tree")
        add_nodes_edges(dot, self.Tree)

        return dot

def split_criteria_with_methods(y, x, pred, split_criteria): 
    """
    Calculate predictions and evaluate split quality using custom methods.

    Args:
        y (np.ndarray): Target values.
        x (np.ndarray): Input features.
        pred (function): Prediction function.
        split_criteria (function): Function to evaluate split quality.

    Returns:
        tuple: Predictions and split evaluation score.
    """
    predictions = pred(y.astype(np.float64), x.astype(np.float64))  
    predictions_all = (predictions * np.ones((y.shape[0], y.shape[1]), dtype=np.float64))  
    split_evaluation = split_criteria(predictions_all.astype(np.float64), y.astype(np.float64))  

    return predictions, split_evaluation
