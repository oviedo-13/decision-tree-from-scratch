import numpy as np
import pandas as pd
from collections import Counter

class Node:
    """
    Represents a node in a decision tree.
    Each node contains information about a split or a leaf.

    Attributes:
        feature: The feature index used for splitting at this node.
        threshold: The threshold value for the feature used for splitting.
        left: The left child node (subtree).
        right: The right child node (subtree).
        label: The predicted label if this node is a leaf.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        # Initialization of instance attributes
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


    def is_leaf(self):
        """
        Check if the current node is a leaf node.

        Returns:
            bool: True if it's a leaf node, False otherwise.
        """
        return self.label is not None
    

class DecisionTree:
    """
    Represents a decision tree classifier.

    Attributes:
        min_samples_split: The minimum number of samples required to split a node.
        max_depth: The maximum depth of the tree.
        root: The root node of the decision tree.
    """

    def __init__(self, min_samples_split=2, max_depth=None):
        # Initialization of instance attributes
        self.min_sample_split = min_samples_split
        self.max_depth = float("inf") if max_depth is None else max_depth
        self.root = None

    
    def train(self, X, y):
        """
        Train the decision tree on the given dataset.

        Args:
            X (DataFrame): The input features.
            y (Series): The target labels.
        """
        # Start growing the tree and store the root node
        self.root = self._grow_tree(X, y)

    
    def predict(self, X):
        """
        Predict labels for a set of input samples.

        Args:
            X (DataFrame): The input features.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        # Traverse the tree for each row and predict a target value.
        return np.array([self._traverse_tree(X.loc[i], self.root) for i in X.index])

    def structure(self):
        """
        Get the tree's structure as a dictionary.

        Returns:
            dict: The tree structure.
        """
        # Buid dictionary recursively starting from root node
        repr = self._build_representation(self.root)
        return repr

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Args:
            X (DataFrame): The input features.
            y (Series): The target labels.
            depth (int): The current depth of the tree.

        Returns:
            Node: The root node of the subtree.
        """
        # Get the number of samples (rows) and unique labels
        n_samples = X.shape[0] # Number of samples in the current subset
        n_labels = len(np.unique(y)) # Number of different unique labels in the target variable

        # Leaf node conditions:
        # 1. If there are fewer samples than the minimum required for a split.
        # 2. If there is only one unique label left (pure node).
        # 3. If the current depth exceeds the maximum allowed depth.
        if n_samples < self.min_sample_split or n_labels == 1 or depth >= self.max_depth:
             # Create a leaf node with the most common label in the current subset
            counter = Counter(y)
            leaf_label = counter.most_common(1)[0][0]

            return Node(label=leaf_label)
        
        # If the current node is not a leaf node, find the best feature and threshold to split on
        best_feature, best_threshold = self._find_best_split(X, y)

        # Split the dataset into left and right subsets based on the best split
        left_vals = X.loc[X[best_feature] <= best_threshold]
        right_vals = X.loc[X[best_feature] > best_threshold]

        # Recursively grow the left and right subtrees
        left = self._grow_tree(left_vals, y.loc[X[best_feature] <= best_threshold], depth+1)
        right = self._grow_tree(right_vals, y.loc[X[best_feature] > best_threshold], depth+1)

        # Create and return a node representing the best split
        return Node(best_feature, best_threshold, left, right)

    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold for splitting.

        Args:
            X (DataFrame): The input features.
            y (Series): The target labels.

        Returns:
            tuple: (best_feature, best_threshold).
        """
        # Initialize variables to track the best feature, threshold, and information gain
        best_gain = -float("inf")  # Initialize with negative infinity to ensure improvement
        best_threshold = None

        # Loop through each feature in the input data
        for feature in X.columns:
            thresholds = np.unique(X[feature]) # Get unique values of the current feature as potential thresholds
            
            # Loop through each unique value of the feature as a potential threshold
            for threshold in thresholds:

                # Calculate the information gain for the current split
                gain = self._information_gain(X[feature], y, threshold)

                 # Check if the current gain is better than the best recorded gain
                if gain > best_gain:
                    # Update the best feature and threshold with the current values
                    best_feature = feature
                    best_threshold = threshold
                    best_gain = gain

        # Return the best feature and threshold found
        return best_feature, best_threshold


    def _information_gain(self, x_col, y, threshold):
        """
        Calculate information gain for a given split.

        Args:
            x_col (Series): The column of the input feature.
            y (Series): The target labels.
            threshold (float): The threshold value for the split.

        Returns:
            float: Information gain.
        """
        # Calculate the entropy of the parent node
        parent_entropy = self._entropy(y)

        # Split the data into left and right child nodes based on the given threshold
        left_data, right_data = self._split(x_col, threshold)

        # Calculate the entropy of the left and right child nodes
        left_entropy = self._entropy(y.loc[left_data.index])
        right_entropy = self._entropy(y.loc[right_data.index])

        # Calculate the weighted average entropy of the children
        n, n_l, n_r = len(y), len(left_data), len(right_data)

        children_entropy = (n_l / n) * left_entropy + (n_r / n) * right_entropy

        # Calculate the information gain by subtracting the children's entropy from the parent's entropy
        gain = parent_entropy - children_entropy

        return gain

    def _split(self, x_col, threshold):
        """
        Split a column into left and right data based on a threshold.

        Args:
            x_col (Series): The column of the input feature.
            threshold (float): The threshold value for the split.

        Returns:
            tuple: (left_data, right_data).
        """
         # Create a subset of the input column containing values less than or equal to the threshold
        left_data = x_col.loc[x_col <= threshold]
        # Create a subset of the input column containing values greater than the threshold
        right_data = x_col.loc[x_col > threshold]

        # Return the left and right subsets as a tuple
        return left_data, right_data


    def _entropy(self, y):
        """
        Calculate the entropy of a set of labels.

        Args:
            y (Series): The target labels.

        Returns:
            float: Entropy.
        """
        # Calculate the probabilities of each unique label in the target labels
        probabilities = np.bincount(y) / len(y)
        # Calculate the entropy using the formula: -Σ(p * log2(p)) for each non-zero probability
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    

    def _traverse_tree(self, x, node):
        """
        Recursively traverse the decision tree to make predictions.

        Args:
            x (Series): The input features for a single sample.
            node (Node): The current node in the tree.

        Returns:
            int: Predicted label.
        """
         # Check if the current node is a leaf node
        if node.is_leaf():
            # Return the label associated with the leaf node as the prediction
            return node.label

        # If not a leaf node, determine whether to move left or right in the tree
        if x[node.feature] <= node.threshold:
            # If the input feature value is less than or equal to the node's threshold,
            # traverse the left subtree
            return self._traverse_tree(x, node.left)
        # If the input feature value is greater than the node's threshold,
        # traverse the right subtree
        return self._traverse_tree(x, node.right)
    

    def _build_representation(self, node):
        """
        Recursively build a dictionary representation of the decision tree structure.

        Args:
            node (Node): The current node in the tree.

        Returns:
            dict: Dictionary representation of the tree structure.
        """
        # Check if the current node is a leaf node
        if node.is_leaf():
            # If it's a leaf node, return a dictionary with the label
            return {"label": node.label}
        
        # If it's not a leaf node, build a dictionary representing the current node
        return {
            "feature": node.feature,         # Feature used for splitting at this node
            "thrs": node.threshold,          # Threshold value for the split
            "left": self._build_representation(node.left),   # Recursively build the left subtree
            "right": self._build_representation(node.right)  # Recursively build the right subtree
        }


if __name__ == "__main__":
     # Import necessary libraries and modules
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.datasets import load_wine

    # Loading data

    data = load_wine()  # Load the Wine dataset
    df = pd.DataFrame(data.data, columns=data.feature_names) # Create a DataFrame from the dataset features
    df['target'] = data.target # Add the target column to the DataFrame

    X = df.drop("target", axis=1) # Features (X)
    y = df["target"] # Target variable (y)

    # Model generalization
    X_train, X_test, y_train, y_test = None, None, None, None # Initialize variables for training and testing data
    tree = None  # Initialize the decision tree model
    y_pred = None  # Initialize predicted labels

    accuracy_lst = [] # Create an empty list to store accuracy scores

    # Perform 10 different train-test splits and evaluate the model
    for i in range(10):
        # Splitting into training set and test set with random state fixed at each iteration so that results are reproducible but different each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i*15)

        # Create a decision tree model
        tree = DecisionTree(min_samples_split=10, max_depth=5)

        # Train the model on the training data
        tree.train(X_train, y_train)

        # Make predictions on the test data
        y_pred = tree.predict(X_test)

        # Calculate and store the accuracy score
        accuracy_lst.append(accuracy_score(y_test, y_pred))

    # Create a bar chart to visualize accuracy for each test
    # The chart shows that de model generalizes adequately
    fig, ax = plt.subplots()
    tests = [f'Test {i+1}' for i in range(1, 11)]
    ax.bar(tests, accuracy_lst, label=tests)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Test number')
    ax.set_title('Accuracy for 10 tests')

    # Model evaluation
    # Displaying the mean accuracy provides an overall measure of how well the model performs
    # across multiple test splits. It gives us a sense of the model's generalization performance.
    print("\n----- Model metrics -----")
    print("Generalization: The bar chart shows that the model's accuracy in the 10 tests is greater than 0.8, so it can be concluded that the model generalizes adequately.")
    print("Mean accuracy: ", np.mean(accuracy_lst)) # Calculate and print the mean accuracy
    print()

    # The confusion matrix provides a detailed breakdown of the model's performance in terms
    # of true positives, true negatives, false positives, and false negatives for each class.
    # This helps us understand where the model is making correct or incorrect predictions.
    confusion_mtx = confusion_matrix(y_test, y_pred) # Calculate the confusion matrix
    print("Confusion matrix: ")
    print(confusion_mtx)
    print()

    # The classification report offers a comprehensive summary of various classification
    # metrics, including precision, recall, F1-score, and support for each class.
    # It provides insights into the model's performance on a per-class basis, which is
    # especially useful when dealing with imbalanced datasets.
    class_report = classification_report(y_test, y_pred, target_names=['class 0', 'class 1', 'class 2']) # Generate the classification report
    print("Classification report: ")
    print(class_report)
    print()

    print("Tree structure: ")
    print(tree.structure())
    print()

    plt.show() # Display the bar chart
