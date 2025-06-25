import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
import os

def load_data(file_path):
    """
    Loads credit data from a CSV file into a Pandas DataFrame.
    If the file is not found, a dummy CSV is created for demonstration.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if an error occurs
                          and a dummy file cannot be created.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure 'credit_data.csv' is in the same directory as the script.")
        print("Creating a dummy 'credit_data.csv' for demonstration purposes.")
        # Create a dummy CSV for demonstration if not found
        dummy_data = {
            'income': [50000, 60000, 30000, 70000, 45000, 55000, 65000, 25000, 80000, 40000,
                       52000, 48000, 75000, 32000, 68000, 28000, 58000, 72000, 38000, 62000],
            'age': [30, 45, 22, 50, 35, 28, 40, 60, 33, 25,
                    31, 44, 23, 51, 36, 29, 41, 61, 34, 26],
            'loan': [10000, 20000, 5000, 30000, 8000, 15000, 25000, 3000, 40000, 7000,
                     12000, 18000, 28000, 6000, 35000, 4000, 22000, 32000, 9000, 11000],
            'default': [0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
                        0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 0 = no default, 1 = default
        }
        pd.DataFrame(dummy_data).to_csv(file_path, index=False)
        print("Dummy 'credit_data.csv' created. Please replace it with your actual data.")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def prepare_features_target(dataframe, feature_cols, target_col):
    """
    Separates the DataFrame into features (X) and target (y).

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        feature_cols (list): A list of column names to be used as features.
        target_col (str): The name of the target column.

    Returns:
        tuple: A tuple containing (features (X), target (y)) as NumPy arrays.
    """
    # Convert features DataFrame and target Series to NumPy arrays
    # Reshape features to be 2D array if it's currently 1D
    # The -1 in reshape means "infer the size of this dimension"
    x = np.array(dataframe[feature_cols])
    y = np.array(dataframe[target_col])
    return x, y

def scale_features(x_data):
    """
    Scales the features using MinMaxScaler.
    MinMaxScaler transforms features by scaling each feature to a given range (default 0 to 1).
    This is often important for distance-based algorithms like K-NN.

    Args:
        x_data (numpy.ndarray): The feature data to be scaled.

    Returns:
        numpy.ndarray: The scaled feature data.
    """
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x_data)
    print("\n--- Scaled Features (first 5 rows) ---")
    print(x_scaled[:5])
    return x_scaled

def split_data(x_data, y_data, test_size=0.3, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        x_data (numpy.ndarray): The feature data.
        y_data (numpy.ndarray): The target data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.
                            Ensures reproducibility.

    Returns:
        tuple: A tuple containing (features_train, features_test, target_train, target_test).
    """
    features_train, features_test, target_train, target_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=random_state
    )
    return features_train, features_test, target_train, target_test

def train_and_evaluate_knn(features_train, target_train, features_test, target_test, n_neighbors=20):
    """
    Trains a K-Nearest Neighbors classifier and evaluates its performance.

    Args:
        features_train (numpy.ndarray): Training features.
        target_train (numpy.ndarray): Training target.
        features_test (numpy.ndarray): Testing features.
        target_test (numpy.ndarray): Testing target.
        n_neighbors (int): The number of neighbors to consider for classification.

    Returns:
        numpy.ndarray: The predictions made on the test set.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    fitted_model = model.fit(features_train, target_train)
    prediction = fitted_model.predict(features_test)

    print(f"\n--- Model Evaluation (n_neighbors={n_neighbors}) ---")
    print('Confusion Matrix:')
    print(confusion_matrix(target_test, prediction))
    print('Accuracy Score:')
    print(accuracy_score(target_test, prediction))
    return prediction

def find_best_k_with_cross_validation(x_data, y_data, k_range=range(1, 100), cv_folds=10):
    """
    Performs K-fold cross-validation to find the optimal 'k' for KNeighborsClassifier.

    Args:
        x_data (numpy.ndarray): The full feature dataset.
        y_data (numpy.ndarray): The full target dataset.
        k_range (range): The range of 'k' values to test (e.g., range(1, 100)).
        cv_folds (int): The number of folds for cross-validation.

    Returns:
        int: The 'k' value that yielded the highest average accuracy.
    """
    cross_valid_scores = []
    print(f"\n--- Finding Best 'k' using {cv_folds}-Fold Cross-Validation ---")
    for k in k_range:
        train_model = KNeighborsClassifier(n_neighbors=k)
        # cv: Determines the cross-validation splitting strategy.
        # scoring: A string (see scikit-learn docs for all options), or a scorer callable.
        scores = cross_val_score(train_model, x_data, y_data, cv=cv_folds, scoring='accuracy')
        score_mean = scores.mean()
        cross_valid_scores.append(score_mean)
        # Optional: print progress
        # if k % 10 == 0 or k == k_range.stop - 1:
        #     print(f"K={k}: Avg Accuracy = {score_mean:.4f}")

    best_k_index = np.argmax(cross_valid_scores)
    best_k = k_range[best_k_index] # Get the actual k value
    print(f'Best K found through cross-validation: {best_k} (with average accuracy: {cross_valid_scores[best_k_index]:.4f})')
    return best_k

def main():
    """
    Main function to execute the K-Nearest Neighbors classification workflow.
    """
    CSV_FILE_PATH = 'credit_data.csv'
    FEATURE_COLUMNS = ['income', 'age', 'loan']
    TARGET_COLUMN = 'default'
    TEST_DATA_SPLIT_RATIO = 0.3
    RANDOM_STATE_FOR_SPLIT = 42 # For reproducibility of train/test split

    # 1. Load Data
    data_df = load_data(CSV_FILE_PATH)
    if data_df is None:
        return # Exit if data loading failed

    # 2. Prepare Features and Target
    # Convert Pandas DataFrame/Series to NumPy arrays for scikit-learn
    x_features_raw, y_target_raw = prepare_features_target(data_df, FEATURE_COLUMNS, TARGET_COLUMN)
    print("\n--- Original Features (first 5 rows) ---")
    print(x_features_raw[:5])
    print("\n--- Original Target (first 5 values) ---")
    print(y_target_raw[:5])

    # 3. Preprocess Data: Scale Features
    # Scaling is crucial for distance-based algorithms like K-NN.
    x_scaled = scale_features(x_features_raw)

    # 4. Split Data into Training and Testing Sets
    features_train, features_test, target_train, target_test = split_data(
        x_scaled, y_target_raw, test_size=TEST_DATA_SPLIT_RATIO, random_state=RANDOM_STATE_FOR_SPLIT
    )
    print(f"\nData split: {len(features_train)} training samples, {len(features_test)} testing samples.")

    # 5. Find Optimal 'k' using Cross-Validation
    best_k = find_best_k_with_cross_validation(x_scaled, y_target_raw)

    # 6. Train and Evaluate K-NN Model with the optimal 'k'
    # Use the best_k found from cross-validation for the final model
    print(f"\n--- Training final K-NN model with best_k={best_k} ---")
    final_prediction = train_and_evaluate_knn(features_train, target_train, features_test, target_test, n_neighbors=best_k)

if __name__ == "__main__":
    main()
