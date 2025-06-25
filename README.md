# Credit Default Prediction using K-Nearest Neighbors

This project uses machine learning to predict whether a person will default on a loan, based on income, age, and loan amount. The model is trained using the **K-Nearest Neighbors (KNN)** algorithm and evaluated with both **train/test split** and **cross-validation**.

---

## 📁 Dataset

The dataset is expected to be a CSV file named `credit_data.csv` with the following columns:

- `income` (float): Monthly income of the individual
- `age` (int): Age of the individual
- `loan` (float): Amount of loan taken
- `default` (int): Target label (1 for default, 0 for non-default)

---

## 🚀 How It Works

### Main Steps:
1. Load the dataset using `pandas`.
2. Extract the features (`income`, `age`, `loan`) and target (`default`).
3. Scale the features to a 0–1 range using `MinMaxScaler`.
4. Split the dataset into **training** and **test** sets.
5. Train a `KNeighborsClassifier` using the training data.
6. Predict on the test data and evaluate with:
   - `confusion_matrix`
   - `accuracy_score`
7. Use **cross-validation** to test many values of **K (neighbors)** from 1 to 99 to find the optimal value.

---

## 🧪 Requirements

Install with pip:

```bash
pip install pandas numpy scikit-learn

🧠 Finding the Best k

A loop from 1 to 99 tries different values of k and uses 10-fold cross-validation to compute the average accuracy. The best k is printed:
cross_val_score(model, x, y, cv=10)

📊 Output Example

best k is : 18
[[104  12]
 [  6  78]]
Accuracy: 0.91
