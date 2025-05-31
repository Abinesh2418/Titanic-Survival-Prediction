# 🚢 Titanic Survival Prediction


This project explores the Titanic dataset to predict passenger survival using various machine learning models. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and hyperparameter tuning.

## 📊 Dataset Description

The dataset contains data about passengers aboard the RMS Titanic, including:

- `Pclass` (Passenger class)
- `Sex`
- `Age`
- `SibSp` (Siblings/Spouses aboard)
- `Parch` (Parents/Children aboard)
- `Fare`
- `Embarked` (Port of embarkation)
- `Survived` (Target variable - 0: No, 1: Yes)

---

## 🔧 Preprocessing Steps

1. **Dropped** unnecessary columns: `PassengerId`, `Name`, `Ticket`, `Cabin`
2. **Handled missing values**:
   - Filled `Age` with median
   - Filled `Embarked` with mode
3. **Label Encoding** for categorical variables: `Sex` and `Embarked`

---

## 📈 Exploratory Data Analysis (EDA)

- **Correlation Matrix**: Visualized feature correlations with a heatmap
- **Count Plots**:
  - Survival count
  - Survival by gender
  - Survival by passenger class

---

## 🧠 Model Building

Four different models were trained and evaluated:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost Classifier**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
```

## 🛠️ Hyperparameter Tuning

Used `GridSearchCV` to find the best parameters for each model with **3-fold cross-validation**. Below are the parameters tuned for each model:

- **Logistic Regression**:  
  - `C`: Inverse of regularization strength  
  - `solver`: Algorithm to use in the optimization problem

- **Decision Tree**:  
  - `max_depth`: Maximum depth of the tree  
  - `min_samples_split`: Minimum number of samples required to split an internal node

- **Random Forest**:  
  - `n_estimators`: Number of trees in the forest  
  - `max_depth`: Maximum depth of the tree  
  - `min_samples_split`: Minimum number of samples required to split an internal node

- **XGBoost Classifier**:  
  - `n_estimators`: Number of boosting rounds  
  - `max_depth`: Maximum depth of a tree  
  - `learning_rate`: Step size shrinkage used to prevent overfitting

## 📬 Contact

For any queries or suggestions, feel free to reach out:

- 📧 Email: [abineshbalasubramaniyam@example.com](mailto:abineshbalasubramaniyam@example.com)
- 💼 LinkedIn: [linkedin.com/in/abinesh-b-1b14a1290/](https://www.linkedin.com/in/abinesh-b-1b14a1290/)
