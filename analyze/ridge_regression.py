import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics

# Load data
dataset = pd.read_csv("datasets_26073_33239_weight-height.csv")

# Convert data
# inches to cm
height = dataset["Height"].tolist()
height_cm = []
for h in height:
    h *= 2.54
    height_cm.append(h)
dataset["Height"] = height_cm
# lbs to kg
weight = dataset["Weight"].tolist()
weight_kg = []
for w in weight:
    w *= 0.453592
    weight_kg.append(w)
dataset["Weight"] = weight_kg

# Convert Gender to number
dataset['Gender'].replace('Female', 0, inplace=True)
dataset['Gender'].replace('Male', 1, inplace=True)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# --- Ridge Regression ---
ridge = Ridge(alpha=1)
ridge.fit(X_train, Y_train)

#Ridge prediction
ridge_pred = ridge.predict(X_test)

# RIDGE REGRESSION METRICS Accuracy
print("-------------- RIDGE REGRESSION METRICS: ")
print('R square Ridge = ', metrics.r2_score(Y_test, ridge_pred))
print('Mean squared Error Ridge = ', metrics.mean_squared_error(Y_test, ridge_pred))
print('Mean absolute Error Ridge = ', metrics.mean_absolute_error(Y_test, ridge_pred))

# Predict weight
weight_pred_ridge = ridge.predict([[1, 170]]) # Gender,Height
print('Predicted weight Linear = ', weight_pred_ridge)