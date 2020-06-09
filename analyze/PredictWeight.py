import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils

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
print("X-train: ", X_train)
print("X-test: ", X_test)
print("Y_train: ", Y_train)
print("Y_test: ", Y_test)

# --- Fit Regression Model ---
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

# Make Prediction using test data
lin_pred = lin_reg.predict(X_test)

# Model Accuracy
print("-------------- LINEAR REGRESSION METRICS: ")
print('R square Linear = ', metrics.r2_score(Y_test, lin_pred))
print('Mean squared Error Linear = ', metrics.mean_squared_error(Y_test, lin_pred))
print('Mean absolute Error Linear = ', metrics.mean_absolute_error(Y_test, lin_pred))

# Predict weight
# weight_pred_lin = lin_reg.predict([[1, 170]])
# print('Predicted weight Linear = ', weight_pred_lin)

# --- KNN ---
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, Y_train)

neigh_pred = neigh.predict(X_test)

print("-------------- KNN METRICS: ")
print('R square Neighbors = ', metrics.r2_score(Y_test, neigh_pred))
print('Mean squared Error Neighbors = ', metrics.mean_squared_error(Y_test, neigh_pred))
print('Mean absolute Error Neighbors = ', metrics.mean_absolute_error(Y_test, neigh_pred))

# weight_pred_neigh = neigh.predict([[1, 170]])
# print('Predicted weight Neighbors = ', weight_pred_neigh)

# -- Logistic Regression --
lab_enc = preprocessing.LabelEncoder()
training_Y_encoded = lab_enc.fit_transform(Y_train)
log_reg = LogisticRegression()
log_reg.fit(X_train, training_Y_encoded)

log_reg_pred = log_reg.predict(X_test)

print("-------------- LOGISTIC REGRESSION METRICS: ")
print('R square Neighbors = ', metrics.r2_score(Y_test, log_reg_pred))
print('Mean squared Error Neighbors = ', metrics.mean_squared_error(Y_test, log_reg_pred))
print('Mean absolute Error Neighbors = ', metrics.mean_absolute_error(Y_test, log_reg_pred))
