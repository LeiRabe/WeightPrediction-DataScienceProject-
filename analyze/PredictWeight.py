# Load data
import pandas as pd
dataset = pd.read_csv("weight-height.csv")

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
y = dataset.iloc[:, 2].values

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make Prediction using test data
lin_pred = lin_reg.predict(X_test)

# Model Accuracy
from sklearn import metrics
print('R square = ', metrics.r2_score(y_test, lin_pred))
print('Mean squared Error = ', metrics.mean_squared_error(y_test, lin_pred))
print('Mean absolute Error = ', metrics.mean_absolute_error(y_test, lin_pred))

# Predict weight
my_weight_pred = lin_reg.predict([[1, 170]])
print('My predicted weight = ', my_weight_pred)
