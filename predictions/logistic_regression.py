import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

# -- Logistic Regression --
lab_enc = preprocessing.LabelEncoder()
training_Y_encoded = lab_enc.fit_transform(Y_train)
log_reg = LogisticRegression(solver='lbfgs', dual=False, max_iter=1000)
log_reg.fit(X_train, training_Y_encoded)

log_reg_pred = log_reg.predict(X_test)

print("-------------- Logistic REGRESSION METRICS: ")
print('R square Logistic = ', metrics.r2_score(Y_test, log_reg_pred))
print('Mean squared Error Logistic = ', metrics.mean_squared_error(Y_test, log_reg_pred))
print('Mean absolute Error Logistic = ', metrics.mean_absolute_error(Y_test, log_reg_pred))

weight_pred_log = log_reg.predict([[1, 170]])  # Gender,Height
print('Predicted weight Logistic = ', weight_pred_log)


