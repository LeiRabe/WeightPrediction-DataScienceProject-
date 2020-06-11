from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from Controlleur import CsvControlleur as CsvControlleur
import matplotlib.pyplot as plt

csvCtrl = CsvControlleur.CsvControlleur()
dataset = csvCtrl.readCsv()

# Convert Gender to number
dataset['Gender'].replace('Female', 0, inplace=True)
dataset['Gender'].replace('Male', 1, inplace=True)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regr = svm.SVR()
regr.fit(X_train, Y_train)
regrpred = regr.predict(X_test)

lin_regPlot = svm.SVR()
lin_regPlot.fit(X, Y)
plt.plot(X, lin_regPlot.predict(X), color='k')
plt.show()

# Model Accuracy
print("-------------- SVM METRICS: ")
print('R square Linear = ', metrics.r2_score(Y_test, regrpred))
print('Mean squared Error Linear = ', metrics.mean_squared_error(Y_test, regrpred))
print('Mean absolute Error Linear = ', metrics.mean_absolute_error(Y_test, regrpred))

# Predict weight
weight_pred_lin = regr.predict([[1, 170]]) # Gender,Height
print('Predicted weight SVM = ', weight_pred_lin)