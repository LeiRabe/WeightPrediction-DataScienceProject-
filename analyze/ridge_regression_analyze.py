import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from Controlleur import CsvControlleur as CsvControlleur, MetricControlleur

csvCtrl = CsvControlleur.CsvControlleur()
dataset = csvCtrl.readCsv()
metricCtrl = MetricControlleur.MetricControlleur()

# Convert Gender to number
dataset['Gender'].replace('Female', 0, inplace=True)
dataset['Gender'].replace('Male', 1, inplace=True)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values

listMetrics = []
for alphas in range(1, 7, 1):
    listMetrics.append(metricCtrl.getmetricsAlpha(0.2, alphas, X, Y))

plt.plot(range(1, 7, 1), listMetrics)
plt.xlabel('alpha variation')
plt.ylabel('score')
plt.show()

listTest = []
for test_size in numpy.arange(0.1, 1, 0.1):
    listTest.append(metricCtrl.getmetricsAlpha(test_size, 7, X, Y))

plt.plot(numpy.arange(0.1, 1, 0.1), listTest)
plt.xlabel('test size')
plt.ylabel('score')
plt.show()

#plot de la prediction
predX = dataset['Weight'].values.reshape(-1,1)
predY = dataset['Height'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(predX, predY, test_size=0.2, random_state=0)

regressor = Ridge(alpha=1)
regressor.fit(X_train, y_train.ravel())

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()