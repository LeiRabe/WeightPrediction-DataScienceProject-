import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from Controlleur import CsvControlleur as CsvControlleur

csvCtrl = CsvControlleur.CsvControlleur()
dataset = csvCtrl.readCsv()

# Convert Gender to number
dataset['Gender'].replace('Female', 0, inplace=True)
dataset['Gender'].replace('Male', 1, inplace=True)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values


def getmetrics(test_size, alphas):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    ridge = Ridge(alpha=alphas)
    y = ridge.fit(X_train, Y_train).predict(X_test)
    return metrics.r2_score(Y_test, y)


listMetrics = []
for alphas in range(1, 7, 1):
    listMetrics.append(getmetrics(0.2, alphas))

plt.plot(range(1, 7, 1), listMetrics)
plt.xlabel('alpha variation')
plt.ylabel('score')
plt.show()

listTest = []
for test_size in numpy.arange(0.1, 1, 0.1):
    listTest.append(getmetrics(test_size, 7))

plt.plot(numpy.arange(0.1, 1, 0.1), listTest)
plt.xlabel('test size')
plt.ylabel('score')
plt.show()
