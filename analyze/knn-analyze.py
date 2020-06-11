import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from Controlleur import CsvControlleur as CsvControlleur

csvCtrl = CsvControlleur.CsvControlleur()
dataset = csvCtrl.readCsv()

# Convert Gender to number
dataset['Gender'].replace('Female', 0, inplace=True)
dataset['Gender'].replace('Male', 1, inplace=True)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values


def getmetrics(test_size, n_neighbors):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)
    y = knn.predict(X_test)
    return metrics.r2_score(Y_test, y)


listMNBN = []
for n_neighbors in range(1, 200, 1):
    listMNBN.append(getmetrics(0.2, n_neighbors))

plt.plot(range(1, 200, 1), listMNBN)
plt.xlabel('nb neighbors')
plt.ylabel('score')
plt.show()


listMTS = []
for test_size in numpy.arange(0.1, 1, 0.1):
    listMTS.append(getmetrics(test_size, 5))

plt.plot(numpy.arange(0.1, 1, 0.1), listMTS)
plt.xlabel('test size')
plt.ylabel('score')
plt.show()
