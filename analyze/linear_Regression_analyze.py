from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from Controlleur import CsvControlleur as CsvControlleur, MetricControlleur
import matplotlib.pyplot as plt
import numpy

csvCtrl = CsvControlleur.CsvControlleur()
dataset = csvCtrl.readCsv()
metricCtrl = MetricControlleur.MetricControlleur()

X = dataset['Weight'].values.reshape(-1,1)
y = dataset['Height'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()

#plot analise
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values
listTest = []
for test_size in numpy.arange(0.1, 1, 0.1):
    listTest.append(metricCtrl.getmetrics(LinearRegression, test_size, X, Y))

plt.plot(numpy.arange(0.1, 1, 0.1), listTest)
plt.xlabel('test size')
plt.ylabel('score')
plt.show()