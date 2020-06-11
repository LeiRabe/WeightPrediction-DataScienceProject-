from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from Controlleur import CsvControlleur as CsvControlleur
import matplotlib.pyplot as plt


csvCtrl = CsvControlleur.CsvControlleur()
dataset = csvCtrl.readCsv()

xPlot = dataset.iloc[:, 1].values
yPlot = dataset.iloc[:, 2].values

lin_regPlot = LinearRegression()
lin_regPlot.fit(X, Y)
plt.scatter(xPlot, yPlot, color='g')
plt.plot(X, lin_regPlot.predict(X))
plt.show()