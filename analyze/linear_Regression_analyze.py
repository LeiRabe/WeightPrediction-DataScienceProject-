from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from Controlleur import CsvControlleur as CsvControlleur
import matplotlib.pyplot as plt

csvCtrl = CsvControlleur.CsvControlleur()
dataset = csvCtrl.readCsv()

X = dataset['Weight'].values.reshape(-1,1)
y = dataset['Height'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
