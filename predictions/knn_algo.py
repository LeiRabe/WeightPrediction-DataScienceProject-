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

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

# --- KNN ---
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, Y_train)

neigh_pred = neigh.predict(X_test)

print("-------------- KNN METRICS: ")
print('R square Neighbors = ', metrics.r2_score(Y_test, neigh_pred))
print('Mean squared Error Neighbors = ', metrics.mean_squared_error(Y_test, neigh_pred))
print('Mean absolute Error Neighbors = ', metrics.mean_absolute_error(Y_test, neigh_pred))

weight_pred_neigh = neigh.predict([[1, 170]])  # Gender,Height
print('Predicted weight Neighbors = ', weight_pred_neigh)
