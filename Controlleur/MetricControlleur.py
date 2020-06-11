from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import Ridge


class MetricControlleur:
    def getmetricsAlpha(self, test_size, alphas, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
        ridge = Ridge(alpha=alphas)
        y = ridge.fit(X_train, Y_train).predict(X_test)
        return metrics.r2_score(Y_test, y)