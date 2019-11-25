import pandas
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def make():

    x, y = make_classification(n_samples=1250, n_features=10,  n_informative=5)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    test = pandas.DataFrame(x_test)
    test['y'] = y_test
    test.to_csv('test.csv', index=False)
    sk = RandomForestClassifier()
    sk.fit(x_train, y_train)
    joblib.dump(sk, 'mdl.joblib')


if __name__ == '__main__':
    make()
