from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
import tensorflow.examples.tutorials.mnist.input_data as input_data
mlp = MLPClassifier(hidden_layer_sizes=(), random_state=111,
                    activation='logistic', verbose=0, max_iter=200)
def train(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X_test, y_test):
    preds = clf.predict(X_test)
    accuracy = np.mean(preds == y_test)
    return accuracy


if __name__ == '__main__':
    iris = datasets.load_iris()
    #print(iris.keys())
    #X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=1)
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)
    X_train, y_train, X_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    start = time()
    clf = train(mlp, X_train, y_train)
    end = time()
    print('Training Time:', round(end-start, 4))
    accuracy = evaluate(clf, X_train, y_train)
    print('Training Score:', round(accuracy, 4))
    accuracy = evaluate(clf, X_test, y_test)
    print('Testing Score', round(accuracy, 4))