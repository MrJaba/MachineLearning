import sklearn.neighbors as neighbours
import numpy as np
from numpy import savetxt

def main():
    knn = neighbours.KNeighborsClassifier()
    dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    train = np.array([x[1:] for x in dataset])
    target = np.array([x[0] for x in dataset])

    knn.fit(train, target)
    testset = np.genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]
    predicted_labels = knn.predict(testset)

    savetxt('Data/submission.csv', predicted_labels, delimiter=',', fmt='%f')


if __name__=="__main__":
    main()
