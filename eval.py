from sklearn import cross_validation
import sklearn.neighbors as neighbours
import numpy as np

def evaluation(actual, predicted):
    actual == predicted

def main():
    knn = neighbours.KNeighborsClassifier(algorithm:"kd_tree", weights:"distance")
    dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    train = np.array([x[1:] for x in dataset])
    target = np.array([x[0] for x in dataset])
    cv = cross_validation.KFold(len(train), k=5, indices=False)

    results = []
    for traincv, testcv in cv:
        predictions = knn.fit(train[traincv], target[traincv]).predict(train[testcv])
        results.append( evaluation(target[testcv], [x for x in predictions]) )
    
    correct_results = filter( (lambda x: x == True), results)
    print "Results: " + str(len(correct_results)/(float(len(results))))


if __name__=="__main__":
    main()

