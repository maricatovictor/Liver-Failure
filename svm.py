from sklearn.svm import SVR

def svm(x_train, y_train):
    clf = SVR(kernel='linear', gamma=1e-9)
    clf.fit(x_train, y_train)
    return clf