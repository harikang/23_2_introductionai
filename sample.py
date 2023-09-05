#write a single python file to perform the following tasks
#(a)load data and split it to train and test.
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
digits = load_digits()
X=digits.data
X_with_bias = np.zeros((len(X),65))
X_with_bias[:, :-1] = X
y=digits.target
X_train, X_test, y_train, y_test = train_test_split(X_with_bias, y, test_size=0.1, random_state=42)

#(b) generate the target output using one hot encoding 
from tensorflow.keras.utils import to_categorical
Ytr = to_categorical(y_train)
Yts = to_categorical(y_test)

#(c) LR
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
def coeffi(N,ts):
    X_train, X_test, y_train, y_test = train_test_split(X_with_bias, y, test_size=ts, random_state=N)
    Ytr = to_categorical(y_train)
    Yts = to_categorical(y_test)
    reg_lambda = 0.0001  # regularization factor λ=0.0001
    LinearRegression = Ridge(alpha=reg_lambda)
    LinearRegression.fit(X_train, Ytr)

    coefficient = LinearRegression.score(X_train, Ytr)

    y_pred_ts = LinearRegression.predict(X_test)
    y_pred_classes_ts = np.argmax(y_pred_ts, axis=1)
    Yts_pred = to_categorical(y_pred_classes_ts)
    accuracy_ts = accuracy_score(Yts, Yts_pred)

    y_pred_tr = LinearRegression.predict(X_test)
    y_pred_classes_tr = np.argmax(y_pred_tr, axis=1)
    Yts_pred = to_categorical(y_pred_classes_tr)
    accuracy_tr = accuracy_score(Yts, Yts_pred)

    train_err = int((1-accuracy_tr)*(len(X_train)))
    test_err = int((1-accuracy_ts)*(len(X_test)))
    return coefficient,train_err,test_err

def HW1_StudentNumber(N):
    digits = load_digits()
    X=digits.data
    y=digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=N)
    Ytr = to_categorical(y_train)
    Yts = to_categorical(y_test)
    w_list,error_train_array,error_test_array=[],[],[]
    for i in [0.1,0.2,0.3,0.4,0.5]:
        a,b,c = coeffi(N,i)
        w_list.append(a)
        error_train_array.append(b)
        error_test_array.append(c)
    return X_train.shape, X_test.shape, y_train.shape, y_test.shape, Ytr.shape, Yts.shape, w_list, error_train_array, error_test_array

N = 10
X_train, X_test, y_train, y_test, Ytr, Yts, w_list, error_train_array, error_test_array = HW1_StudentNumber(N)
