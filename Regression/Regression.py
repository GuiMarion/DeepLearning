
import urllib3
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import timeit
from keras.models import model_from_json
from keras import backend as K
import random


def load_data1():
    dataframe = pandas.read_csv("housing.data.txt", delim_whitespace=True, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]

    return X, Y


def f2(x):
    return x +1

def Sum(L):
    N = 0
    for elem in L:
        N += elem
    return N

def load_data():
    X = np.random.uniform(1, 10000, [5000, 13])
    Y = []

    for x in X:
        Y.append(Sum(x))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def load_data2():
    X = []
    Y = []

    for x in range(1000):
        X.append(x)
        Y.append(f2(x))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def load_data3():
    X = np.random.uniform(1, 1000, [5000, 2])
    Y = []

    for x in X:
        Y.append(x[0] + x[1])

    X = np.array(X)
    Y = np.array(Y)

    for i in range(len(X)):
    	print(X[i], Y[i])

    return X, Y

def load_data4():
    X = []
    Y = []

    # for i in range(1000):
    #     for e in range(100):
    #         X.append([e,i])

    # for i in range(1000):
    #     for e in range(10):
    #         X.append([i,e])

    # for i in range(1000):
    #     X.append([np.random.randint(1,1000),i])


    # for i in range(1000):
    #     X.append([np.random.randint(1,1000),i])


    # for i in range(1000):
    #     X.append([np.random.randint(1,1000),i])

    # for x in X:
    #     Y.append(x[0] * x[1])

    X_test = []
    Y_test = []

    for x1 in range(1,100):
    	for x2 in range(1, 100):
    		if random.randint(1,3) == 3:
    			X_test.append([x1,x2])
    			Y_test.append(x1*x2)
    		else:	
    			X.append([x1,x2])
    			Y.append(x1*x2)

    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)

    X_test = np.array(X).astype(float)
    Y_test = np.array(Y).astype(float)

    return (X, Y), (X_test, Y_test)


def basic_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def deeper_model():
    # create model
    model = Sequential()
    model.add(Dense(13, kernel_initializer='normal', activation='relu', input_dim=13))
    model.add(Dense(6,  kernel_initializer='normal', activation='relu'))
    model.add(Dense(1,  kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train(X, Y, fn, standardize=True, seed=7):
    estimators = []
    if standardize:
        estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=fn, epochs=30, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print(estimators)
    print('Result: %.2f (%.2f) MSE' % (results.mean(), results.std()))


def model1(X,Y):
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x=X, y=Y, batch_size=5, epochs=100, verbose=1)

    return model

def model2(X,Y):
    np.random.seed(7)
 
    # create model
    model = Sequential()
    model.add(Dense(1, input_dim=1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x=X, y=Y, batch_size=30   , epochs=1000, verbose=1)

    return model

def model3(X,Y): 
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x=X, y=Y, batch_size=5   , epochs=100, verbose=1)

    return model

def myloss(y_true, y_pred):

	return K.mean(K.square((y_pred-y_true)/y_true))




def model4(X,Y): 
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=2, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16,  activation='relu'))
    model.add(Dense(1), )
    # compile model
    sgd = optimizers.Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=sgd)

    model.fit(x=X, y=Y, batch_size=64   , epochs=400, verbose=1,
    	callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=0.0001, verbose=1)]
    )

    return model

def Somme():
    pred = []
    N = 784
    for i in range(N,N+13):
        pred.append(i)
    pred2 = pred
    pred = np.array([pred])
    if not os.path.isfile("Somme.h5"):
        X, Y = load_data()
        model = model1(X,Y)
        print(pred,":",model.predict(pred, verbose=0))
        print("True: ", Sum(pred2))
        # serialize model to JSON
        model_json = model.to_json()
        with open("Somme.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("Somme.h5")
        print("Model saved to disk")
    else: 
        # load json and create model
        json_file = open('Somme.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("Somme.h5")
        print("Loaded model from disk")
        start_time = timeit.default_timer()
        print("Prediction:",round(float(loaded_model.predict(pred, verbose=0)[0]))-1)
        print("Time execution",timeit.default_timer() - start_time,"s")
        start_time = timeit.default_timer()
        print("True: ", Sum(pred2))
        print("VS : Time execution",timeit.default_timer() - start_time,"s")

def Succ():
    X, Y = load_data2()
    model = model2(X,Y)
    pred = []
    pred.append(1500)
    pred = np.array(pred)

    print("1500:",model.predict(pred, verbose=0))
    print("True: 1501")

def Add():

    pred = []
    pred.append(996)
    pred.append(582)
    pred2 = np.array([pred])


    if not os.path.isfile("Add.h5"):
        X, Y = load_data3()
        model = model3(X,Y)
        # serialize model to JSON
        model_json = model.to_json()
        with open("Add.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("Add.h5")
        print("Model saved to disk")
    else: 
        # load json and create model
        json_file = open('Add.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("Add.h5")
        print("Loaded model from disk")
        start_time = timeit.default_timer()
        print("Prediction:",round(float(loaded_model.predict(pred2, verbose=0)[0])))
        print("Time execution",timeit.default_timer() - start_time,"s")
        start_time = timeit.default_timer()
        print("True:",pred[0] +pred[1] )
        print("VS : Time execution",timeit.default_timer() - start_time,"s")

def Mult():
    pred2 = np.random.randint(size=(1,2),low=1,high=100).astype(float)
    pred = pred2[0]

    if not os.path.isfile("Mult.h5"):
        (X, Y), (X_test,Y_test) = load_data4()
        model = model4(X,Y)
        # serialize model to JSON
        model_json = model.to_json()
        with open("Mult.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("Mult.h5")
        print("Model saved to disk")

        for i in range(10):
        	print(round(float(model.predict((np.array([X_test[i]])), verbose=0)[0])), Y_test[i])
        
        scores = model.evaluate(X_test, Y_test)
        print()
        print("Score :", float(scores)*100)


        #print("Prediction:",round(float(model.predict(X_test, verbose=0)[0])))
        #print("True:",Y_test )
    else: 
        # load json and create model
        json_file = open('Mult.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("Mult.h5")
        print("Loaded model from disk")
        start_time = timeit.default_timer()
        print("Prediction:",round(float(loaded_model.predict(pred2, verbose=0)[0])))
        print("Time execution",timeit.default_timer() - start_time,"s")
        start_time = timeit.default_timer()
        print("True:",pred[0] *pred[1] )
        print("VS : Time execution",timeit.default_timer() - start_time,"s")


if __name__ == '__main__':

    #Somme()
    #Add()
    Mult()
    #train(X, Y, fn=basic_model, standardize=False, seed=7)
    #train(X, Y, fn=basic_model, standardize=True,  seed=7)
    #train(X, Y, fn=deeper_model, standardize=True,  seed=7)
    #print(X[3],":",basic_model().predict(X[3:4]))
    #print("True: ", Y[3])
    #train(X, Y, fn=wider_model,  standardize=True,  seed=7)




