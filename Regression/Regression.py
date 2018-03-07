
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


# For the succesor function
def f2(x):
    return x +1

# Compute the sum of a list 
def Sum(L):
    N = 0
    for elem in L:
        N += elem
    return N


# Load data for the sum, I didn't use validation set for this one, I just check on some 
# random values in the last function.
def load_data():
    X = np.random.uniform(1, 10000, [5000, 13])
    Y = []

    for x in X:
        Y.append(Sum(x))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# Load data for the succesor, I didn't use validation set for this one, I just check on some 
# random values in the last function.
def load_data2():
    X = []
    Y = []

    for x in range(1000):
        X.append(x)
        Y.append(f2(x))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


# Load data for the addition, I didn't use validation set for this one, I just check on some 
# random values in the last function.
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



# Load data for the multiplication, I choose randomely 1/3 of the dataset for the validation set.
# So the training set contains the first 10 000 possibilities for multiplication, without
# the ones for the validation set. It is an easy-to-train set, if you want to make it trickier,
# you can uncomment the code above and work on a larger numeric domain. 
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


# Some exemples of models if you want (I don't use them here)
def basic_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Some exemples of models if you want (I don't use them here)
def deeper_model():
    # create model
    model = Sequential()
    model.add(Dense(13, kernel_initializer='normal', activation='relu', input_dim=13))
    model.add(Dense(6,  kernel_initializer='normal', activation='relu'))
    model.add(Dense(1,  kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Some exemples of models if you want (I don't use them here)
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Model for the Sum, it get 13 integers as input and return the sum of these 13 integers.
# It's an easy one.
def model1(X,Y):
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x=X, y=Y, batch_size=5, epochs=100, verbose=1)

    return model

# Model for the Succesor, takes one intger and return its succosor (n+1)
# suprising, it's harder-to-train then the sum. 
def model2(X,Y):
    # You can use this, it's define a seed for random number generation for the reproductability. 
    #np.random.seed(7)
 
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=1, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1,))
    # compile model

    sgd = optimizers.Adam(lr=1e-4)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    model.fit(x=X, y=Y, batch_size=30   , epochs=500, verbose=1, 
        callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=0.0000001, verbose=1)])

    return model


# Model for the Sum, it get 13 integers as input and return the sum of these 13 integers.
# It's an easy one.
def model3(X,Y): 
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x=X, y=Y, batch_size=64   , epochs=100, verbose=1)

    return model


# Loss function we definied for the multiplication, indeed the mse doesn't work well
# because a mistake in big numbers is bigger problem then in small ones. It's because
# the loss is not balanced with the size of the expected number.
def myloss(y_true, y_pred):

    return K.mean(K.square((y_pred-y_true)/y_true))



# Model for the Multiplication, N^2 -> N, returns le mult of two integers. 
# It's not a so-easy one. That's why we add more and bigger layers. 
# The accuracy is not about 100% but it works enought to be convinced that it's working.
# The callback function is for decreasing the learning rate when we get stuck in 
# a plateau. 
def model4(X,Y): 
    # create model
    model = Sequential()
    model.add(Dense(36, input_dim=2, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(26, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8,  activation='relu'))
    model.add(Dense(1), )
    # compile model
    sgd = optimizers.Adam(lr=1e-3)
    model.compile(loss=myloss, optimizer=sgd)

    model.fit(x=X, y=Y, batch_size=64   , epochs=400, verbose=1,
        callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=0.0001, verbose=1)]
    )

    return model


# Training function for the Sum
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

# Training function for the Succesor
def Succ():
    pred = (np.random.randint(2000))
    pred2 = np.array([pred])

    if not os.path.isfile("Succ.h5"):
        X, Y = load_data2()
        model = model2(X,Y)
        # serialize model to JSON
        model_json = model.to_json()
        with open("Succ.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("Succ.h5")
        print("Model saved to disk")
        print(pred,":",model.predict(pred2, verbose=0))
        print("True:", pred+1)
    else: 
        # load json and create model
        json_file = open('Succ.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("Succ.h5")
        print("Loaded model from disk")
        start_time = timeit.default_timer()
        print("Prediction:",round(float(loaded_model.predict(pred2, verbose=0)[0])))
        print("Time execution",timeit.default_timer() - start_time,"s")
        start_time = timeit.default_timer()
        print("True:",pred + 1 )
        print("VS : Time execution",timeit.default_timer() - start_time,"s")

# Training function for the Addition
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


# Training function for the Multiplication
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

# Main, uncomment the function you want. 
if __name__ == '__main__':

    #Somme()
    #Add()
    Succ()
    #Mult()


