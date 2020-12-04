from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

######################################################################################################
    # [note]
    # e.g. input X of 2 columns: 1 feature 2 timesteps or 2 features, 1 timestep? 
    #      1 feature 2 time steps
    # data = data.reshape((data.shape[0], data.shape[1], 1))
    #      2 features 1 time step
    # data = data.reshape((data.shape[0], 1, data.shape[1]))

    # model: 2 timesteps, 1 feature for a univarite sequence with 2 lag observations per row 
    #   X<t> X<t-1> 
    #    2    6  
    #    6    5
    # model = Sequential()
######################################################################################################



# generate a sequence of random numbers in [0, n_features)
def generate_sequence(length, n_features):
    # generate (random) integer array
	return [randint(0, n_features-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# generate one example for an lstm
def generate_example(length, n_features, out_index, verbose=True):
    # generate sequence
    sequence = generate_sequence(length, n_features)
    print('sequence>\n%s\n' % sequence)

    # one hot encode
    encoded = one_hot_encode(sequence, n_features)
    print('> encoded: %s' % encoded)

    # reshape sequence to be 3D
    X = encoded.reshape((1, length, n_features))  # samples, time steps, features
    # select output
    y = encoded[out_index].reshape(1, n_features)
    return X, y

def t_tset(**kargs):

    # 1. generate integer sequence 
    # 2. one-hot encode 
    # 3. reshape the one hot encoded sequences into a format that can be used as input to the LSTM.
    # --- 
    # e.g. X, y = generate_example(3, 10, 2)
    # sequence> [2, 6, 5]
    # encoded> [[0 0 1 0 0 0 0 0 0 0]
    #           [0 0 0 0 0 0 1 0 0 0]
    #           [0 0 0 0 0 1 0 0 0 0]]
    X, y = generate_example(3, 10, 2)   # n_timesteps: 3, n_features: 10



    # [note]
    # univariate sequence with two lag values per row
    # model.add(LSTM(5, input_shape=(2,1)))   
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))  # default activation function: linear for regression


    return

def t_model(**kargs): 
    """

    Memo
    ----
    1. input must be 3-D: samples, comprised of samples, time steps and features in that order. 
    """
    # define model
    length = 5  
    n_features = 10
    out_index = 2

    # method 1
    model = Sequential()
    model.add(LSTM(25, input_shape=(length, n_features)))   # input_shape: (n_timesteps, n_features)
    model.add(Dense(n_features, activation='softmax'))   # output as a multiclass predictor

    # method 2: alternatively 
    # layers = [LSTM(2), Dense(1)]
    # model = Sequential(layers)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    # fit model
    for i in range(10000):
	    X, y = generate_example(length, n_features, out_index)  # training data ... [1]
	    model.fit(X, y, epochs=1, verbose=2)

    # evaluate model
    correct = 0
    for i in range(100):
	    X, y = generate_example(length, n_features, out_index)
	    yhat = model.predict(X)
	    if one_hot_decode(yhat) == one_hot_decode(y):
		    correct += 1
    print('Accuracy: %f' % ((correct/100)*100.0))

    # prediction on new data
    X, y = generate_example(length, n_features, out_index)
    yhat = model.predict(X)
    print('Sequence:  %s' % [one_hot_decode(x) for x in X])
    print('Expected:  %s' % one_hot_decode(y))
    print('Predicted: %s' % one_hot_decode(yhat))

    return 

def test(**kargs):

    ### generate data 
    t_tset()

    ### demo 
    # t_model()

    return 

if __name__ == "__main__": 
    test() 