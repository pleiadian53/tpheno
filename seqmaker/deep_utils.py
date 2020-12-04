# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

def t_sentiment_classification(): 
    # CNN for the IMDB problem
    from keras.datasets import imdb
    # from keras.models import Sequential
    # from keras.layers import Dense
    # from keras.layers import Flatten
    from keras.layers.convolutional import Convolution1D
    from keras.layers.convolutional import MaxPooling1D
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence

    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

    # pad dataset to a maximum review length in words
    max_words = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)

    # create the model
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
  
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    return

def plot_word_usage(X=None, y=None): 
    # Load and Plot the IMDB dataset
    from keras.datasets import imdb
    from matplotlib import pyplot

    # X: documents
    if X is None or y is None: 
        # load the dataset
        (X_train, y_train), (X_test, y_test) = imdb.load_data()
        X = numpy.concatenate((X_train, X_test), axis=0)
        y = numpy.concatenate((y_train, y_test), axis=0)

    # summarize size
    print("Training data: ")
    print(X.shape)
    print(y.shape)

    # Summarize number of classes 
    print("Classes: ")
    print(numpy.unique(y))

    # Summarize number of words
    print("Number of words: ")
    print(len(numpy.unique(numpy.hstack(X))))

    # Summarize review length
    print("Review length: ")
    result = map(len, X)
    print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))

    # plot review length as a boxplot and histogram
    pyplot.subplot(121)
    pyplot.boxplot(result)
    pyplot.subplot(122)
    pyplot.hist(result)
    pyplot.show()

    return

def load_data_mnist():     
    """

    Memo
    ----
    1. deep_learn/ch19
    """
    from keras import backend as K

    K.set_image_dim_ordering('th')
    
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return (X_train, y_train, X_test, y_test)

def t_larger_cnn(num_classes):
	# create model
	model = Sequential()

    # 30 feature detectors, 5-by-5 filter | input: 1 channel 28 by 28
	model.add(Convolution2D(30, 5, 5, input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def build(): 
    
    X_train, y_train, X_test, y_test = load_data_mnist() # load data
    n_classes = y_test.shape[1]  # this only makes sense because np_utils.to_categorical(y_test)

    # build the model
    model = t_larger_cnn(num_classes=n_classes)

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    
    print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

    return

def test(**kargs): 

    # CNN example #1 
    build()

    return

if __name__ == "__main__": 
    test()