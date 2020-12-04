# encoding: utf-8

# import numpy
import numpy as np
import os, sys, collections, random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# tokenization
from keras.preprocessing.text import Tokenizer

# word embedding layer
from keras.layers import Embedding

def tokenzier_demo0(D=None): 
    tokenizer = Tokenizer()
    if D is None: 
        D = ["The sun is shining in June!","September is grey.",
                 "Life is beautiful in August.", 
                 "I like it","This and other things?"]
    
    tokenizer.fit_on_texts(D)
    print(tokenizer.word_index)

    print tokenizer.text_to_word_sequence(D)

    Din = ["June is beautiful and I like it!", "x y z", "I like it"]  # "x y z" not in training examples => []
    X = tokenizer.texts_to_sequences(Din)
    

    n_tokens = len(np.unique(np.hstack(tokenizer.text_to_word_sequence(D))))
    print('> Din:\n%s\n' % Din)
    print('> X:\n%s\n' % X)

    Din = ["June is beautiful and I like it!", "Like August"]
    Xp = tokenizer.texts_to_matrix(Din)
    print('> Din:\n%s\n' % Din)
    print('> Xp (n_tokens=%d, dim(x): %d):\n%s\n' % (n_tokens, len(Xp[0]), Xp))

    return

def docToInteger(D, policy='exact', **kargs): 
    import docProc 
    return docProc.docToInteger(D, policy=policy, **kargs)

def embedding_demo0(X=[], target_dim=2, input_length=None, value_range=None): 
    """

    Memo
    ----
    1. The first value of the Embedding constructor is the range of values in the input. 
       For binary input, it’s 2. 
       The second value is the target dimension. 
       The third is the length of the vectors we give.
    """
    def verify_dim(): 
    	nX = len(X)
    	dim0 = len(X[1])
        for vec in random.sample(X, min(nX, 10)): 
            assert len(vec) == dim0, "inconsistent vector dim! dim(X[0]): %d but got dim(X[j]): %d" % (dim0, len(vec))
        return dim0  

    # example input D (2D)
    if not D: 
    	D = np.array([[0,1,0,1,1,0,0]]) # D must have been tokenized to an integer repr

    # deduce input length (assuming each vector has the same length)
    if input_length is None: 
    	# input_length = verify_dim()
        input_length = X[1]

    model = Sequential()

    if value_range is None: value_range = len(np.unique(np.hstack(X)))
    model.add(Embedding(value_range, target_dim, input_length=input_length))
    model.compile('rmsprop', 'mse')
    return model.predict(X)

def test(**kargs): 

    # tokenizer examples 
    tokenzier_demo0()
    
    return

if __name__ == "__main__":
    test() 