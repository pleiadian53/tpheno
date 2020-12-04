import gensim
import os
import collections
import smart_open
import random


### Memo
#
# 1. Warning: 
#     anaconda/lib/python2.7/site-packages/gensim/utils.py:1015: UserWarning: Pattern library is not installed, 
#     lemmatization won't be available. warnings.warn("Pattern library is not installed, lemmatization won't be available.")

# Set file names for train and test data
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
# print('> test_data_dir: %s' % test_data_dir)  # .../anaconda/lib/python2.7/site-packages/gensim/test/test_data
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

# print('> train file: %s\n> test file:%s\n' % (lee_train_file, lee_test_file))


def read_corpus(fname, tokens_only=False):
	"""

	Memo
	----
	1. For a given file (aka corpus), each continuous line constitutes a single document and the length of each line (i.e., document) can vary.
	"""
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])