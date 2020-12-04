# ==============================================================================
#  
# Objective: Test Tensorflow implementation of word2vec models (e.g. skip-gram)
# 
# Reference: The TensorFlow Authors. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from batchpheno.utils import div  # assuming that PROJECT_DIR.../tpheno has been added to PYTHONPATH
import seqAnalyzer 


# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# def maybe_download(filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""
#   if not os.path.exists(filename):
#     filename, _ = urllib.request.urlretrieve(url + filename, filename)
#   statinfo = os.stat(filename)
#   if statinfo.st_size == expected_bytes:
#     print('Found and verified', filename)
#   else:
#     print(statinfo.st_size)
#     raise Exception(
#         'Failed to verify ' + filename + '. Can you get to it with a browser?')
#   return filename

# filename = maybe_download('text8.zip', 31344016)


# # Read the data into a list of strings.
# def read_data(filename):
#   """Extract the first file enclosed in a zip file as a list of words"""
#   with zipfile.ZipFile(filename) as f:
#     data = tf.compat.as_str(f.read(f.namelist()[0])).split()
#   return data

import seqAnalyzer


# words = read_data(filename)
words = seqAnalyzer.tokenize(load_=True)

print('Data size', len(words))
div(message='Example words:\n%s\n' % words[:100], adaptive=False)
# [log]
# Data size 48820817
# Example words: ['72702', '58980010817', '55513053010', '62934', '68115015090', '55289062730', '62439', ...]

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 219917 # setting to a smaller value results in some of infrequent words to be reduced to UNK tokens
# [params] for adpating to seqAnalyzer: 
#        Number of sentences: 15755982, unique tokens: 219917


def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  # div(message='+ count:\n%s\n' % count[:20], adaptive=False)
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)   # size is the index
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word] 
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)  
del words  # Hint to reduce memory.

print('Most common words (+UNK)', count[:5])  # [log] [['UNK', 208930], ('401', 1170415), ('250', 990197), ('V72', 829926), ('V22', 803985)]
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])  # data: word indecs | reverse dict: index -> word
# [log]
# Most common words (+UNK) [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)]
# Sample data [103, 1423, 14168, 23, 1596, 2036, 9, 1177, 1343, 1993] ... 
#             ['72702', '58980010817', '55513053010', '62934', '68115015090', '55289062730', '62439', '68180051503', '58016021360', '68387036530']

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=5)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 5       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


  # [params] loss function
  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for sid, step in enumerate(xrange(num_steps)):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    if sid < 10: 
      print('info> feed_dict:\n%s\n' % feed_dict)

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  """

  Diagnosis
  ---------
  * UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
  """
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  print('debug> final embeddings:\n%s\n' % final_embeddings[:plot_only, :])  # ValueError: array must not contain infs or NaNs
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

def t_textproc(**kargs): 
    """

    Memo
    ----
    nb_words: Maximum number of words to work with (if set, tokenization will be restricted to the top nb_words most common words in the dataset
    """

    import keras 


    # [params]
    token_sep = ','

    text = """
72702
58980010817,55513053010
62934,68115015090,55289062730,62439
68180051503,58016021360,68387036530,2036,00169008283,467
68180051403,58016086490
106480
106393,106392,106394
62525,102417,66042,61190,61326,63509,81149,72900,71169,62973,62714,62643,61171,60802,61509,61631
00781580792,68180051403,72900
86671,102417,69155,61618
    """
    texts = text.split('\n')

    print('input> got %d texts' % len(texts))
    # keras.preprocessing.text.one_hot(text, n,
    #      filters=base_filter(), lower=True, split=" ")
    tokenizer = keras.preprocessing.text.Tokenizer(nb_words=None, filters=base_filter(), 
             lower=True, split=token_sep)
    sequences = tokenizer.texts_to_sequences(texts)
    print('output> sequences:\n%s\n' % sequences)
    

    return

def test(**kargs): 
    # t_textproc(**kargs) 

    return 

if __name__ == "__main__": 
   test()