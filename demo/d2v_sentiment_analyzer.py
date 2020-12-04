from gensim.models.word2vec import Word2Vec


model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)

[(u'queen', 0.711819589138031),
 (u'monarch', 0.618967592716217),
 (u'princess', 0.5902432799339294),
 (u'crown_prince', 0.5499461889266968),
 (u'prince', 0.5377323031425476)]