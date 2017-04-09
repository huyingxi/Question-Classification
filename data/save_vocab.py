# -*- coding:utf-8 -*-
import numpy as np
import sys
from gensim.models import Word2Vec
import cPickle as pickle
from string import punctuation
from keras.preprocessing.text import Tokenizer
reload(sys)
sys.setdefaultencoding('utf-8')

def base_filters():
    filters = punctuation
    filters += '\t\n'
    filters += '-'
    return filters
word2vec_model = Word2Vec.load_word2vec_format('/home/wxr/blazer/goole_word2vec300/GoogleNews-vectors-negative300.bin', binary=True)
print "start"
total_file = open('total_corpus.txt','rU')
texts = total_file.readlines()
tokenizer = Tokenizer(lower=False, filters=base_filters())
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
embedding_matrix = np.random.rand(len(word_index) + 1, 300)
embedding_matrix[0,:]=np.zeros((300,))
print embedding_matrix.shape
for word, i in word_index.items():
    if word in word2vec_model.vocab:
        embedding_matrix[i] = word2vec_model[word]
        if i == 1:
            print '1y',embedding_matrix[i]
        elif i==2:
            print '2y',embedding_matrix[i]
    else:
        if i == 1:
            print '1n',embedding_matrix[i]
        elif i==2:
            print '2n',embedding_matrix[i]
print embedding_matrix[0]
with open('random_vocab.pkl', 'wb') as f:
    pickle.dump(word_index,f,2)
    pickle.dump(embedding_matrix,f,2)
print 'save'
         

