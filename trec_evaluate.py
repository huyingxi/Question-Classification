import numpy as np
import sys
import os
import codecs
import tensorflow as tf
import keras as keras
import h5py
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential,Model,load_model
from keras.layers import Activation,Dense,Dropout,Embedding,Flatten,Input,merge,Convolution2D,MaxPooling2D,Convolution1D,MaxPooling1D,merge,Reshape
from keras.optimizers import SGD,sgd
from keras.callbacks import EarlyStopping ,ModelCheckpoint
from gensim.models import Word2Vec
import cPickle as pickle

reload(sys)
sys.setdefaultencoding('utf-8')


# Training parameters
batch_size_set = 32
num_epoch_set = 300
sequence_length = 40

text_paths=["/home/wxr/hyx/TRECdata/test",]

#Data Preproccesing
def load_data():
    #load data
    print("loading data...")
    texts = []  # list of text samples
    labels = []  # list of label ids
    count = -1
    d = {"DESC" : 0, "ENTY" : 1, "ABBR" : 2, "HUM" : 3, "LOC" : 4, "NUM" : 5}
    for path_name in text_paths:
        for name in sorted(os.listdir(path_name)):
            path = os.path.join(path_name,name)
            with codecs.open(path,'rU','utf-8-sig') as f:
                line = f.readline()          
                while line:  
                    lines = line.split(":")
                    labels.append(d[lines[0]])
                    temp_line = ""
                    for i in xrange(1,len(lines)):
                        temp_line += lines[i]
                    texts.append(temp_line)
                    line = f.readline()  
                f.close()

    
    sequences = []
    for text in texts:
        num = 0
        sequence = []
        text_split = text.split(" ")
        for word in text_split:
            if word!="" and num <= sequence_length:
                num += 1
                word = word.decode("utf-8")
                if word in word_index:
                    sequence.append(word_index[word])
                else:
                    sequence.append(0)
        sequences.append(sequence)

    #padding the sequences to the same length
    data = pad_sequences(sequences, maxlen = sequence_length)
 
    labels = to_categorical(np.asarray(labels))

    return data, labels



if __name__ == "__main__":
    
    with tf.device('/gpu:1'):
        with open('/home/wxr/wxr/TREC/corpus/random_vocab.pkl','rb') as vocab:
            word_index = pickle.load(vocab)
            embedding_matrix = pickle.load(vocab)
        X_test,  Y_test =load_data()

        model = load_model('/home/wxr/hyx/trec_model/trec_cnn_lstm_model_weights_col_size_train_405_nothighway.h5')

        print "evaluate:"

        score = model.evaluate(X_test, Y_test)
        print('test loss:', score[0])
        print('test accuracy:', score[1])

