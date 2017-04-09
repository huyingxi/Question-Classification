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
from keras.models import Sequential,Model
from keras.layers import Activation,Dense,Dropout,Embedding,Flatten,Input,merge,Convolution2D,MaxPooling2D,Convolution1D,MaxPooling1D,merge,Reshape
from keras.optimizers import SGD,sgd,Adadelta
from keras.callbacks import EarlyStopping ,ModelCheckpoint
from gensim.models import Word2Vec
import cPickle as pickle
from keras.layers import Input, Embedding, LSTM, Dense, merge, Bidirectional

reload(sys)
sys.setdefaultencoding('utf-8')


# Model Hyperparameters
sequence_length = 40
embedding_dim = 300
filter_sizes = (1, 2, 3, 4)
feature_map = 128
dropout_prob = (0.4, 0.4)
hidden_dims = (128,60,6)
labels = 6


# Training parameters
batch_size_set = 32
num_epoch_set = 200
val_split = 0.1


text_paths=["/home/wxr/hyx/TRECdata/tmp",]

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
                    #print "coming"
                else:
                    #print "no"
                    sequence.append(0)
        sequences.append(sequence)

    #padding the sequences to the same length
    data = pad_sequences(sequences, maxlen = sequence_length)
    
    
    labels = to_categorical(np.asarray(labels))
    
    #print labels[0],labels[1],labels[2],labels[3],labels[5]
    
    indices = np.arange(data.shape[0])
    #print "indices_shape : ",indices
    np.random.shuffle(indices)
    #print "random indices : ",indices
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(0.2 * data.shape[0])

    x_train = data[:-1*nb_validation_samples]
    y_train = labels[:-1*nb_validation_samples]
    x_val = data[-1*nb_validation_samples:]
    y_val = labels[-1*nb_validation_samples:]
    


    return x_train, y_train, x_val, y_val






if __name__ == "__main__":
    with tf.device('/gpu:0'):
        for col_size in range(300,301):
            print "now_col_size : ",col_size
            ##############
            #load data
            ##############
            with open('/home/wxr/wxr/TREC/corpus/random_vocab.pkl','rb') as vocab:
                word_index = pickle.load(vocab)
                embedding_matrix = pickle.load(vocab)
            #charvec_model = Word2Vec.load_word2vec_format('/home/wxr/blazer/goole_word2vec300/GoogleNews-vectors-negative300.bin', binary=True)
            x_train, y_train, x_valid, y_valid = load_data()


            ##############
            #set model
            ##############    



            nb_words = len(word_index)
            print "embedding:%d" % (nb_words + 1)

            #load pre-trained word embeddings into an Embedding layer
            embedding_layer1 = Embedding( input_dim=nb_words + 1,
                               output_dim=embedding_dim,
                               weights=[embedding_matrix],
                               input_length=sequence_length,
                               trainable = False)
            embedding_layer2 = Embedding( input_dim=nb_words + 1,
                               output_dim=embedding_dim,
                               weights=[embedding_matrix],
                               input_length=sequence_length,
                               trainable = False)

            ###############
            #building model
            ###############
            #model-1
            #char_input = Input(shape=(sequence_length, embedding_dim), dtype='float32')
            word_input = Input(shape=(sequence_length,))
            model1 = embedding_layer1(word_input)
            model1 = Dropout(0.5)(model1)
            model1 = Reshape((sequence_length,embedding_dim,1))(model1)

            model2=embedding_layer2(word_input)
            model2=Dropout(0.5)(model2)
            model2=Reshape((sequence_length,embedding_dim,1))(model2)

            biLSTM_Input = Reshape((sequence_length,embedding_dim))(model1)
            left_branch = LSTM(300,input_shape = (40,300),return_sequences='True',input_length=40)(biLSTM_Input)

            right_branch = LSTM(300,input_shape=(40,300),return_sequences='True',input_length=40,go_backwards=True)(biLSTM_Input)

            print "left_branch.get_shape()",left_branch.get_shape()
            print "right_branch.get_shape()",right_branch.get_shape()
            #merge link tensor ; Merge link model layer
            lstm_merged = merge([left_branch,right_branch],mode='ave')
            lstm_merged = Reshape([40,300,1])(lstm_merged)
            lstm_merged = Dropout(0.2)(lstm_merged)
            graph_in_temp = merge([model1, model2,lstm_merged],mode='concat',concat_axis=-1)
            #graph_in_temp = lstm_merged
            #print "merge befor after : ", graph_in_temp.get_shape()

            graph_in = Reshape((40,300,3))(graph_in_temp)
            #graph_in = Reshape((40,300,1))(graph_in_temp)

            print graph_in.get_shape()

            conv_11 = Convolution2D(nb_filter=feature_map, nb_row=filter_sizes[0], nb_col=col_size, border_mode='valid', activation='relu')(graph_in)
            conv_22 = Convolution2D(nb_filter=feature_map, nb_row=filter_sizes[1], nb_col=col_size, border_mode='valid', activation='relu')(graph_in)
            conv_33 = Convolution2D(nb_filter=feature_map, nb_row=filter_sizes[2], nb_col=col_size, border_mode='valid', activation='relu')(graph_in)

            conv_11 = MaxPooling2D(pool_size=(int(conv_11.get_shape()[1]),int(conv_11.get_shape()[2])))(conv_11)
            conv_22 = MaxPooling2D(pool_size=(int(conv_22.get_shape()[1]),int(conv_22.get_shape()[2])))(conv_22)
            conv_33 = MaxPooling2D(pool_size=(int(conv_33.get_shape()[1]),int(conv_33.get_shape()[2])))(conv_33)


            conva = merge([conv_11, conv_22, conv_33], mode='concat',concat_axis=-1)
            
            #conva = merge([conv_11, conv_22], mode='concat',concat_axis=-1)
            conva = Dropout(dropout_prob[1])(conva)

            print conva.get_shape()
            #model-2
            out = Reshape((3*128,))(conva)
            out = Dense(hidden_dims[0], activation='relu', W_regularizer=l2(0.02))(out)
            out = Dropout(dropout_prob[1])(out)
            out = Dense(hidden_dims[1], activation='relu', W_regularizer=l2(0.02))(out)
            out = Dropout(dropout_prob[1])(out)
            out = Dense(hidden_dims[2], activation='softmax')(out)

            total = Model(input=word_input, output=out)
            #print total.summary()

            #sgd1=keras.optimizers.SGD(lr=0.003, decay=1e-5, momentum=0.9, nesterov=True)
            sgd1=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)

            total.compile(optimizer = sgd1, loss='categorical_crossentropy', metrics=["accuracy"])
            earlyStopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
            load_file = "/home/wxr/hyx/trec_model/trec_cnn_lstm_model_weights_col_size_train_405_nothighway.h5"
            checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,
                                       save_best_only=True)
            total.fit(x_train, y_train, validation_data=(x_valid, y_valid), nb_epoch=num_epoch_set, batch_size=batch_size_set, shuffle=True,callbacks=[checkpointer, earlyStopping])


            total.load_weights(load_file)
            #training data \  validation data \ test data  --- evaluate
            print"evaluate:"
            score = total.evaluate(x_train, y_train)
            print('Train loss:' , score[0])
            print('Train accuracy:', score[1])

            score = total.evaluate(x_valid, y_valid)
            print('validation loss:', score[0])
            print('validation accuracy:', score[1])


