# Question-Classification
my deep learning model  for solving the TREC question classification problem


#深度学习模型
用了blstm  +  cnn的网络


#data文件夹
存放了数据及数据预处理工作
train和test文件夹包括了TREC原始数据集（来源）
save_vocab.py用于对原始数据集的清洗，采用word2vec模型，过滤出本项目需要的词向量


#trec_cnn_blstm.py
用于模型的搭建和模型的训练

#trec_test.py
用户加载训练好的模型并验证模型在测试集上的效果
