#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
# import sys
from random import shuffle
# import argparse
#
import numpy as np
import scipy.io
#
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
#
from sklearn import preprocessing
from sklearn.externals import joblib
#
from spacy import load

questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().splitlines()
answers_train = open('../data/preprocessed/answers_train2014_modal.txt', 'r').read().splitlines()
images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().splitlines()

import operator
from collections import defaultdict

from keras.models import model_from_json
import os
from IPython.display import Image

# 在新的环境下：

# 载入NLP的模型
#nlp = English()
nlp = load('en')

# 以及label的encoder
labelencoder = joblib.load('../data/labelencoder.pkl')

# 接着，把模型读进去
model = model_from_json(open(
    '../data/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1.json').read())
model.load_weights(
    '../data/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1_epoch_000.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#nlp = English()
nlp = load('en')
labelencoder = joblib.load('../data/labelencoder.pkl')

flag = True

# 所有需要外部导入的料
caffe = '/home/ubuntu/Downloads/caffe'
vggmodel = '../data/VGG_ILSVRC_19_layers.caffemodel'
prototxt = '../data/VGG-Copy1.prototxt'
img_path = '../data/test_img.png'
image_features = '../data/test_img_vgg_feats.mat'

while flag:
    # 首先，给出你要提问的图片
    img_path = str(raw_input('Enter path to image : '))
    # 对于这个图片，我们用caffe跑一遍VGG CNN，并得到4096维的图片特征
    os.system(
        'python /home/gkangning/Final/Notebook/extract_features.py --caffe ' + caffe + ' --model_def ' + prototxt + ' --model ' + vggmodel + ' --image ' + img_path + ' --features_save_to ' + image_features)
    print 'Loading VGGfeats'
    # 把这个图片特征读入
    features_struct = scipy.io.loadmat(image_features)
    VGGfeatures = features_struct['feats']
    print "Loaded"
    # 然后，你开始问他问题
    question = unicode(raw_input("Ask a question: "))
    if question == "quit":
        flag = False
    timesteps = max_len
    X_q = get_questions_tensor_timeseries([question], nlp, timesteps)
    X_i = np.reshape(VGGfeatures, (1, 4096))
    # 构造成input形状
    X = [X_q, X_i]
    # 给出prediction
    y_predict = model.predict_classes(X, verbose=0)
    print labelencoder.inverse_transform(y_predict)
