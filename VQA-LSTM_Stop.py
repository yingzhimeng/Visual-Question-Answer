#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from random import shuffle
import numpy as np
import scipy.io
#
import keras.backend as K

import numpy as np
from keras.utils import np_utils, generic_utils

from sklearn.externals import joblib
from spacy import load

questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().splitlines()
answers_train = open('../data/preprocessed/answers_train2014_modal.txt', 'r').read().splitlines()
images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().splitlines()

import operator
from collections import defaultdict

max_answers = 1000
answer_fq= defaultdict(int)
for answer in answers_train:
    answer_fq[answer] += 1
sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:max_answers]
top_answers, top_fq = zip(*sorted_fq)
new_answers_train=[]
new_questions_train=[]
new_images_train=[]
for answer,question,image in zip(answers_train, questions_train, images_train):
    if answer in top_answers:
        new_answers_train.append(answer)
        new_questions_train.append(question)
        new_images_train.append(image)

questions_train = new_questions_train
answers_train = new_answers_train
images_train = new_images_train


labelencoder = joblib.load('../data/labelencoder.pkl')
nb_classes = len(list(labelencoder.classes_))


def get_answers_matrix(answers, encoder):
    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    return Y

vgg_model_path = '../Downloads/coco/vgg_feats.mat'

features_struct = scipy.io.loadmat(vgg_model_path)
VGGfeatures = features_struct['feats']
image_ids = open('../data/coco_vgg_IDMap.txt').read().splitlines()
id_map = {}
for ids in image_ids:
    id_split = ids.split()
    id_map[id_split[0]] = int(id_split[1])

def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
    nb_samples = len(img_coco_ids)
    nb_dimensions = VGGfeatures.shape[0]
    image_matrix = np.zeros((nb_samples, nb_dimensions))
    for j in range(len(img_coco_ids)):
        image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]
    return image_matrix

nlp = load('en')
img_dim = 4096
word_vec_dim = 384

def get_questions_matrix_sum(questions, nlp):
    # assert not isinstance(questions, basestring)
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0].decode('utf-8'))[0].vector.shape[0]
    questions_matrix = np.zeros((nb_samples, word_vec_dim))
    for i in range(len(questions)):
        tokens = nlp(questions[i].decode('utf-8'))
        for j in range(len(tokens)):
            questions_matrix[i,:] += tokens[j].vector
    return questions_matrix

from itertools import izip_longest
def batches(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def get_questions_tensor_timeseries(questions, nlp, timesteps):
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0].decode('utf-8'))[0].vector.shape[0]
    questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in xrange(len(questions)):
        tokens = nlp(questions[i].decode('utf-8'))
        for j in xrange(len(tokens)):
            if j< timesteps:
                questions_tensor[i, j, :] = tokens[j].vector

    return questions_tensor

max_len = 30
word_vec_dim = 384
img_dim = 4096
dropout = 0.5
activation_mlp = 'tanh'
num_epochs = 100
model_save_interval = 5

num_hidunit_mlp = 1024
num_hidunit_lstm = 512
num_hidlayer_mlp = 3
num_hidlayer_lstm = 1
batch_size = 128
model_file_name = '../data/my_LSTM'

from keras.models import model_from_json
model = model_from_json(open(
    '../data/my_LSTM.json').read())
model.load_weights(
    '../data/my_LSTM_epoch_10.h5')

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy',mean_pred])
#model.evaluate(np.asarray([np.zeros((10))]), np.asarray([np.zeros((20))]))
print 'Compilation done'


from keras.utils.vis_utils import plot_model

plot_model(model, to_file='../data/model_lstm.png', show_shapes=True)
from IPython.display import Image

Image(filename='../data/model_lstm.png')


features_struct = scipy.io.loadmat(vgg_model_path)
VGGfeatures = features_struct['feats']
print 'loaded vgg features'
image_ids = open('../data/coco_vgg_IDMap.txt').read().splitlines()
img_map = {}
for ids in image_ids:
    id_split = ids.split()
    img_map[id_split[0]] = int(id_split[1])

nlp = load('en')

print 'loaded word2vec features...'
## training
print 'Training started...'
for k in xrange(num_epochs):

    progbar = generic_utils.Progbar(len(questions_train))

    for qu_batch, an_batch, im_batch in zip(batches(questions_train, batch_size, fillvalue=questions_train[-1]),
                                            batches(answers_train, batch_size, fillvalue=answers_train[-1]),
                                            batches(images_train, batch_size, fillvalue=images_train[-1])):
        X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, max_len)
        X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
        Y_batch = get_answers_matrix(an_batch, labelencoder)
        loss = model.train_on_batch([X_q_batch, X_i_batch], Y_batch)
        #progbar.add(batch_size, values=[("train loss", loss)])
        progbar.add(batch_size, values=[("train loss", loss)])
    if k % model_save_interval == 0:
        model.save_weights(model_file_name + '_epoch_' + str(k) + '.h5')

model.save_weights(model_file_name + '_epoch_' + str(k) + '.h5')
