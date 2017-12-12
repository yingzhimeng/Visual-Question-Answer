#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from random import shuffle
import numpy as np
import operator
from collections import defaultdict
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, generic_utils
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Merge
from keras.layers.recurrent import LSTM
from keras.preprocessing import image as img
from sklearn import preprocessing
from sklearn.externals import joblib
from spacy import load

questions_train = open('questions_train2014.txt', 'r').read().splitlines()
answers_train = open('answers_train2014_modal.txt', 'r').read().splitlines()
images_train = open('images_train2014.txt', 'r').read().splitlines()

# get 1000 most frequent words
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

labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(answers_train)
nb_classes = len(list(labelencoder.classes_))
joblib.dump(labelencoder,'labelencoder.pkl')

def get_answers_matrix(answers, encoder):
    # string转化成数字化表达
    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    # 并构造成标准的matrix
    return Y


def get_images_matrix(img_coco_ids):
    nb_samples = len(img_coco_ids)
    nb_dimensions = (img_rows, img_cols, img_channel)
    image_matrix = np.zeros((nb_samples, img_rows, img_cols, img_channel))
    for j in range(len(img_coco_ids)):
        img_path = '../vqa_data/train2014/COCO_val2014_'+ str(img_coco_ids[j]).zfill(12) + '.jpg'
        img_file = img.load_img(img_path, target_size=(224, 224))
        image_matrix[j, :] = img.img_to_array(img_file)

    image_matrix = preprocess_input(image_matrix)
    return image_matrix

nlp = load('en')
img_dim = 4096
word_vec_dim = 384

def get_questions_tensor_timeseries(questions, nlp, timesteps):
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0].decode('utf-8'))[0].vector.shape[0]
    questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in xrange(len(questions)):
        tokens = nlp(questions[i].decode('utf-8'))
        for j in xrange(len(tokens)):
            if j<timesteps:
                questions_tensor[i,j,:] = tokens[j].vector

    return questions_tensor


max_len = 30
word_vec_dim = 384
img_dim = 4096
dropout = 0.5
activation_mlp = 'tanh'
num_epochs = 100
model_save_interval = 5

num_hidden_units_mlp = 1024
num_hidden_units_lstm = 512
num_hidden_layers_mlp = 3
num_hidden_layers_lstm = 1
batch_size = 32

img_rows, img_cols, img_channel = 224, 224, 3

image_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
flatten_image_model = Sequential()
flatten_image_model.add(image_model)
flatten_image_model.add(Flatten())
flatten_image_model.add(Dense(4096, activation='relu'))
flatten_image_model.add(Dropout(dropout))
flatten_image_model.add(Dense(4096, activation='relu'))
flatten_image_model.add(Dropout(dropout))

language_model = Sequential()
if num_hidden_layers_lstm == 1:
    language_model.add(
        LSTM(output_dim=num_hidden_units_lstm, return_sequences=False, input_shape=(max_len, word_vec_dim)))
else:
    language_model.add(
        LSTM(output_dim=num_hidden_units_lstm, return_sequences=True, input_shape=(max_len, word_vec_dim)))
    for i in xrange(num_hidden_layers_lstm - 2):
        language_model.add(LSTM(output_dim=num_hidden_units_lstm, return_sequences=True))
    language_model.add(LSTM(output_dim=num_hidden_units_lstm, return_sequences=False))

final_mlp = Sequential()
final_mlp.add(Merge([language_model, flatten_image_model], mode='concat', concat_axis=1))
for i in xrange(num_hidden_layers_mlp):
    final_mlp.add(Dense(num_hidden_units_mlp, init='uniform'))
    final_mlp.add(Activation(activation_mlp))
    final_mlp.add(Dropout(dropout))
final_mlp.add(Dense(nb_classes))
final_mlp.add(Activation('softmax'))

json_string = final_mlp.to_json()
model_file_name = '../data/full_implement'
open(model_file_name + '.json', 'w').write(json_string)

final_mlp.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print 'Compilation done'

from keras.utils.vis_utils import plot_model
plot_model(final_mlp, to_file='../data/model_full_implement.png', show_shapes=True)
from IPython.display import Image
Image(filename='../data/model_full_implement.png')

from itertools import izip_longest
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

for k in range(num_epochs):
    index_shuf = [i for i in range(len(questions_train))]
    shuffle(index_shuf)
    questions_train = [questions_train[i] for i in index_shuf]
    answers_train = [answers_train[i] for i in index_shuf]
    images_train = [images_train[i] for i in index_shuf]
    progbar = generic_utils.Progbar(len(questions_train))
    for qu_batch,an_batch,im_batch in zip(grouper(questions_train, batch_size, fillvalue=questions_train[-1]),
                                        grouper(answers_train, batch_size, fillvalue=answers_train[-1]),
                                        grouper(images_train, batch_size, fillvalue=images_train[-1])):
        X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, max_len)
        X_i_batch = get_images_matrix(im_batch, img_rows, img_cols, img_channel)
        Y_batch = get_answers_matrix(an_batch, labelencoder)
        loss = final_mlp.train_on_batch([X_q_batch, X_i_batch], Y_batch)
        progbar.add(batch_size, values=[("train loss", loss)])
    if k%model_save_interval == 0:
        final_mlp.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))
final_mlp.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))
