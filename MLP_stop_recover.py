from __future__ import unicode_literals
import argparse
from keras.models import model_from_json
from random import shuffle
import numpy as np
import scipy.io
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from sklearn import preprocessing
from sklearn.externals import joblib
from spacy import load

questions = open('/Users/jiafangliu/Documents/class/ML/Final/preprocessed/questions_train2014.txt', 'r').read().splitlines()
answers = open('/Users/jiafangliu/Documents/class/ML/Final/preprocessed/answers_train2014_modal.txt', 'r').read().splitlines()
images = open('/Users/jiafangliu/Documents/class/ML/Final/preprocessed/images_train2014.txt', 'r').read().splitlines()

import operator
from collections import defaultdict

max_answers = 1000
answer_fq= defaultdict(int)
for answer in answers:
    answer_fq[answer] += 1

sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:max_answers]
top_answers, top_fq = zip(*sorted_fq)
new_answers=[]
new_questions=[]
new_images=[]

for answer,question,image in zip(answers, questions, images):
    if answer in top_answers:
        new_answers.append(answer)
        new_questions.append(question)
        new_images.append(image)

questions = new_questions
answers = new_answers
images = new_images


def get_answers_matrix(answers, encoder):

    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)

    return Y

vgg_model_path = '/Users/jiafangliu/Documents/class/ML/Final/coco/vgg_feats.mat'

features_struct = scipy.io.loadmat(vgg_model_path)
VGGfeatures = features_struct['feats']

image_ids = open('/Users/jiafangliu/Documents/class/ML/Final/data/coco_vgg_IDMap.txt').read().splitlines()
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


nlp = load('en', vectors='en_glove_cc_300_1m_vectors')

img_dim = 4096

word_vec_dim = 300


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

num_hidden_units = 1024
num_hidden_layers = 3
dropout = 0.5
activation = 'tanh'

num_epochs = 1
model_save_interval = 1
batch_size = 128

# model = model_from_json(open('/Users/jiafangliu/Documents/class/ML/Final/test/num_hidden_layers_mlp_3.json').read())
# model.load_weights('/Users/jiafangliu/Documents/class/ML/Final/test/num_hidden_layers_mlp_3_epoch_070.hdf5')
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str,
                    default='/Users/jiafangliu/Documents/class/ML/Final/test/num_hidden_layers_mlp_3.json')
parser.add_argument('-weights', type=str,
                    default='/Users/jiafangliu/Documents/class/ML/Final/test/num_hidden_layers_mlp_3_epoch_070.hdf5')
parser.add_argument('-sample_size', type=int, default=25)
args = parser.parse_args()

model = model_from_json(open(args.model).read())
model.load_weights(args.weights)
labelencoder = joblib.load('/Users/jiafangliu/Documents/class/ML/Final/labelencoder.pkl')

from itertools import izip_longest
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

for k in range(2):

    index_shuf = [i for i in range(len(questions))]
    shuffle(index_shuf)

    questions = [questions[i] for i in index_shuf]
    answers = [answers[i] for i in index_shuf]
    images = [images[i] for i in index_shuf]

    progbar = generic_utils.Progbar(len(questions))

    for qu_batch,an_batch,im_batch in zip(grouper(questions, batch_size, fillvalue=questions[-1]),
                                          grouper(answers, batch_size, fillvalue=answers[-1]),
                                          grouper(images, batch_size, fillvalue=images[-1])):
        X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
        X_i_batch = get_images_matrix(im_batch, id_map, VGGfeatures)
        X_batch = np.hstack((X_q_batch, X_i_batch))
        Y_batch = get_answers_matrix(an_batch, labelencoder)
        loss = model.train_on_batch(X_batch, Y_batch)
        progbar.add(batch_size, values=[("train loss", loss)])

    if k%model_save_interval == 0:
        model.save_weights('/Users/jiafangliu/Documents/class/ML/Final/result/' + '_epoch_{:02d}.hdf5'.format(k))

model.save_weights('/Users/jiafangliu/Documents/class/ML/Final/result/'+ '_epoch_{:02d}.hdf5'.format(k))






