from collections import defaultdict, Counter
import collections
import os
import re
import string
import random
import pandas as pd
import numpy as np
import math
import csv
from statistics import mean
import spacy
import pickle

import nltk
from nltk.corpus import stopwords
from nltk import FreqDist, ngrams
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import ToktokTokenizer, word_tokenize
from nltk.lm import MLE
from nltk.stem.snowball import EnglishStemmer
from textaugment import EDA
import spelling_confusion_matrices
nltk.download(['wordnet', 'punkt', 'averaged_perceptron_tagger'])

import torch
import torch.nn as nn  # linear, cnn, ...
import torch.nn.functional as F  # relu, tanh, ...
import torch.optim as optim   # adam, sgd, ...
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torchtext  # torchtext==0.11.0
# import torchnlp
# import pytorch-nlp  # pytorch-nlp==0.5.0

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *

import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import time


"""
Welcome

Text Classification performed with linear, non-linear and neural network classifiers (sklearn & pytorch).

Data: Trump tweets.

Objective: Classify if a new tweet is written by Trump or by his public relation figures.

"""


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 16)
        self.out = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(input_size, input_size),
        #     nn.ReLU(),
        #     nn.Linear(int(input_size/2), int(input_size/2)),
        #     nn.ReLU(),
        #     nn.Linear(input_size, num_classes),
        # )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

    def forward2(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward3(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class RNN1(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size, layer_size, sequence_size, device):
        # super(LSTM, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(input_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.4)
        self.device = device

    def forward(self, x, s):
        x = self.emb(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.fc(ht[-1])
        return out


class RNN2(nn.Module):
    def __init__(self, vocab_size, hidden_size, layer_size, emb_size, device):
        super().__init__()
        self.dropout = nn.Dropout(0.4)
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = layer_size
        self.embed_size = emb_size
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.rnn = nn.LSTM(input_size=self.embed_size,
                           hidden_size=hidden_size,
                           dropout=0.4,
                           num_layers=layer_size, bidirectional=True)
        self.hidden2target = nn.Linear(2*hidden_size, 2)
        # self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropoutLayer = nn.Dropout()

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_size))
        return h0, c0

    def forward(self, inputs, input_lengths):
        self.hidden = self.init_hidden(inputs.size(-1))
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, self.hidden = self.rnn(packed, self.hidden)
        output, output_lengths = pad_packed_sequence(outputs, batch_first=False)
        output = torch.transpose(output, 0, 1)
        output = torch.transpose(output, 1, 2)
        output = torch.tanh(output)
        output, indices = F.max_pool1d(output,output.size(2), return_indices=True)
        output = torch.tanh(output)
        output = output.squeeze(2)
        output = self.dropoutLayer(output)
        output = self.hidden2target(output)
        output = self.softmax(output)
        return output, self.hidden


class RNN3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(RNN3, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class RNN4(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, device):
        super(RNN4, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.rnn = nn.RNN(input_size, hidden_size, layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.device = device
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Set initial hidden and cell states
        size = x.size(0)
        size = 1
        h0 = torch.zeros(self.layer_size, size, self.hidden_size).to(self.device)
        x = torch.unsqueeze(x, 0)
        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

    def forward2(self, x):
        h0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).to(self.device)  # sets initial hidden and cell states
        out, _ = self.rnn(x, h0)  # forward propagate LSTM
        out = out.reshape(out.shape[0], -1)  # decodes hidden state of the last time step
        out = self.fc(out)
        x = self.dropout(x)
        # x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        # out_pack, (ht, ct) = self.rnn(x_pack)
        # out = self.fc(ht[-1])
        return out


class RNN5(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, layer_size, input_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=layer_size, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(hidden_size, 32)
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.linear_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, input1, input2, s):
        x = self.emb(input1)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        lstm_out = self.linear(ht[-1])
        y = self.fc1(input2)
        y = self.relu(y)
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc3(y))
        tabu_out = self.dropout(y)
        z = torch.cat((lstm_out, tabu_out), 1)
        out = self.linear_out(z)
        return out


class RNN6(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN6, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.sigmoid(self.in2hidden(combined))
        output = self.in2output(combined)
        return output, hidden

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))


class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, num_classes, device):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.layer_size * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.layer_size * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, num_classes, sequence_size, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.gru = nn.GRU(input_size, hidden_size, layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_size, num_classes)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).to(self.device)  # sets initial hidden and cell states
        out, _ = self.gru(x, h0)  # forward propagate LSTM
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)  # decodes hidden state of the last time step
        return out


class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, emb_size, layer_size, sequence_size, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(input_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        # self.fc = nn.Linear(hidden_size * sequence_size, num_classes)    
        self.dropout = nn.Dropout(0.2)
        self.device = device

    def forward(self, x, s):
        # h0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).to(self.device)
        # out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out = out.reshape(out.shape[0], -1)
        # out = self.fc(out)  # decodes hidden state of the last time step
        x = self.emb(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.fc(ht[-1])
        return out


class Train(Dataset):

    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


class Test(Dataset):

    def __init__(self, x_test):
        self.x = x_test

    def __getitem__(self, i):
        return self.x[i]

    def __len__(self):
        return len(self.x)


class Text(Dataset):

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        y_size = len(self.y)
        return y_size

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i][0].astype(np.int32)), self.y[i], self.x[i][1]


class History(tf.keras.callbacks.Callback):

    """ Class 'History' based on Tensor for specified results """

    def __init__(self, test):
        super(History, self).__init__()
        self.test = test

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        loss, acc = self.model.evaluate(self.test)
        if 'test_loss' not in logs:
            logs['test_loss'] = []
        logs['test_loss'].append(loss)
        if 'test_accuracy' not in logs:
            logs['test_accuracy'] = []
        logs['test_accuracy'] = acc
        if 'test_f1' not in logs:
            logs['test_f1'] = []
        logs['test_f1'] = f1_score(self.y_test, self.y_pred)


class Stemmer:

    def __init__(self):
        self.stemmer = EnglishStemmer()
        self.s_corp = {}  # stemms for corpus
        # self.s_doc = {}  # stemms for a doc (hash cache)

    def _stem(self, term):
        if term in self.s_corp:  # (1) existing term
            this_count = self.s_corp[term]
            self.s_corp.update({term: this_count + 1})
            return term
        else:
            stemmed_term = self.stemmer.stem(term)
            try:  # (2) existing term and stem term exists
                this_count = self.s_corp[stemmed_term]
                self.s_corp.update({stemmed_term: this_count + 1})
            except KeyError:  # (3) new stem term
                self.s_corp[stemmed_term] = 1
            return stemmed_term

    def save_stemmed_vocabulary(self,path):
        with open(path +'/Statistics/Stemmed_Vocabulary.pkl', 'wb') as output:
            pickle.dump(self.s_corp, output, pickle.HIGHEST_PROTOCOL)


class TextDataset(Dataset):
    def __init__(self, input_size, x, y):
        self.input_size = input_size
        self.x, self.y = x.values, y.values
        self.x_input = x[input_size].values
        self.x_encoded = x['encoded'].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x_input[i].astype(np.float)), torch.from_numpy(
            self.x_encoded[i][0].astype(np.float)), self.y[i], self.x_encoded[i][1]


class Text_Classification:

    def __init__(self):
        self.p_project = os.path.dirname(os.path.dirname(__file__))
        self.p_project = self.p_project + r'\NLP_lang-model'
        self.p_resource = self.set_dir_get_path(self.p_project, 'resources')
        self.p_corpus = self.set_dir_get_path(self.p_project, 'corpus')
        self.p_models = self.set_dir_get_path(self.p_project, 'models')
        self.d_model, self.d_model2 = dict(), dict()
        self.d_models = dict()
        self.d_best_model = dict()
        self.vectorizer = None
        self.i_cv = 10
        # self.i_cv = 2
        self.i_splits = 10
        # self.i_splits = 5
        # self.i_splits = 2
        self.l_scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        self.l_test_cols = ['tweet id', 'user handle', 'tweet text', 'time stamp']
        self.l_cols = ['tweet id', 'user handle', 'tweet text', 'time stamp', 'device', 'features_group']
        self.l_features = ['i_hash', 'i_url', 'i_tag', 'i_capital', 'i_punc', 'i_terms', 'puncs', 'i_stopword', 'i_h_c', 
                           'i_b_o', 'i_hours', 'i_days']
        self.df_one_hot = pd.DataFrame()
        self.get_synonym = EDA()
        self.df_results = pd.DataFrame(columns=['Model', 'Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'PRAUC'])
        self.p_results = self.p_project + '\\' + 'results' + '.csv'
        self.i_count_terms = 0
        self.i_count_sequences = 0
        # self.tok = spacy.load('en')
        self.tok = spacy.load('en_core_web_sm')
        self.i_features = 0
        self.emb_size = 16
        self.hidden_size = 16
        self.layer_size = 4
        self.vocabulary_size = 0
        self.i_input_size = 0
        self.i_classes = 2
        self.input_shape = 0
        self.s_model = ''
        self.i_hash, self.i_via, self.i_url, self.i_capital = 0, 0, 0, 0
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'

    @staticmethod
    def set_dir_get_path(path, folder_name):
        """
        function loads a model type '.joblib'
        :param path
        :param folder_name
        :return path
        """
        p_new_dir = path + '\\' + folder_name
        if not os.path.exists(p_new_dir):
            os.makedirs(p_new_dir)
        return p_new_dir

    def set_df_to_csv(self, df, filename, path, b_append=True):
        """
        function appends new data onto a dataframe and saves on disk
        :param df dataframe
        :param filename the name of the file
        :param path of file
        :param b_append boolean to append to dataframe
        :return updated dataframe saved in resources directory path
        """
        p_write = path + '\\' + filename + '.csv'
        if b_append:
            df.to_csv(path_or_buf=p_write, mode='a', index=False, na_rep='NA', encoding='utf-8-sig')
        else:
            df.to_csv(path_or_buf=p_write, mode='w', index=False, na_rep='NA', encoding='utf-8-sig')

    @staticmethod
    def randomize(x_curr, y_curr):
        """
        function randomizes data
        :param x_curr
        :param y_curr
        """
        if isinstance(x_curr, np.ndarray) and isinstance(y_curr, np.ndarray):
            x_rows, x_cols = x_curr.shape
            y_rows, y_cols = y_curr.shape
            nd_curr_data = np.concatenate((x_curr, y_curr), axis=1)
            nd_curr_data = np.random.permutation(nd_curr_data)
            x_new = nd_curr_data[:, :x_cols].copy()
            y_new = nd_curr_data[:, x_cols:].copy()
        else:
            if isinstance(x_curr, pd.Series):
                x_curr = pd.DataFrame(x_curr, columns=['tweet text'])
            y_curr = pd.DataFrame(y_curr, columns=['user handle'])
            l_cols_x = list(x_curr.columns)
            l_cols_y = list(y_curr.columns)
            df_curr_data = pd.DataFrame()
            df_curr_data = df_curr_data.assign(**x_curr)
            df_curr_data = df_curr_data.assign(**y_curr)
            df_curr_data = df_curr_data.sample(frac=1).reset_index(drop=True)
            x_new = df_curr_data[l_cols_x].copy()
            y_new = df_curr_data[l_cols_y].copy()
        return x_new, y_new

    def synonym(self, value):
        """
        function returns synonyms
        :param value term to convert
        """
        m_syn = self.get_synonym.synonym_replacement(value)
        return m_syn

    def synonym_apply(self, value, threshold=0.75):
        """
        function returns synonyms
        :param value term to convert
        :param threshold what probabilty to convert terms
        """
        curr_sequence = ' '.join(value)
        syn_sequence = self.synonym(curr_sequence)
        return syn_sequence

    def split_sets(self, x, y, i_ratio=0.2):
        """
        function splits sets
        """
        i_test_range = int(i_ratio * (x.shape[0]))
        i_train_range = y.shape[0] - i_test_range
        x_test, y_test = x[-i_test_range:, :], y[-i_test_range:]
        test = np.concatenate((x_test, y_test), axis=1)
        x_train, y_train = x[:i_train_range, :], y[:i_train_range:]
        train = np.concatenate((x_train, y_train), axis=1)
        return x_train, y_train, x_test, y_test, train, test

    def split_fold_sets(self, x, y, i_train, i_test, f_val=0):
        """
        function splits sets
        """
        x_eval, x_test = x[i_train], x[i_test]
        y_eval, y_test = y[i_train], y[i_test]
        eval, test = (x_eval, y_eval), (x_test, y_test)
        x_fold, y_fold = np.concatenate((x_eval, x_test), axis=0), np.concatenate((y_eval, y_test), axis=0)
        if f_val == 0:
            return x_eval, None, x_test, y_eval, None, y_test, eval, None, test, x_fold, y_fold
        else:
            length = x_eval.shape[0]
            i_split_val = int(length * f_val)
            i_split_train = length - i_split_val
            x_train, x_val = x_eval[:i_split_train], x_eval[i_split_train:]
            y_train, y_val = y_eval[:i_split_train], y_eval[i_split_train:]
            train, val = (x_train, y_train), (x_val, y_val)
            return x_train, x_val, x_test, y_train, y_val, y_test, train, val, test, x_fold, y_fold

    def load_text(self, s_data):
        """
        function loads data
        :param s_data data name
        """
        p_data = self.p_corpus + '\\' + s_data + '.tsv'
        text = pd.read_csv(p_data, sep='\t', error_bad_lines=False)
        l_row = list(text.columns)
        text.loc[-1] = l_row
        text.index += 1
        text = text.sort_index()
        l_col = list()

        if 'train' in s_data:
            l_col = ['tweet id', 'user handle', 'tweet text', 'time stamp', 'device']  # user handle = author class

        elif 'test' in s_data:
            l_col = ['user handle', 'tweet text', 'time stamp']

        for i in range(len(l_row)):
            text = text.rename(columns={l_row[i]: l_col[i]})

        return text

    def load_dataset(self, s_data):
        """
         function loads data
         :param s_data data name
         """
        p_data = self.p_corpus + '\\' + s_data + '.tsv'

        # read_file = open(p_data, 'r')
        # text = read_file.read()
        # text = read_file.read().rstrip()

        text = pd.read_csv(p_data, sep='\t', error_bad_lines=False)
        l_row = list(text.columns)
        text.loc[-1] = l_row
        text.index += 1
        text = text.sort_index()
        l_col = list()

        if 'train' in s_data:
            l_col = ['tweet id', 'user handle', 'tweet text', 'time stamp', 'device']  # user handle = author class

        elif 'test' in s_data:
            l_col = ['user handle', 'tweet text', 'time stamp']

        for i in range(len(l_row)):
            text = text.rename(columns={l_row[i]: l_col[i]})

        if 'train' in s_data:
            text.drop(['tweet id', 'time stamp'], axis=1, inplace=True)

        elif 'test' in s_data:
            text.drop(['time stamp'], axis=1, inplace=True)

        return text

    def analysis_text(self, data):
        """
         function reports term count, sequence count and class ratios
         :param data data
         """
        print('After Text Normalization:')
        self.count_terms(data)
        self.count_sequences(data)
        self.i_input_size = self.vocabulary_size
        self.class_count(data)

    def transform(self, series):
        """
         function replaces title with first row
         :param series data
         """
        df = pd.DataFrame(series, columns=[series.name])
        l_cols = list(df.columns)
        if len(l_cols) == 1:
            l_cols = l_cols[0]
        nd_uniques = df[l_cols].unique()
        l_uniques = list(nd_uniques)

        for i in range(len(l_uniques)):
            key = l_uniques[i]
            if not pd.isna(key):
                df = df.replace(key, i)

        if series.name == 'user handle' and len(l_uniques) > 2:  # one hot encoding
            df = self.target_encoder(df)

        return df

    def target_encoder(self, curr_df):
        """
         function apllies one hot encoding.
         :param curr_df data
         """
        curr_list = curr_df['user handle']
        curr_dummies = pd.get_dummies(curr_list, prefix=['user handle'])
        self.df_one_hot = self.df_one_hot.append(curr_dummies)
        curr_df = curr_df.replace(2, 1)
        return curr_df

    def parse_text(self, data):
        """
        function parses text and extracts additional features
        :param data data
        """
        data_parser = pd.DataFrame(data[['tweet text', 'user handle']], columns=['tweet text', 'user handle'])
        self.i_hash, self.i_via, self.i_url, self.i_capital = self.parser(data_parser)
        data_parser['tweet text'] = data_parser['tweet text'].apply(self.filter_text)
        return data_parser

    def class_count(self, df_data):
        """
        function plots class ratios
        :param df_data data
        """
        d_percentages = {}
        d_count = {}
        # print('Classes Counts: ')
        srs_numeric = pd.to_numeric(df_data['user handle'], errors='coerce')
        value = srs_numeric.sum()
        fig = plt.figure()
        fig.suptitle('Existing Classes Chart', fontsize=20)
        plt.ylabel('Count', fontsize=16)
        d_count['Non-Trump'] = value / df_data.shape[0]
        d_count['Trump'] = (df_data.shape[0] - value) / df_data.shape[0]
        d_percentages['Non-Trump'] = value
        d_percentages['Trump'] = df_data.shape[0] - value
        plt.bar(*zip(*d_percentages.items()))
        fig1 = plt.gcf()
        # plt.show()
        # plt.draw()
        plt.bar(*zip(*d_count.items()))
        fig2 = plt.gcf()
        # plt.show()
        # plt.draw()

    def parser(self, text):
        """
        function parses text and extracts additional features
        :param text
        """
        text['tweet text'] = text['tweet text'].astype(str)
        reg_caps = r'[A-Z]'
        self.i_hash, self.i_via, self.i_url, self.i_capital = 0, 0, 0, 0
        for row in text['tweet text']:
            for term in row.split(' '):
                if term.startswith('#'):
                    self.i_hash += 1
                elif term.startswith('@'):
                    self.i_via += 1
                elif term.startswith('http'):
                    self.i_url += 1
                else:
                    self.i_capital += len(re.findall(reg_caps, term))
        return self.i_hash, self.i_via, self.i_url, self.i_capital

    @staticmethod
    def filter_text(text):
        """
        function normalizes text (tweets)
        :param text data
        """
        filtered_text = text.replace('<a href="http://instagram.com" rel="nofollow">Instagram</a>', 'instagram')
        filtered_text = filtered_text.replace('<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>', 'webclient')
        filtered_text = filtered_text.replace('<a href="http://www.twitter.com" rel="nofollow">Twitter for BlackBerry</a>', 'blackberry')
        filtered_text = filtered_text.replace('<a href="http://twitter.com/#!/download/ipad" rel="nofollow">Twitter for iPad</a>', 'ipad')
        filtered_text = filtered_text.replace('<a href="https://periscope.tv" rel="nofollow">Periscope.TV</a>', 'tv')
        filtered_text = filtered_text.replace('<a href="https://about.twitter.com/products/tweetdeck" rel="nofollow">TweetDeck</a>', 'webclient')
        filtered_text = filtered_text.replace('<a href="http://www.facebook.com/twitter" rel="nofollow">Facebook</a>', 'facebook')
        filtered_text = lower_case(filtered_text)
        reg1 = '[\s+\n\t\r\d\/\.\,\&\!\"\?\!\@\$\(\)\_\%\^\*\[\]\}\{\+\-\#\:\;\'-]'
        reg2 = '[\s+\n\t\r\d\/\.\,\&\!\"\?\!\$\(\)\_\%\^\*\[\]\}\{\+\-\:\;\'-]'
        reg3 = '([,.;!?:)])'
        reg4 = '([/"&*=])'
        reg5 = '([(])'
        reg6 = '\s+'
        reg7 = '\s{2,}'
        sub1 = ' '
        sub2 = ''
        sub3 = r' \1'
        sub4 = r' \1 '
        sub5 = r'\1 '
        filtered_text = re.sub(reg3, sub3, filtered_text)
        filtered_text = re.sub(reg4, sub4, filtered_text)
        filtered_text = re.sub(reg5, sub5, filtered_text)
        filtered_text = re.sub(reg6, sub1, filtered_text)
        filtered_text = re.sub(reg7, sub1, filtered_text)
        filtered_text = re.sub(reg1, sub1, filtered_text)
        filtered_text = filtered_text.replace('https', '')
        filtered_text = filtered_text.strip()
        filtered_text = ' '.join(filtered_text.split())
        return filtered_text

    def preprocess_text(self, data):
        """
        function preprocess text
        :param data text
        """
        nd_uniques = data['user handle'].unique()
        l_uniques = list(nd_uniques)

        for i in range(len(l_uniques)):
            key = l_uniques[i]
            if not pd.isna(key):
                data = data.replace(key, i)

        if len(l_uniques) > 2:  # one hot encoding
            data = self.target_encoder(data)

        return data

    def tokenize_nltk(self, text):
        """
        function applies and returns word tokenizing on text
        :param text
        """
        return word_tokenize(text)

    def tokenize(self, text):
        """
        function tokenizes text with spacy
        :param text
        """
        return [token.text for token in self.tok.tokenizer(text)]

    def encode_sentence(self, text, vocab2index, N=70):
        """
        function encodes unknown terms after normalization
        """
        tokenized = self.tokenize_nltk(text)
        encoded = np.zeros(N, dtype=int)
        enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
        length = min(N, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded, length

    def set_train_features(self, data):
        """
        function uses TF-IDF Vectorizer to transform from string to columned features in Train set.
        :param data train text
        """
        l_stopwords = set_stopwords()
        max_features = len(self.i_count_terms)
        min_features = 2
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=l_stopwords, max_features=max_features,
                                     min_df=min_features)
        x_1 = self.vectorizer.fit_transform(data['tweet text'])
        l_cols_features = self.vectorizer.get_feature_names_out()
        i_features = len(l_cols_features)
        self.i_features = i_features
        x_1 = pd.DataFrame(x_1.toarray(), columns=l_cols_features)
        self.i_input_size = x_1.shape[-1]
        y = pd.DataFrame(data['user handle'], columns=['user handle'])
        x_1 = x_1.assign(**y)
        return x_1

    def set_test_features(self, test):
        """
        function uses TF-IDF Vectorizer to transform from string to columned features in Test set.
        :param test text
        """
        l_stopwords = set_stopwords()
        s_test = ' '.join(test['tweet text'])
        l_test = s_test.split(' ')
        i_terms = collections.Counter(l_test)
        test_vocabulary_size = len(i_terms)
        max_features = len(i_terms)
        min_features = 2
        x_test = self.vectorizer.transform(test['tweet text'])
        l_cols_features = self.vectorizer.get_feature_names_out()
        i_features = len(l_cols_features)
        x_test = pd.DataFrame(x_test.toarray(), columns=l_cols_features)
        y = pd.DataFrame(test['user handle'], columns=['user handle'])
        test = x_test.assign(**y)
        return test

    def preprocess(self, x, y):
        """
        function preprocess data and transforms text to TF-IDF vectors.
        :param x text
        :param y label
        """
        y = self.transform(y['user handle'])
        l_curr_cols = list(x.columns)
        x_1 = pd.DataFrame(x['tweet text'], columns=['tweet text'])
        x_1['tweet text'] = x_1['tweet text'].astype(str)
        l_stopwords = set_stopwords()
        # x = self.get_time stamps(x_1)
        self.i_hash, self.i_via, self.i_url, self.i_capital = self.parser(x_1)
        if 'device' in l_curr_cols:
            x_2 = self.transform(x['device'])
            print('Before Normalizing Terms:')
            self.count_terms(x_1)
            x_1['tweet text'] = x_1['tweet text'].apply(self.filter_text)
            print('After Normalizing Terms:')
            self.count_terms(x_1)
            self.count_sequences(x_1)
            
            # self.i_input_size = i_features
            # self.i_input_size = self.i_count_terms
            # self.i_input_size = x_1.shape[0]
            # self.i_input_size = x_1.shape[1]
            self.i_input_size = x_1.shape[-1]

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=l_stopwords, max_features=len(self.i_count_terms))
            x_1 = vectorizer.fit_transform(x_1['tweet text'])
            l_cols_features = vectorizer.get_feature_names_out()
            i_features = len(l_cols_features)
            self.i_features = i_features

            x_1 = pd.DataFrame(x_1.toarray(), columns=l_cols_features)
            # x_1 = x_1.assign(**x_2)
        else:
            x_1['tweet text'] = x_1['tweet text'].apply(self.filter_text)
            l_stopwords = set_stopwords()
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=l_stopwords,
                                         max_features=len(self.i_count_terms))
            x_1 = vectorizer.fit_transform(x_1['tweet text'])
            l_cols_features = vectorizer.get_feature_names_out()  # does it need to be same columns as train?
            x_1 = pd.DataFrame(x_1.toarray(), columns=l_cols_features)

        return x_1, y

    @staticmethod
    def get_timestamps(x):
        """
        function appends timestamp feature to data
        :param x text
        """
        data = x.copy()
        data['time stamp'] = pd.to_datetime(data['time stamp'])
        data['hour'] = data['time stamp'].dt.hour
        data['day'] = data['time stamp'].dt.weekday
        data['year'] = data['time stamp'].dt.year
        data['month'] = data['time stamp'].dt.month
        data = data.drop('time stamp', axis=1, inplace=True)
        return data

    @staticmethod
    def remove_na(x, y):
        """
        function removes na if discvoers
        :param x text
        :param y text
        """
        if isinstance(x, pd.Series):
            arr_na = x.index[x.isnull()]
        else:
            x = np.delete(x, -1, axis=1)
            # arr_na = np.argwhere(np.isnan(x))
            # for i in range(len(arr_na)):
            #     x = np.delete(x, arr_na[i][0], 0)
            #     y = np.delete(y, arr_na[i][0], 0)
        return x, y

    def init_models(self):
        """
        function loads models
        """
        self.d_models['svm_linear'] = SVC(kernel='rbf', C=1, random_state=1, gamma='scale')

        self.d_models['svm_non_linear'] = SVC(kernel='linear', C=1, random_state=2, gamma='auto')

        self.d_models['log_reg'] = LogisticRegression(class_weight='balanced', penalty='l2', random_state=3)

        # sequence_size = 8
        sequence_size = 16
        # sequence_size = 32
        # sequence_size = 64
        # sequence_size = 128
        # sequence_size = 256

        layer_size = 2
        # layer_size = 4
        # layer_size = 8

        # hidden_size = 512
        # hidden_size = 256
        # hidden_size = 128
        hidden_size = 32

        emb_size = 32
        # emb_size = 128
        # emb_size = 512

        max_terms = 50000  # max count of most frequent terms
        max_sequences = 256  # max number of words per sentence

        # self.input_shape = 12

        self.d_models['NN'] = NN(input_size=self.input_shape, num_classes=self.i_classes).to(self.device)

        emb_size = 16
        hidden_size = 16
        layer_size = 4
        input_size = len(self.l_features)

        # self.d_models['RNN'] = RNN5(vocab_size=self.vocabulary_size, emb_size=emb_size, hidden_size=hidden_size,
        #                             layer_size=layer_size, input_size=input_size, device=self.device).to(self.device)

        self.d_models['RNN'] = RNN5(vocab_size=self.vocabulary_size, emb_size=emb_size, hidden_size=hidden_size,
                                   layer_size=layer_size, input_size=input_size,
                                   device=self.device).to(self.device)
        
        # self.d_models['BRNN'] = BRNN(input_size=self.vocabulary_size, hidden_size=hidden_size, layer_size=layer_size,
        #                              num_classes=self.i_classes, device=self.device).to(self.device)
        
        # self.d_models['GRU'] = GRU(self.vocabulary_size, hidden_size, layer_size, self.i_classes,
        #                                    sequence_size, self.device).to(self.device)
        
        self.d_models['LSTM'] = LSTM(self.vocabulary_size, hidden_size, emb_size, layer_size,
                                     sequence_size, self.device).to(self.device)

    def test(self, model, test):
        """
        function tests pytorch model
        :param model
        :param test set
        """
        i_correct, i_samples = 0, 0
        num_correct, num_samples = 0, 0
        l_preds = list()
        model.eval()
        with torch.no_grad():
            for x_batch in test:
                x_batch = x_batch.to(self.device)
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                x = x.reshape(x.shape[0], -1)
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
                test_acc = num_correct / num_samples
                test_acc = f" {float(test_acc) * 100:.3f}"
                print(f'Accuracy: {test_acc}')
                y_test_pred = model(x_batch)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_preds = torch.round(y_test_pred)
                l_preds.append(y_preds.cpu().numpy())
        l_curr_preds = [a.squeeze().tolist() for a in l_preds]
        return l_curr_preds

    @staticmethod
    def shuffle(data):
        """
        function randomizes the data
        :param data
        """
        i_errors = 0
        try:
            if isinstance(data, torch.Tensor):
                data = tf.random.shuffle(data, seed=2, name=None)
            else:
                data = data.sample(frac=1).reset_index(drop=True)
        except AttributeError as ae:
            i_errors += 1
            # print(ae)
        return data

    @staticmethod
    def split_train(data, i_train, i_eval, thresh=0.75):
        """
        function splits training data
        """
        size = data.shape[0]
        split = int(size*thresh)
        y = pd.DataFrame(data[['user handle']], columns=['user handle'])
        x = data.drop('user handle', axis=1).copy()
        # x = x.iloc[:split]
        # y = y.iloc[:split]
        return x, y

    @staticmethod
    def split_test(data):
        """
        function splits test data
        """
        y = pd.DataFrame(data[['user handle']], columns=['user handle'])
        x = data.drop('user handle', axis=1).copy()
        return x, y

    def set_dummies(self, l_cols, df_dummies):
        """
        function creates a one-hot vector
        :param l_cols list of columns to transform (must be strings)
        :param df_dummies columns to apply
        """
        df_one_hot = pd.DataFrame()
        for col in l_cols:
            curr_list = (df_dummies[col])
            curr_dummies = pd.get_dummies(curr_list, prefix=[col])
            df_one_hot = df_one_hot.append(curr_dummies)
        return df_one_hot

    def test_rnn(self, model, test):
        """
        function tests RNN pytorch model
        :param model
        :param test set
        """
        l_preds = []
        model.eval()
        with torch.no_grad():
            for x_input, x_enc, y, l in test:
                x_input = x_input.float().to(self.device)
                x_enc = x_enc.long().to(self.device)
                y = y.long().to(self.device)
                y_test_pred = model(x_enc, x_input, l).to(self.device)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred = torch.round(y_test_pred)
                l_preds.append(y_pred.cpu().numpy())

        l_curr_preds = [a.squeeze().tolist() for a in l_preds]
        l_preds_new = list()
        for element in l_curr_preds:
            for i in l:
                l_preds_new.append(i)
        l_curr_preds = np.array(l_preds_new)
        return l_curr_preds

    def test_lstm(self, model, test):
        """
        function tests LSTM pytorch model
        :param model
        :param test set
        """
        l_preds = list()
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, l in test:
                y_test_pred = model(x_batch, l)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_preds = torch.round(y_test_pred)
                l_preds.append(y_preds.cpu().numpy())
        l_curr_preds = [a.squeeze().tolist() for a in l_preds]
        l_preds_new = list()
        for element in l_curr_preds:
            for i in l:
                l_preds_new.append(i)
        l_curr_preds = np.array(l_preds_new)
        return l_curr_preds
    
    def train_rnn(self, model, train):
        """
        function trains RNN pytorch model
        :param model
        :param train set
        """
        # i_epochs = 50
        i_epochs = 25
        # i_epochs = 1
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        size = len(train)
        for curr_epoch in range(i_epochs):
            model.train()
            sum_loss = 0.0
            total = 0
            epoch_loss = 0
            epoch_acc = 0
            for x_input, x_enc, y, l in train:
                x_input = x_input.float().to(self.device)
                x_enc = x_enc.long().to(self.device)
                y = y.long().to(self.device)
                y_pred = model(x_enc, x_input, l).to(self.device)
                optimizer.zero_grad()
                loss = criterion(y_pred, y.unsqueeze(1).float())              
                y_test = y.unsqueeze(1).float()
                y_pred_tag = torch.round(torch.sigmoid(y_pred))
                correct_results_sum = (y_pred_tag == y_test).sum().float()
                acc = correct_results_sum / y_test.shape[0]
                acc = torch.round(acc * 100)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += acc.item()

            final_acc = epoch_acc / size
            final_loss = epoch_loss / size
            final_acc, final_loss = float("{:.3f}".format(final_acc)), float("{:.3f}".format(final_loss))
            print(f'RNN Training: Epoch {curr_epoch}: Accuracy: {final_acc} and Loss: {final_loss}.')

    def train_lstm(self, model, train):
        """
        function trains LSTM pytorch model
        :param model
        :param train set
        """
        # i_epochs = 1
        # i_epochs = 2
        i_epochs = 25
        # i_epochs = 50
        # learning_rate = 0.001
        learning_rate = 0.0001

        criterion = nn.BCEWithLogitsLoss()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, learning_rate)
        size = len(train)
        for curr_epoch in range(i_epochs):
            model.train()
            epoch_loss = 0
            epoch_acc = 0
            for x, y, l in train:
                x, y = x.long(), y.long()
                y_pred = model(x, l)
                optimizer.zero_grad()
                loss = criterion(y_pred, y.unsqueeze(1).float())
                # loss = criterion(y_pred, y.float())
                y_pred_tag = torch.round(torch.sigmoid(y_pred))
                correct_results_sum = (y_pred_tag == y).sum().float()
                acc = correct_results_sum / y.shape[0]
                acc = torch.round(acc * 100)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
            final_acc = epoch_acc / size
            final_loss = epoch_loss / size
            final_acc, final_loss = float("{:.3f}".format(final_acc)), float("{:.3f}".format(final_loss))
            print(f'LSTM Training: Epoch {curr_epoch}: Accuracy: {final_acc} and Loss: {final_loss}.')

    def train(self, model, train, s_model):
        """
        function trains pytorch model
        :param model
        :param train set
        :param s_model name
        """
        # i_epochs = 1
        # i_epochs = 2
        # i_epochs = 25
        i_epochs = 50
        # learning_rate = 0.001
        learning_rate = 0.0001
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        size = len(train)
        model.train()
        for curr_epoch in range(1, i_epochs):
            epoch_loss = 0
            epoch_acc = 0
            for x_batch, y_batch in train:
                # size = train.shape[1]
                # data = x_batch
                # targets = y_batch
                # data = torch.from_numpy(data).float()
                # targets = torch.from_numpy(targets).float()
                # data = data.to(device=self.device)
                # targets = targets.to(device=self.device)
                # scores = model(data)
                # loss = criterion(scores, targets)  # forward propagation
                # optimizer.zero_grad()  # zero previous gradients
                # loss.backward()  # back-propagation
                # optimizer.step()  # gradient descent or adam step
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                # shape1 = x_batch.shape[0]
                # shape2 = x_batch.shape[1]
                # x_batch = tf.reshape(x_batch, [shape1, shape2])
                # x_batch = tf.transpose(x_batch)
                optimizer.zero_grad()

                y_pred = model(x_batch)

                # loss = criterion(y_pred, y_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                y_pred_tag = torch.round(torch.sigmoid(y_pred))
                correct_results_sum = (y_pred_tag == y_batch).sum().float()
                acc = correct_results_sum / y_batch.shape[0]
                acc = torch.round(acc * 100)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            final_acc = epoch_acc / size
            final_loss = epoch_loss / size
            final_acc, final_loss = float("{:.3f}".format(final_acc)), float("{:.3f}".format(final_loss))
            print(f'Model {s_model} Finished Training: Epoch {curr_epoch}')

    def prepare(self, data):
        """
        function preprocess data
        :param data
        """
        if data.shape[1] == 2:  # test set
            x = data['tweet text']
        else:  # train set
            x = data[['tweet text', 'device']]
        y = data['user handle']
        x, y = self.randomize(x, y)  # shuffles data for the first time
        x, y = self.preprocess(x, y)  # applies target encoders and generates term frequency vectors
        x, y = self.reformat(x, y)  # transforms to numpy
        return x, y

    def count_terms(self, data):
        """
        function counts terms
        :param data
        """
        data = ' '.join(data['tweet text'])
        l_data = data.split(' ')
        self.i_count_terms = collections.Counter(l_data)
        self.vocabulary_size = len(self.i_count_terms)
        print(f'Detected {len(self.i_count_terms)} terms.')

    def count_sequences(self, data):
        """
        function counts sequences
        :param data
        """
        sequences = data['tweet text'].str.len()
        self.i_count_sequences = sequences.size
        max_seq = int(sequences.max())
        min_seq = int(sequences.min())
        mean_sec = int(sequences.mean())
        print(f'Detected {self.i_count_sequences} sequences.')
        print(f'Max terms in a sequence: {max_seq}')
        print(f'Min terms in a sequence: {min_seq}')
        print(f'Mean terms in a sequence: {mean_sec}')

    @staticmethod
    def reformat(x, y):
        """
        function transform to numpy
        """
        return x.to_numpy(), y.to_numpy()

    @staticmethod
    def set_data(train, test, b_train=True):
        """
        function splits train and test to x and y
        """
        if b_train:
            x = train['tweet text']
            y = train['user handle']
        else:
            x = np.concatenate((train['tweet text'], test['tweet text']), axis=0)
            y = np.concatenate((train['user handle'], test['user handle']), axis=0)
        return x, y

    def get_params(self, model):
        """
        function return model parameters
        :param model
        """
        d_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                d_params[name] = param.data
        model = NN(input_size=self.input_shape, num_classes=self.i_classes).to(self.device)
        return d_params, model

    def decay(self, epoch):
        """
        function modifies decay changes while training
        :param epoch
        """
        if epoch < 3:
            print('Learning Rate for Epoch: 1e-6')
            return 1e-6
        elif epoch >= 3 and epoch < 10:
            print('Learning Rate for Epoch: 1e-8')
            return 1e-8
        elif epoch >= 10 and epoch < 20:
            print('Learning Rate for Epoch: 1e-10')
            return 1e-10
        else:
            print('Learning Rate for Epoch: 1e-12')
            return 1e-12

    def set_callbacks(self, test):
        """
        function sets callbacks
        :param test set for the Tensor History class to monitor its performance on the validation set
        :returns list of callbacks
        """
        l_callbacks = list()
        params = "val_loss"
        # lr_factor = 0.99
        lr_factor = 0.1
        # min_d = 1e-4
        min_d = 0

        cb_stop = tf.keras.callbacks.EarlyStopping(monitor=params, patience=8, verbose=1, min_delta=min_d,
                                                   restore_best_weights=True)

        cb_plateu = tf.keras.callbacks.ReduceLROnPlateau(monitor=params, factor=lr_factor, patience=8, verbose=1,
                                                         min_delta=min_d)
        cb_history = History(test)

        cb_board = tf.keras.callbacks.TensorBoard(self.p_resource, update_freq=1)

        cb_lr_decay = tf.keras.callbacks.LearningRateScheduler(self.decay)

        l_callbacks = [cb_stop, cb_plateu, cb_lr_decay, cb_history, cb_board]

        return l_callbacks

    def read_files(self, path, columns, dt_index):
        """
        function reads files
        """
        df = pd.read_csv(path, sep='\t', header=None, quoting=csv.QUOTE_NONE, parse_dates=[dt_index],
                         infer_datetime_format=True)
        df.columns = columns
        return df

    def norm_text(self, text):
        """
        function normalizes text
        :param text data
        """
        text = text.lower()
        text = re.sub('\s+', ' ', text)
        text = re.sub('([,.;!?:)])', r' \1', text)
        text = re.sub('([(])', r'\1 ', text)
        text = re.sub('([/"&*=])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        return text

    def extract_content_features(self, text):
        """
        function extracts additional features
        :param text data
        """
        words = text.split(' ')
        content_words = []
        i_hash = 0
        i_tag = 0
        i_url = 0
        i_stopword = 0
        i_capital = 0
        i_b_o = 0
        i_h_c = 0
        clinton_names = ['hillary', 'clinton', 'crooked']
        obama_names = ['obama', 'barack']

        for word in words:
            if word.startswith(('"', '.')):
                content_words.append(word[0])
                word = word[1:]
            if word.startswith('@'):
                i_tag += 1
                if word == '@HillaryClinton':
                    i_h_c += 1
                elif word == '@BarackObama':
                    i_b_o += 1
                normal = self.norm_text(word)
                for token in normal.split(' '):
                    content_words.append(token)
            elif word.startswith('#'):
                i_hash += 1
                normal = self.norm_text(word)
                for token in normal.split(' '):
                    content_words.append(token)
            elif word.startswith('http'):
                i_url += 1
            else:
                i_capital += len(re.findall(r'[A-Z]', word))
                normal = self.norm_text(word)
                for token in normal.split(' '):
                    content_words.append(token)
                    if token in stopwords.words('english'):
                        i_stopword += 1
                    if any(obama in token for obama in obama_names):
                        i_b_o += 1
                    if any(clinton in token for clinton in clinton_names):
                        i_h_c += 1

        features_group = ' '.join(content_words)
        i_punc = len(re.findall(r'[!()"-?=,.:;]', features_group))
        i_terms = len(content_words) - i_punc
        length = 1 if i_terms == 0 else i_terms
        puncs = float(i_punc) / length

        return features_group, i_hash, i_url, i_tag, i_capital, i_punc, i_terms, puncs, i_stopword, i_h_c, i_b_o

    def extract_features(self, df):
        """
        function extracts additional features
        :param df data
        """
        content_df = df.apply(lambda row: self.extract_content_features(row['tweet text']), axis='columns',
                              result_type='expand')
        content_df.columns = ['features_group', 'i_hash', 'i_url', 'i_tag', 'i_capital',
                              'i_punc', 'i_terms', 'puncs', 'i_stopword',
                              'i_h_c', 'i_b_o']
        out_df = pd.concat([df, content_df], axis='columns')
        out_df['i_hours'] = out_df['time stamp'].dt.hour
        out_df['i_days'] = out_df['time stamp'].dt.dayofweek
        return out_df

    def process_text(self):
        """
        function reports mean, min and max terms per sequence
        """
        l_cols_train = ['tweet id', 'user handle', 'tweet text', 'time stamp', 'device']
        l_cols_test = ['user handle', 'tweet text', 'time stamp']
        p_train = self.p_corpus + '\\' + 'trump_train' + '.tsv'
        p_test = self.p_corpus + '\\' + 'trump_test' + '.tsv'
        train_data = self.read_files(p_train, l_cols_train, 3)
        train_data['target'] = np.where(train_data['device'] == 'android', 0, 1)
        test_data = self.read_files(p_test, l_cols_test, 2)
        train_data[train_data['target'] == 1].head(50)

        sequences = train_data['tweet text'].str.len()
        max_seq = int(sequences.max())
        min_seq = int(sequences.min())
        mean_sec = int(sequences.mean())
        print(f'Max terms in a sequence: {max_seq}')
        print(f'Min terms in a sequence: {min_seq}')
        print(f'Mean terms in a sequence: {mean_sec}')

        train_features = self.extract_features(train_data)
        train = train_features.copy()
        test_features = self.extract_features(test_data)
        test = test_features.copy()
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', min_df=2)
        sequences = train.features_group.to_numpy()
        text = self.vectorizer.fit_transform(sequences)
        text = pd.DataFrame(text.toarray(), columns=self.vectorizer.get_feature_names())

        for curr_col in train.columns:
            if curr_col not in self.l_cols:
                text[curr_col] = train[curr_col]

        text['target'] = train['target']
        self.scaler = MinMaxScaler()
        text[self.l_features] = self.scaler.fit_transform(text[self.l_features])
        x = text.drop(['target'], axis=1)
        y = text['target']
        self.input_shape = x.shape[-1]
        self.input_size = x.shape[-1]
        return x, y, text, train_features

    def train_models(self, train, test):
        """
        main function in charge of training, testing and evaluating all models
        :param train data
        :param test data
        """

        # x_train_set, y_train_set = self.prepare(train)  # preprocess train set
        # x_test, y_test = self.prepare(test)  # preprocess test set
        # test = np.concatenate((x_test, y_test), axis=1)
        # test = self.shuffle(test)
        
        x, y, text, train_features = self.process_text()

        # x_test, y_test = self.split_test(test)
        # x_test, y_test = self.reformat(x_test, y_test)

        self.init_models()

        high_score = float('-inf')
        l_scores = list()
        l_best_model = list()
        l_model = list()
        l_model2 = list()

        i_epochs = 1
        # i_epochs = 25
        # i_epochs = 50

        # steps_epoch = len(x) // i_batch
        steps_epoch = 8
        # steps_val = len(x_val) // i_batch
        steps_val = 4

        l_callbacks = self.set_callbacks(test)

        for s_model, o_model in self.d_models.items():
            i_fold = 0
            d_scores = {'test_acc': list(), 'precision': list(), 'recall': list(), 'f1': list(), 'auc': list(),
                        'pr_auc': list()}

            # for i_train, i_eval in k_outer.split(x_train_set, y_train_set):
            #     x_train, x_eval = x_train_set[i_train], x_train_set[i_eval]
            #     y_train, y_eval = y_train_set[i_train], y_train_set[i_eval]
            k_outer = StratifiedKFold(n_splits=self.i_splits, random_state=9, shuffle=True)
            k_outer = KFold(n_splits=self.i_splits, random_state=9, shuffle=True)
            # for i_train, i_eval in k_outer.split(train):
            for i_train, i_eval in k_outer.split(x):
                # train = self.shuffle(train)
                # x_train, y_train = self.split_train(train, i_train, i_eval)
                # x_train, y_train = self.reformat(x_train, y_train)

                x_train, x_test = x.iloc[i_train], x.iloc[i_eval]
                y_train, y_test = y.iloc[i_train], y.iloc[i_eval]

                # x_train, y_train = self.remove_na(x_train, y_train)
                # x_eval, y_eval = self.remove_na(x_eval, y_eval)
                # x_train_set, y_train_set = self.remove_na(x_train_set, y_train_set)

                i_run_start = time.time()

                if s_model == 'NN':
                    # i_batch = 16
                    # i_batch = 32
                    i_batch = 64
                    batch = 1
                    # train = Train(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
                    train = Train(torch.FloatTensor(x_train.to_numpy()),
                                           torch.FloatTensor(y_train.to_numpy()))
                    train_batch = DataLoader(dataset=train, batch_size=i_batch, shuffle=True)
                    # test = Test(torch.FloatTensor(x_test))
                    test = Test(torch.FloatTensor(x_test.to_numpy()))
                    test_batch = DataLoader(dataset=test, batch_size=batch)
                    self.train(o_model, train_batch, s_model)  # train
                    y_preds = self.test(o_model, test_batch)  # predict
                    # train = np.concatenate((x_train, y_train), axis=1)
                    # logits = o_model(train)  # training
                    # pred_probab = nn.Softmax(dim=1)(logits)  # predict
                    # y_pred = pred_probab.argmax(1)
                    # print(f'Predicted class: {y_pred}')
                    # for name, param in o_model.named_parameters():
                    #     print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n')
                elif s_model == 'RNN':
                    text['features_group'] = train_features['features_group']
                    text = text[text['features_group'].str.len() > 0]
                    counts = Counter()
                    for index, row in text.iterrows():
                        counts.update(self.tokenize(row['features_group']))
                    vocab2index = {"": 0, "UNK": 1}
                    words = ["", "UNK"]
                    for word in counts:
                        vocab2index[word] = len(words)
                        words.append(word)
                    text['encoded'] = text['features_group'].apply(lambda x: np.array(self.encode_sentence(x, vocab2index)))
                    x_enc = list(text['encoded'])
                    y_target = list(text['target'])
                    x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(text.drop(['features_group', 'target'], axis=1),
                                                                        text['target'], test_size=0.1)
                    train_ds = TextDataset(self.l_features, x_train_set, y_train_set)
                    valid_ds = TextDataset(self.l_features, x_test_set, y_test_set)
                    batch_size = 64
                    vocab_size = len(words)
                    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
                    val_dl = DataLoader(valid_ds, batch_size=batch_size)
                    emb_size = 16
                    hidden_size = 16
                    layer_size = 4
                    input_size = len(self.l_features)
                    d_params, o_model = self.get_params(o_model)
                    i_batch = 64
                    batch = 1
                    train = Train(torch.FloatTensor(x_train.to_numpy()),
                                           torch.FloatTensor(y_train.to_numpy()))
                    train_batch = DataLoader(dataset=train, batch_size=i_batch, shuffle=True)
                    test = Test(torch.FloatTensor(x_test.to_numpy()))
                    test_batch = DataLoader(dataset=test, batch_size=batch)
                    self.train(o_model, train_batch, s_model)  # train
                    y_preds = self.test(o_model, test_batch)  # predict
                elif s_model == 'LSTM':
                    batch_size = 64
                    train_set = Text(x_train, y_train)
                    eval_set = Text(x_test, y_test)
                    train_batch = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                    eval_batch = DataLoader(eval_set, batch_size=batch_size)
                    d_params, o_model = self.get_params(o_model)
                    i_batch = 64
                    batch = 1
                    train = Train(torch.FloatTensor(x_train.to_numpy()),
                                           torch.FloatTensor(y_train.to_numpy()))
                    train_batch = DataLoader(dataset=train, batch_size=i_batch, shuffle=True)
                    test = Test(torch.FloatTensor(x_test.to_numpy()))
                    test_batch = DataLoader(dataset=test, batch_size=batch)
                    self.train(o_model, train_batch, s_model)  # train
                    y_preds = self.test(o_model, test_batch)  # predict
                else:
                    # scores = cross_validate(o_model, x, y, cv=10, scoring=self.l_scoring)
                    # curr_history = o_model.fit(x_train_set, y_train_set)
                    curr_history = o_model.fit(x_train, y_train)
                    y_preds = o_model.predict(x_test)
                    # y_preds = np.squeeze(y_preds)
                    y_preds = y_preds.reshape(-1, 1)
                i_run_end = time.time()
                run_time = i_run_end - i_run_start
                run_time = float("{:.2f}".format(run_time))
                print(f'Done training and testing {s_model}. Completed in: {run_time} seconds.')
                test_acc = accuracy_score(y_test, y_preds) * 100
                precision = precision_score(y_test, y_preds) * 100  # micro for imbalanced classes
                recall = recall_score(y_test, y_preds) * 100
                f1 = f1_score(y_test, y_preds) * 100
                arr_fpr, arr_tpr, threshold_auc = roc_curve(y_test, y_preds)  # auc curve
                auc_score = roc_auc_score(y_test, y_preds)  # auc score
                arr_precision, arr_recall, threshold_pr_auc = precision_recall_curve(y_test, y_preds)  # pr_auc curve
                pr_auc_score = auc(arr_recall, arr_precision)  # pr_auc score

                test_acc, precision, recall, f1, auc_score, pr_auc_score = float("{:.3f}".format(test_acc)), \
                                                                           float("{:.3f}".format(precision)), \
                                                                           float("{:.3f}".format(recall)), \
                                                                           float("{:.3f}".format(f1)), \
                                                                           float("{:.3f}".format(auc_score)), \
                                                                           float("{:.3f}".format(pr_auc_score))

                if f1 > high_score:
                    high_score = f1
                    l_best_model = [s_model, test_acc, precision, recall, f1, auc_score, pr_auc_score, o_model]
                if s_model == 'NN' or s_model == 'RNN' or s_model == 'LSTM':
                    l_model = [s_model, test_acc, precision, recall, f1, auc_score, pr_auc_score, o_model]
                else:
                    l_model2 = [s_model, test_acc, precision, recall, f1, auc_score, pr_auc_score, o_model]

                print(f'{s_model} {self.i_cv}-Fold CV Scores: ')
                print(f'Accuracy: {test_acc}')
                print(f'Precision: {precision}')
                print(f'Recall: {recall}')
                print(f'F1: {f1}')
                print(f'AUC: {auc_score}')
                print(f'PR-AUC: {pr_auc_score}')

                d_scores['test_acc'].append(test_acc)
                d_scores['precision'].append(precision)
                d_scores['recall'].append(recall)
                d_scores['f1'].append(f1)
                d_scores['auc'].append(auc_score)
                d_scores['pr_auc'].append(pr_auc_score)

                d_curr_results = {'Model': s_model, 'Fold': i_fold,
                                  'Accuracy': test_acc,
                                  'Precision': precision,
                                  'Recall': recall,
                                  'F1': f1,
                                  'AUC': auc_score,
                                  'PRAUC': pr_auc_score}

                self.df_results.index += 1  # Save
                self.df_results = self.df_results.append(d_curr_results, ignore_index=True)

                l_scores.append(l_model2)
                l_curr_best = l_best_model
                i_fold += 1

            mean_acc = mean(d_scores['test_acc'])
            mean_precision = mean(d_scores['precision'])
            mean_recall = mean(d_scores['recall'])
            mean_f1 = mean(d_scores['f1'])
            mean_auc = mean(d_scores['auc'])
            mean_pr_auc = mean(d_scores['pr_auc'])

            mean_acc, mean_precision, mean_recall, mean_f1, mean_auc, mean_pr_auc = float("{:.3f}".format(mean_acc)), \
                                                                                    float("{:.3f}".format(mean_precision)), \
                                                                                    float("{:.3f}".format(mean_recall)), \
                                                                                    float("{:.3f}".format(mean_f1)), \
                                                                                    float("{:.3f}".format(mean_auc)), \
                                                                                    float("{:.3f}".format(mean_pr_auc))

            d_curr_results = {'Model': s_model, 'Fold': 'AVG',
                              'Accuracy': mean_acc,
                              'Precision': mean_precision,
                              'Recall': mean_recall,
                              'F1': mean_f1,
                              'AUC': mean_auc,
                              'PRAUC': mean_pr_auc}

            self.df_results.index += 1
            self.df_results = self.df_results.append(d_curr_results, ignore_index=True)
            # self.df_results = pd.DataFrame()

        if not os.path.exists(self.p_results):
            self.set_df_to_csv(self.df_results, 'results', self.p_project)

        s_best_model = l_model[0]
        best_score = l_model[4]  # by f1
        print(f'Best score by F1 is for model {s_best_model}: {best_score}')
        o_best_model = l_model[7]  # model object
        self.save_model(o_best_model, self.p_models, s_best_model)

    def save_model(self, o_model, p_model, s_model):
        """
        function saves a model type '.h5'
        :param curr_model_object current model builder
        :param curr_model_name string
        :return model saved in models directory path
        """
        if s_model == 'NN':
            p_save = p_model + '\\' + s_model + '_' + str(self.input_shape)
            torch.save(o_model.state_dict(), p_save)
        elif s_model == 'RNN' or s_model == 'LSTM':
            p_save = p_model + '\\' + s_model + '_' + str(self.vocabulary_size)
            torch.save(o_model.state_dict(), p_save)
        elif s_model == 'svm-linear' or s_model == 'svm-non-linear' or s_model == 'log-reg':
            p_save = p_model + '\\' + s_model
            p_save_joblib = p_save + '.joblib'
            joblib.dump(o_model, p_save_joblib)

    def load_model(self, p_best_model, s_model):
        """
        function loads a model type '.h5'
        :param b_json if a json object needs to be loaded as well.
        :return model
        """
        if s_model == 'NN':
            o_model = NN(input_size=self.input_shape, num_classes=self.i_classes).to(self.device)
        elif s_model == 'RNN':
            emb_size = 16
            hidden_size = 16
            layer_size = 4
            input_size = len(self.l_features)
            o_model = RNN5(vocab_size=self.vocabulary_size, emb_size=emb_size, hidden_size=hidden_size,
                           layer_size=layer_size, input_size=input_size,
                           device=self.device).to(self.device)
        elif s_model == 'LSTM':
            sequence_size = 16
            emb_size = 16
            hidden_size = 16
            layer_size = 4
            o_model = LSTM(self.vocabulary_size, hidden_size, emb_size, layer_size,
                           sequence_size, self.device).to(self.device)
        else:
            o_model = joblib.load(p_best_model)
        if s_model == 'NN' or s_model == 'RNN' or s_model == 'LSTM':
            d_params, o_model = self.get_params(o_model)
            # o_model.load_state_dict(torch.load(p_best_model))
            # o_model.to(self.device)
        return o_model

    def get_model_name(self, path):
        if 'NN' in path:
            index1 = path.rfind('\\')
            index2 = path.rfind('_')
            index3 = len(path)
            s_input_shape = path[index2+1:index3]
            self.input_shape = int(s_input_shape)
        elif 'RNN' in path or 'LSTM' in path:
            index1 = path.rfind('\\')
            index2 = path.rfind('_')
            index3 = len(path)
            s_vocabulary_size = path[index2+1:index3]
            self.vocabulary_size = int(s_vocabulary_size)
        else:
            index1 = path.rfind("\\")
            index2 = path.rfind(".")
        s_model = path[index1+1:index2]
        return s_model

    def load_best_model(self):
        """
        functions loads best performing model
        """
        l_model_path = set_file_list(self.p_models)
        p_best = l_model_path[0]
        s_model = self.get_model_name(p_best)
        o_best_model = self.load_model(p_best, s_model)
        return o_best_model, p_best

    def train_best_model(self):
        """
        functions trains best performing model
        """
        x, y, text, train_features = self.process_text()
        o_best_model, p_best_model = self.load_best_model()
        s_model = self.get_model_name(p_best_model)
        print(f'Loaded best model {s_model}.')
        k_outer = KFold(n_splits=self.i_splits, random_state=5, shuffle=True)
        print(f'Running training on {s_model}...')
        for i_train, i_eval in k_outer.split(x):
            x_train, x_test = x.iloc[i_train], x.iloc[i_eval]
            y_train, y_test = y.iloc[i_train], y.iloc[i_eval]
            if s_model == 'svm_linear' or s_model == 'svm_non_linear' or s_model == 'log_reg':
                history = o_best_model.fit(x_train, y_train)
            else:
                train = Train(torch.FloatTensor(x_train.to_numpy()), torch.FloatTensor(y_train.to_numpy()))
                i_batch = 64
                train_batch = DataLoader(dataset=train, batch_size=i_batch, shuffle=True)
                self.train(o_best_model, train_batch, s_model)
        print(f'Done training on {s_model}.')
        return o_best_model, p_best_model

    def process_test(self):
        """
        function processes test text
        """
        l_cols_test = ['user handle', 'tweet text', 'time stamp']
        p_test = self.p_corpus + '\\' + 'trump_test' + '.tsv'
        test_data = self.read_files(p_test, l_cols_test, 2)
        test_features = self.extract_features(test_data)
        test = test_features.copy()

        sequences = test.features_group.to_numpy()
        
        text = self.vectorizer.transform(sequences)
        text = pd.DataFrame(text.toarray(), columns=self.vectorizer.get_feature_names())

        for curr_col in test.columns:
            if curr_col not in self.l_test_cols:
                text[curr_col] = test[curr_col]

        test = self.preprocess_text(test)
        text['target'] = test['user handle']
        text[self.l_features] = self.scaler.transform(text[self.l_features])
        x = text.drop(['target'], axis=1)
        y = text['target']

        return x, y, text, test_features

    def predict(self, m, fn):
        """
        functions predicts on best performing model
        :param m model
        :param fn model path
        """
        self.s_model = self.get_model_name(fn)

        print(f'Running test on {self.s_model}.')

        x, y, text, test_features = self.process_test()

        i_run_start = time.time()
        if self.s_model == 'svm_linear' or self.s_model == 'svm_non_linear' or self.s_model == 'log_reg':
            x = x.drop(['features_group'], axis=1)
            y_preds = m.predict(x).reshape(-1, 1)
        else:
            vocab2index = {'': 0, 'UNK': 1}
            text['encoded'] = text['features_group'].apply(lambda x: np.array(self.encode_sentence(x, vocab2index)))
            text = text[text['encoded'].str[1] > 0]
            x_set = text.drop(['features_group', 'target'], axis=1)
            y_set = text['target']
            i_batch = 64
            batch = 1
            x_eval = TextDataset(self.l_features, x_set, y_set)
            x_eval = DataLoader(x_eval, batch_size=i_batch)
            x = x.drop(['features_group'], axis=1)
            test = Test(torch.FloatTensor(x.to_numpy()))
            test_batch = DataLoader(dataset=test, batch_size=batch)
            m.eval()
            y_preds = self.test(m, test_batch)

        i_run_end = time.time()
        run_time = i_run_end - i_run_start
        run_time = float("{:.2f}".format(run_time))
        print(f'Done training and testing {self.s_model}. Completed in: {run_time} seconds.')

        test_acc = accuracy_score(y, y_preds) * 100
        precision = precision_score(y, y_preds) * 100
        recall = recall_score(y, y_preds) * 100
        f1 = f1_score(y, y_preds) * 100
        arr_fpr, arr_tpr, threshold_auc = roc_curve(y, y_preds)  # auc curve
        auc_score = roc_auc_score(y, y_preds)  # auc score
        arr_precision, arr_recall, threshold_pr_auc = precision_recall_curve(y, y_preds)
        pr_auc_score = auc(arr_recall, arr_precision)  # pr_auc score

        test_acc, precision, recall, f1, auc_score, pr_auc_score = float("{:.3f}".format(test_acc)), \
                                                                   float("{:.3f}".format(precision)), \
                                                                   float("{:.3f}".format(recall)), \
                                                                   float("{:.3f}".format(f1)), \
                                                                   float("{:.3f}".format(auc_score)), \
                                                                   float("{:.3f}".format(pr_auc_score))

        print(f'{self.s_model} Best Model Scores: ')
        print(f'Accuracy: {test_acc}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
        print(f'AUC: {auc_score}')
        print(f'PR-AUC: {pr_auc_score}')

        i_trump, i_non_trump = 0, 0
        for ans in y_preds:
            if ans == 0:
                i_trump += 1
            else:
                i_non_trump += 1
        print(f'Tweets Results: \n Trump Posts: {i_trump} \n Non-Trump Posts: {i_non_trump}')

        p_file = self.p_project + r'\203339510.txt'
        with open(p_file, 'w') as write_file:
            for output in y_preds:
                write_file.write(str(int(output)))
                write_file.write(' ')

        print('Done writing results to file.')


class Spell_Checker:
    """
    Task Assumptions:
    (1) Terms contain at most 2 errors.
    (2) Sentences contain at most one term with an error.
    (3) Substitution and transposition are considered as a single error and a single edit.

    The class implements a context sensitive spell checker. The corrections
    are done in the Noisy Channel framework, based on a language model and
    an error distribution model.
    """

    def __init__(self, lm=None):
        """
        Initializing a spell checker object with a language model as an
        instance variable. The language model supports the evaluate()
        and the get_model() functions.

        Args:
            lm: a language model object. Defaults to None
        """
        self.o_curr_lang = lm
        self.l_error_types = ['deletion', 'transposition', 'substitution', 'insertion']
        self.error_tables = self.init_d_errors()  # inits error table
        self.d_error_proba = self.init_d_errors()  # inits error probability table
        self.i_count = collections.Counter  # counts number of words
        self.d_errors = collections.defaultdict(int)  # dict for characters error probability
        self.d_chars = collections.defaultdict(int)  # dict for ranking 1 and 2 characters
     
    def init_d_errors(self):
        """
        Returns: a nested dictionary {str:dict} where str is in:
        <'deletion', 'insertion', 'transposition', 'substitution'> and the
        inner dict {str: int} represents the confusion matrix of the
        specific errors.
        """
        d_errors = dict()
        for error_type in self.l_error_types:
            d_errors[error_type] = collections.defaultdict(float)
        return d_errors

    def set_letter_tf(self, words):
        """
        Returns: updates dict of letter frequencies
        """
        for word in words:  # counts terms generated by 1 or 2 letters
            for i in range(len(word) - 1):
                self.d_chars[word[i:i + 2]] += 1  # 2 chars
                self.d_chars[word[i]] += 1  # 1 char
            self.d_chars[word[-1]] += 1  # last char

    def build_model(self, text, n=3):
        """
        Returns a language model object built on the specified text. The language
        model supports evaluate() and get_model() functions.

        Args:
            text (str): the text to construct the model from.
            n (int): the order of the n-gram model (defaults to 3).

        Returns:
            A language model object
        """
        if n < 1:
            n = 3
            
        o_lang = Ngram_Language_Model(n)  # inits object language model

        self.o_curr_lang = o_lang  # Option #1: Uncomment to generate a new language model from a given text
        # self.o_curr_lang = o_lang.get_model()  # Option #2: Uncomment to load a pre-set language model
        
        filtered_text = normalize_text(text)  # filters to lower-case letters and no punctuations
        o_lang.build_model(filtered_text)  # updates word term frequencies
        filtered_text = re.findall(r'\w+', filtered_text)

        self.d_chars = collections.defaultdict(int)  # dict for character frequencies
        self.i_count = collections.Counter(filtered_text)  # word counter
        self.set_letter_tf(filtered_text)  # updates character frequencies
        
        return o_lang

    @staticmethod
    def set_terms(d_curr_model):
        """
        Function sets word list of the language model
        Args:
            d_curr_model: dict of the current language model object
        """
        l_words = list()
        curr_n_gram = None

        for curr_n_gram, value in d_curr_model.items():
            gram_size = len(curr_n_gram)
            if curr_n_gram is not None and gram_size > 0:
                word = curr_n_gram.split()
                word = word[0]
                for i in range(value):
                    l_words.append(word)

        final_gram = curr_n_gram.split()
        final_gram_size = len(curr_n_gram)

        if final_gram_size > 1:  # sets last gram list
            for word in final_gram[1:]:
                l_words.append(word)

        return l_words

    def get_ngrams(self, text):
        """
        Function uses nltk to generate a language model and for spell checking
        Args:
            text: text to preprocess with normalization (puncs/stopwords/incorrect formats)
        """
        stop_words = set(stopwords.words('english'))
        string.punctuation = string.punctuation + '"' + '"' + '-' + '''+''' + '—'
        l_remove = list(stop_words) + list(string.punctuation) + ['lt', 'rt']
        unigram, bigram, trigram = list(), list(), list()
        tokenized_text = list()
        threshold = 19

        for sentence in text:
            sentence = list(map(lambda x: x.lower(), sentence))
            for term in sentence:
                if term == '.':
                    sentence.remove(term)
                else:
                    unigram.append(term)

            tokenized_text.append(sentence)
            bigram.extend(list(ngrams(sentence, 2, pad_left=True, pad_right=True)))
            trigram.extend(list(ngrams(sentence, 3, pad_left=True, pad_right=True)))

        unigram = self.remove_stopwords(unigram, l_remove)  # filters stopwords
        bigram = self.remove_stopwords(bigram, l_remove)
        trigram = self.remove_stopwords(trigram, l_remove)

        freq_bi = FreqDist(bigram)  # updates frequencies
        freq_tri = FreqDist(trigram)
        d_count = defaultdict(collections.Counter)
        d_count = collections.defaultdict()
        d_count = dict()
        for i, j, k in freq_tri:
            if i is not None and i != ' ' \
                    and j is not None and j != ' ' \
                    and k is not None and k != ' ':
                try:
                    d_count[i, j] += freq_tri[i, j, k]
                except KeyError as ke:
                    continue

        prefix = "he", "said"  # predict next term
        cand = " ".join(prefix)
        for i in range(threshold):
            suffix = self.predict(d_count[prefix])
            cand = cand + ' ' + suffix
            # print(cand)
            prefix = prefix[1], suffix

    def add_language_model(self, lm):
        """
        Adds the specified language model as an instance variable.
        (Replaces an older LM dictionary if set)

        Args:
            lm: a language model object
        """
        d_curr_model = lm.get_model()  # loads desired language model
        l_words = self.set_terms(d_curr_model)  # creates a word list
        self.d_chars = collections.defaultdict(int)  # updates character frequencies
        self.i_count = collections.Counter(l_words)  # counts terms
        self.set_letter_tf(l_words)  # updates character frequencies
        self.o_curr_lang = lm  # sets the language model

    @staticmethod
    def set_candidates(term):
        """
        Function returns corrected candidates per selected term
        :param term: current term
        """
        split = [(term[:i], term[i:]) for i in range(len(term) + 1)]
        delete = [L + R[1:] for L, R in split if R]
        transpose = [L + R[1] + R[0] + R[2:] for L, R in split if len(R) > 1]
        replace = [L + chr(c) + R[1:] for L, R in split if R for c in range(97, 123)]
        insert = [L + chr(c) + R for L, R in split for c in range(97, 123)]
        set_error_types = set(delete + transpose + replace + insert)
        return set_error_types

    def filter_candidates(self, d_candidates):
        """
        Function filters candidates and returns their set
        :param d_candidates: dictionary of candidates
        """
        vocab = self.i_count.keys()
        cands = [c for c in d_candidates if c in vocab]
        set_cands = set(cands)
        return set_cands

    def get_errors(self, term_error, d_term_correct):
        """
        Finds errors in a given text.
        
        Args:
            term_error: terms with error
            d_term_correct: dict of terms with error and their correction
        """
        for curr_error in term_error:  # updates error count and probabilities
            d_cands = self.set_candidates(curr_error)
            correct = d_term_correct[curr_error]
            cands_correct = self.set_candidates(correct)
            curr_func = None
            if curr_error in cands_correct:  # checks error type
                if len(curr_error) == len(correct):
                    curr_func = self.set_sub
                if len(curr_error) < len(correct):
                    curr_func = self.set_del
                if len(curr_error) > len(correct):
                    curr_func = self.set_insert
                elif sorted(curr_error) == sorted(correct):
                    curr_func = self.set_transpose
                curr_func(curr_error, correct)

    def learn_error_tables(self, errors_file):
        """
        Returns: a nested dictionary {str:dict} where str is in:
        <'deletion', 'insertion', 'transposition', 'substitution'> and the
        inner dict {str: int} represents the confusion matrix of the
        specific errors, where str is a string of two characters matching the
        row and column "indices" in the relevant confusion matrix and the int is the
        observed count of such an error (computed from the specified errors file).
        Examples of such string are 'xy', for deletion of a 'y'
        after an 'x', insertion of a 'y' after an 'x'  and substitution
        of 'x' (incorrect) by a 'y'; and example of a transposition is 'xy' indicates the characters that are transposed.

        Notes:
            1. Ultimately, one can use only 'deletion' and 'insertion' and have
                'substitution' and 'transposition' derived. Again,  we use all
                four types explicitly in order to keep things simple.
        Args:
            errors_file (str): full path to the errors file. File format, TSV:
                                <error>    <correct>

        Returns:
            A dictionary of confusion "matrices" by error type (dict).
        """
        errors_file = open(errors_file, "r")  # loads the error file
        
        for table_name in self.error_tables.keys():  # sets a dictionary of confusion "matrices" by error type
            self.error_tables[table_name] = collections.defaultdict(int)
            
        d_term_correct = dict()  # updates dict: <key=term with error, value=correct term> #
        for line in errors_file:
            line = line.strip().replace('\n', '')
            term_correct = re.split('\t', line)
            d_term_correct[term_correct[0]] = term_correct[1]
        term_error = d_term_correct.keys()
        
        self.d_errors = collections.defaultdict(int)  # dictionary of error counts
        self.get_errors(term_error, d_term_correct)  # searches for errors
        self.set_errors(term_error)  # updates errors
        for error_type in self.l_error_types:  # updates error probability by its type
            self.error_proba(error_type)
        return self.error_tables  # returns dict of confusion "matrices" of errors

    def add_error_tables(self, error_tables):
        """
        Adds the specified dictionary of error tables as an instance variable.
        (Replaces an older value dictionary if set)

        Args:
            error_tables (dict): a dictionary of error tables in the format
            returned by learn_error_tables()
        """
        self.error_tables = error_tables
        for error_type in self.l_error_types:
            self.chars_error_proba(error_type)

    def set_errors(self, words):
        """
        Returns: updated error count of character terms
        """
        for word in words:
            word_size = len(word)
            for i in range(word_size - 1):
                self.d_errors[word[i]] += 1
                self.d_errors[word[i:i + 2]] += 1
            self.d_errors[word[-1]] += 1

    def evaluate(self, text):
        """
        Returns the log-likelihood of the specified text given the language
        model in use. Smoothing is applied on texts containing OOV words

       Args:
           text (str): Text to evaluate.

       Returns:
           Float. The float should reflect the (log) probability.
        """
        result = self.o_curr_lang.evaluate(text)
        return result

    def spell_check(self, text, alpha):
        """
        Returns the most probable fix for the specified text. Use a simple
        noisy channel model is the number of tokens in the specified text is
        smaller than the length (n) of the language model.

        Args:
            text (str): the text to spell check.
            alpha (float): the probability of keeping a lexical word as is.

        Return:
            A modified string (or a copy of the original if no corrections are made.)
        """
        s_period = '.'
        if not text.endswith(s_period):
            text += s_period

        text = normalize_text(text)

        l_sentences = self.parse_text(text)
        
        i_log = math.log(alpha)
        i_log1 = math.log(1 - alpha)
        l_candidates = list()
        l_clean = list()

        for curr_sentence in l_sentences:  # loops over sentences
            cand_sentence = curr_sentence
            prev_result = i_log + self.o_curr_lang.evaluate(curr_sentence)
            l_words = re.findall(r'\w+', curr_sentence)

            for i_term, term in enumerate(l_words):  # loops over words
                d_sentences = dict()  # candidate sentences for each term
                d_candidates = self.set_candidates(term)  # possible candidates for the error type
                d_filtered_cands = self.filter_candidates(d_candidates)
                f_proba = 0

                for candidate in d_candidates:  # removes terms not from the vocabulary
                    if l_words[i_term] == candidate or candidate == '':
                        continue

                    error_func, error_type = self.detect_error_format(term, candidate)  # detects mistake type
                    curr_error = error_func(term, candidate)
                    curr_error_prob = self.d_error_proba[error_type][curr_error]

                    if curr_error_prob == 0:  # candidates selection
                        continue  # if prob=0 don't correct
                    elif candidate in d_filtered_cands:
                        f_proba += curr_error_prob
                        l_next_sentence = l_words.copy()
                        l_next_sentence[i_term] = candidate
                        s_next_sentence = ' '.join(l_next_sentence)

                        if s_next_sentence in d_sentences:  # updates probability
                            d_sentences[s_next_sentence] += curr_error_prob
                        else:
                            d_sentences[s_next_sentence] = curr_error_prob

                    cands_correct = self.set_candidates(candidate)  # removes terms not from the vocabulary
                    cands_correct = self.filter_candidates(cands_correct)

                    for curr_term in cands_correct:  # detects up to 2 mistake types
                        if l_words[i_term] == curr_term:
                            continue

                        error_func, error_type = self.detect_error_format(candidate, curr_term)
                        curr_error = error_func(candidate, curr_term)
                        score = self.d_error_proba[error_type][curr_error]
                        score *= curr_error_prob

                        if score == 0:
                            continue
                        else:
                            f_proba += score

                        l_next_sentence = l_words.copy()  # candidate selection
                        l_next_sentence[i_term] = curr_term
                        s_next_sentence = ' '.join(l_next_sentence)

                        if s_next_sentence in d_sentences:  # calculates the probability of the errors
                            d_sentences[s_next_sentence] += score
                        else:
                            d_sentences[s_next_sentence] = score

                dom = 1 - alpha
                f_proba /= dom  # updates term probability score

                for curr_sentence, curr_prob in d_sentences.items():  # updates result if the new one is higher
                    result = self.evaluate(curr_sentence)
                    new_result = i_log1 + math.log(curr_prob/f_proba) + result
                    if new_result > prev_result:
                        prev_result = new_result
                        cand_sentence = curr_sentence + ' ' + s_period

            l_clean.append(cand_sentence)
        l_results = self.del_tokens(l_clean)  # checks again for invalid formats
        return l_results

    @staticmethod
    def replace_tokens(text):
        """
        Function replaces invalid characters with valid ones
        :param text: text to parse
        :return: filtered text
        """
        d_rep_tokens = {
            "\n": " ",
            "?": ["?\"", "\"?"],
            "!": ["!\"", "\"!"],
            "”": [".”", "”."],
            "\"": [".\"", "\"."],
            "e.g.": "e<period>g<period>",
            "i.e.": "i<period>e<period>",
            "Ph.D": "Ph<period>D<period>",
            "..": "<period><period>",
            "...": "<period><period><period>",
            ".": ".<stop>",
            "?": "?<stop>",
            "!": "!<stop>",
            "<period>": "."
        }
        
        text = " " + text + "  "
        for bad_token, correct_token in d_rep_tokens.items():
            if bad_token in text:
                if isinstance(correct_token, list):
                    bad_token = correct_token[0]
                    correct_token = correct_token[1]
                text = text.replace(bad_token, correct_token)
            
        return text
    
    @staticmethod
    def sub_tokens(text):
        """
        Function substitutes invalid characters with valid ones
        :param text: text to parse
        :return: filtered text
        """
        digits = "([0-9])"
        digits1 = digits + "[.]" + digits
        
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        suffix1 = " " + suffixes + "[.]"
        
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        
        abbr = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        abbr0 = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
        abbr1 = acronyms + " " + abbr
        abbr2 = " " + suffixes + "[.] " + abbr

        alphabets = "([A-Za-z])"
        alphabets0 = " " + alphabets + "[.]"
        alphabets1 = "\s" + alphabets + "[.] "
        alphabets2 = alphabets + "[.]" + alphabets + "[.]"
        alphabets3 = alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]"

        urls = "[.](com|net|org|io|gov|me|edu)"

        d_sub_tokens = {
            digits1: "\\1<period>\\2",
            abbr0: "\\1<period>",
            abbr1: "\\1<stop> \\2",
            abbr2: " \\1<stop> \\2",
            alphabets0: " \\1<period>",
            alphabets1: " \\1<period> ",
            alphabets2: "\\1<period>\\2<period>",
            alphabets3: "\\1<period>\\2<period>\\3<period>",
            suffix1: " \\1<period>",
            urls: "<period>\\1",
        }
        
        for bad_token, correct_token in d_sub_tokens.items():
            text = re.sub(bad_token, correct_token, text)
            
        return text
    
    @staticmethod
    def split_tokens(text):
        """
        Function splits text to sentences by delimiters
        :param text: text to split
        :return: filtered sentences
        """
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

    @staticmethod
    def del_tokens(l_clean):
        """
        Function post-processes the outputs
        Args:
            l_clean: list of terms to filter
        """
        l_clean_filtered = list()
        d_puncwords = set_puncwords()
        for term in l_clean:
            if term not in d_puncwords:
                l_clean_filtered.append(term)
        l_clean_filtered = '\n'.join(l_clean_filtered)
        return l_clean_filtered

    def parse_text(self, text):
        """
        Function receives text as input and outputs filtered sentence without invalid formats.
        :param text: normalized text
        :return: parsed and filtered sentences
        """
        text = self.replace_tokens(text)
        text = self.sub_tokens(text)
        text = self.split_tokens(text)
        return text

    def tokenize(self, text):
        """
        Function tokenizes text with nltk
        Args:
            text: text input
        """
        tokenized_text = [list(map(str.lower, nltk.word_tokenize(sent)))
                          for sent in nltk.sent_tokenize(text)]
        try:
            nltk.word_tokenize(nltk.sent_tokenize(text)[0])
        except Exception as e:
            # print(e)
            sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
            toktok = ToktokTokenizer()
            word_tokenize = toktok.tokenize

        n = 3
        train_data, padded_sents = nltk.padded_everygram_pipeline(n, tokenized_text)
        model = MLE(n)  # builds model
        model.fit(train_data, padded_sents)  # training
        # print(model.vocab)

    def predict(self, counter):
        """
        Function uses nltk to predict upcoming candidates.
        Args:
            counter: term count
        """
        try:
            cand = random.choice(list(counter.elements()))  # gets candidates
        except KeyError as ke:
            # print(ke)
            cand = ''
        return cand

    def remove_stopwords(self, curr_model, l_remove):
        """
        Function removes stopwords from text
        Args:
            curr_model: unigram/bigram/trigram type ngram model
            l_remove: stopwords
        """
        l_new = []
        for pair in curr_model:
            count = 0
            for word in pair:
                if word in l_remove:
                    count = count or 0
                else:
                    count = count or 1
            if count == 1:
                l_new.append(pair)
        return l_new

    def detect_error_format(self, error, correct):
        """
        Function detects and returns the proper error type
        :param error: term with an error
        :param correct: optional term correction
        :return: error format
        """
        if len(error) == len(correct):
            return self.detect_substitution, 'substitution'
        if len(error) < len(correct):
            return self.detect_deletion, 'deletion'
        if len(error) > len(correct):
            return self.detect_insertion, 'insertion'
        elif sorted(error) == sorted(correct):
            return self.detect_transposition, 'transposition'
        
    @staticmethod
    def detect_substitution(error, correct):
        """
        Function detects and returns substitution of multiple characters
        :param error: term with an error
        :param correct: optional candidate for term correction
        :return: character replacement
        """
        i_error = len(error)
        for i in range(i_error):
            if correct[i] != error[i]:
                return error[i] + correct[i]

    @staticmethod
    def detect_deletion(error, correct):
        """
        Function detects and returns characters that need to be deleted
        :param error: term with an error
        :param correct: optional candidate for term correction
        :return: character deletion
        """
        i_error = len(error) - 1
        i_candidate = len(correct)
        for i in range(i_candidate):
            if i != i_error:
                if error[i] is not correct[i]:
                    if i == 0:
                        return correct[i]  # first character
                    else:
                        return correct[i - 1] + correct[i]  # previous characters
            else:
                return correct[i - 1] + correct[i]

    @staticmethod
    def detect_insertion(error, correct):
        """
        Function detects and returns characters that need to be added to the term
        :param error: term with an error
        :param correct: optional candidate for term correction
        :return: character insertion
        """
        i_error = len(error)
        i_candidate = len(correct)
        for i in range(i_error):
            if i >= i_candidate - 1:
                return correct[i - 1] + error[i]
            elif i_candidate == 0:
                return correct[i]
            else:
                if error[i] is not correct[i]:
                    if i != 0:
                        return correct[i - 1] + error[i]                 
                    else:
                        return correct[i]

    def detect_transposition(self, error, correct):
        """
        Function detects and returns transposition of characters,
        characters are checked by substitution of transposition isn't detected.
        :param error: term with an error
        :param correct: optional candidate for term correction
        :return: character transposition/substitution
        """
        i_error = len(error) - 1
        for i in range(i_error):
            if correct[i + 1] == error[i] and correct[i] == error[i + 1]:
                return correct[i] + correct[i + 1]     
        return self.detect_substitution(error, correct)

    def error_proba(self, error):
        """
        (1) Generates mistake type probabilities for all types of errors.
        (2) Score is normalized by corpus terms size.
        (3) Applies smoothing if the correction is not found.
        """
        if error == 'deletion':
            error_values = self.error_tables['deletion'].keys()
            for name in error_values:
                correction_keys = self.d_errors.keys()
                i_correct = self.d_errors[name]
                if i_correct == 0:
                    i_correct = len(correction_keys)
                curr_error_table = self.error_tables['deletion'][name]
                self.d_error_proba['deletion'][name] = curr_error_table / i_correct
        else:
            error_values = self.error_tables[error].keys()
            new = ''
            for name in error_values:
                if error == 'transposition':
                    new = name
                if error == 'insertion':
                    new = name[0]
                if error == 'substitution':
                    new = name[1]
                    
                l_correct = self.d_errors.keys()
                i_correct = self.d_errors[new]
                if i_correct == 0:
                    i_correct = len(l_correct)
                    
                curr_error_table = self.error_tables[error][name]
                self.d_error_proba[error][name] = curr_error_table / i_correct

    def chars_error_proba(self, error):
        """
        (1) Generates mistake type probabilities for all types of errors.
        (2) Score is normalized by the count of characters by the given text.
        (3) Applies smoothing if the correction is not found.
        """
        if error == 'deletion':
            error_values = self.error_tables['deletion'].keys()
            for name in error_values:
                l_correct = self.d_chars.keys()
                i_correct = self.d_chars[name]
                if i_correct == 0:
                    i_correct = len(l_correct)
                curr_error_table = self.error_tables['deletion'][name]
                self.d_error_proba['deletion'][name] = curr_error_table / i_correct      
        else:
            error_values = self.error_tables[error].keys()
            for name in error_values:
                l_correct = self.d_chars.keys()
                i_correct = self.d_chars[name]
                if i_correct == 0:
                    i_correct = len(l_correct)
                curr_error_table = self.error_tables[error][name]
                self.d_error_proba[error][name] = curr_error_table / i_correct
                
    def set_sub(self, error, correct):
        """
        Function adds substitution mistake types to error types dictionary
        :param error: term with an error
        :param correct: optional candidate for term correction
        :return: updated error types dictionary
        """
        i_error = len(error)
        for i in range(i_error):
            if error[i] != correct[i]:
                curr_value = error[i] + correct[i]
                self.error_tables['substitution'][curr_value] += 1
                new = error[i] + correct[i]
                return new

    def set_del(self, error, correct):
        """
        Function adds deletion mistake types to error types dictionary
        :param error: term with an error
        :param correct: optional candidate for term correction
        :return: updated error types dictionary
        """
        i_error = len(error)
        i_correct = len(correct)
        for i in range(i_correct):
            if i > i_error - 1:
                self.error_tables['deletion'][correct[i] + error[i - 1]] += 1
                return correct[i] + error[i - 1]
            else:
                if error[i] is not correct[i]:
                    if i > 0:
                        self.error_tables['deletion'][correct[i] + error[i - 1]] += 1
                        return correct[i] + error[i - 1]
                    else:
                        self.error_tables['deletion'][correct[i]] += 1
                        return correct[i]
                    
    def set_insert(self, error, correct):
        """
        Function adds insertion mistake types to error types dictionary
        :param error: term with an error
        :param correct: optional candidate for term correction
        :return: updated error types dictionary
        """
        for i in range(len(error)):
            if i >= len(correct)-1:
                self.error_tables['insertion'][error[i] + error[i - 1]] += 1
                return error[i] + error[i - 1]
            else:
                if error[i] is not correct[i]:
                    if i > 0:
                        self.error_tables['insertion'][error[i] + error[i - 1]] += 1
                        return error[i] + error[i - 1]
                    else:
                        self.error_tables['insertion'][error[i]] += 1
                        return error[i]
                    break

    def set_transpose(self, error, correct):
        """
        Function adds transpose mistake types to error types dictionary or substitution error type
        :param error: term with an error
        :param correct: optional candidate for term correction
        :return: updated error types dictionary
        """
        i_error = len(error) - 1
        for i in range(i_error):
            if error[i] == correct[i + 1] and error[i + 1] == correct[i]:
                self.error_tables['transposition'][error[i] + error[i + 1]] += 1
                return error[i] + error[i + 1] + correct[i] + correct[i + 1]
        self.set_sub(error, correct)


class Ngram_Language_Model:
    """
    The class implements a Markov Language Model that learns a language model from a given text.
    It supports language generation and the evaluation of a given string.
    The class can be applied on both word level and character level.
    """
    def __init__(self, n=3, b_chars_only=False):
        """
        Initializing a language model object.

        Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            b_chars_only (bool): True iff the model consists of ngrams of characters rather then word tokens.
                          Defaults to False
        """
        if n < 1:
            n = 3
        self.n = n
        self.corpus_size = 0
        self.corpus_size_filtered = 0
        self.i_tokens = 0
        self.b_chars_only = b_chars_only
        # filtering dictionary for normalizing the text
        self.d_puncwords = set_puncwords()  # filtering irrelevant formats
        self.d_stopwords = set_stopwords()  # prior penalty - v2
        # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
        self.model_dict = defaultdict(int)
        # a dictionary of the form {term:frequency}, holding counts of all terms in the specified text.
        self.model_dict_tf = defaultdict(int)
        self.defaultdict = collections.defaultdict(int)
        self.l_ngrams = list()

    def build_model(self, text):
        """
        Populates the instance variable model_dict.

        Args:
            text (str): the text to construct the model from.
        """

        l_words = text.split()
        self.corpus_size = len(l_words)

        # text_filtered = re.compile("[\t\r\/\.\,\'-]").sub(" ", text).split()
        # l_words = re.sub("[^\w]", " ", text).split()
        # regex = re.compile('[%s]' % re.escape(string.punctuation))
        # text2 = regex.sub('', text)
        # l_words2 = text.split()

        for term in l_words:  # (1) extracts tokens for terms
            if term not in self.d_puncwords:
                term = self.clean_format(term)  # fixes bad character formats
                if term != '':
                    if term not in self.model_dict_tf:  # term frequency update
                        self.model_dict_tf[term] = 1
                    else:
                        self.model_dict_tf[term] += 1

        # text_filtered = ' '.join(word.strip(string.punctuation) for word in text.split())
        # l_words_filtered = text_filtered.split()
        # self.corpus_size_filtered = len(l_words_filtered)
                        
        if self.b_chars_only:  # (2) extracts tokens for characters
            limit = len(text) - self.n + 1
            self.l_ngrams = [text[i_token:i_token + self.n] for i_token in range(limit)]
        else:  # creates n-gram list
            tokens = [token for token in text.split(' ') if token != '']
            ngrams = zip(*[tokens[i_token:] for i_token in range(self.n)])
            self.l_ngrams = [' '.join(ngram) for ngram in ngrams]

        self.i_tokens = len(text)

        for gram in self.l_ngrams:
            self.defaultdict[gram] += 1

    def get_model_dictionary(self):
        """
        Returns the dictionary class object
        """
        return self.model_dict
    
    def get_model(self):
        """
        Returns the dictionary class <key=ngram, value=count>
        """
        d_curr = collections.defaultdict(int)
        for curr_term in self.l_ngrams:
            d_curr[curr_term] += 1
        return d_curr

    def get_model_window_size(self):
        """
        Returning the size of the curr_context window (the n in "n-gram")
        """
        return self.n

    def select_candidates(self, text, curr_context, size, n):
        s_del = '[ \t\n\r\f\v]'  # white-space
        while size < n:  # text generation ends before the n'th word
            l_prefix = list()
            l_prefix_weights = list()
            curr_n_grams = self.defaultdict.keys()
            prefix = ''

            if not self.b_chars_only:  # (1) terms
                l_words = re.split(s_del, curr_context)
                for i, word in enumerate(l_words):
                    length = len(l_words) - self.n + 1
                    if i < length:
                        continue
                    prefix += word + ' '
                prefix = prefix[:-1]
            else:  # (2) characters
                length_context = len(curr_context) - self.n + 1
                prefix = curr_context[length_context:]

            for n_gram in curr_n_grams:  # detects n-grams from seed context
                if n_gram.startswith(prefix):
                    l_prefix_weights.append(self.defaultdict[n_gram])
                    l_prefix.append(n_gram)

            if len(l_prefix) > 0:
                selected_cand = random.choices(l_prefix, l_prefix_weights, k=1)
                cand = selected_cand[0]
            else:  # if no ngram candidates were found, they are chosen randomly by uniform distribution
                cand = random.choice(list(curr_n_grams))

            if self.b_chars_only:  # update text size and context by last n-gram
                new = cand[-1]
                text += new
                size = len(text)
            else:
                new = re.split(s_del, cand)[-1]
                text += ' ' + new
                l_terms = re.split(s_del, text)
                size = len(l_terms)

            curr_context = cand

        return text

    def generate(self, curr_context=None, n=20):
        """
        Returns a string of the specified length, generated by applying the language model
        to the specified prefix curr_context. If no curr_context is specified the curr_context should be sampled
        from the models' curr_contexts distribution. Generation should stop before the n'th word if the
        curr_contexts are exhausted. If the length of the specified curr_context exceeds (or equal to)
        the specified n, the method should return the a prefix of length n of the specified curr_context.

        Args:
            curr_context (str): a prefix curr_context to start the generated string from. Defaults to None
            n (int): the length of the string to be generated.

        Return:
            String. The generated text.
        """
        if curr_context is None:  # loads seed context, which are the input prefixes from the dictionary
            d_vocab = collections.defaultdict(int)
            for curr_n_gram in self.l_ngrams:
                d_vocab[curr_n_gram] += 1
            curr_context = random.choice(d_vocab)

        text = ''
        text += curr_context
        s_del = '[ \t\n\r\f\v]'  # white-space

        if self.b_chars_only:
            size = len(curr_context)
        else:
            size = len(re.split(s_del, curr_context))  # string to list by white-space delimiter

        text = self.select_candidates(text, curr_context, size, n)

        return text

    def get_tf(self, sub):
        """
        Returns the term frequency of the n-gram that begins with prefix of a given window input and applies smoothing

        Args:
            sub (str): the input sub-string

        Return:
            term frequency
        """
        curr_n_grams = self.defaultdict.keys()
        count = 0
        reward = 1
        const = 0.1

        for n_gram in curr_n_grams:
            if n_gram.startswith(sub):
                count += self.defaultdict[n_gram]

                # if not self.b_chars_only:
                    # if n_gram in self.d_stopwords:  # tf with penalty v2
                    #     reward *= const
                    # count += reward

        if count == 0:
            count = self.smooth(sub)  # applies laplace

        return count

    @staticmethod
    def split_text_by_space(text):
        """
        Splits text by white space delimiter
        """
        s_del = '[ \t\n\r\f\v]'
        return re.split(s_del, text)

    def n_proba(self, tokens):
        """
        Returns normalized term frequency of split terms

        Args:
           tokens (list): Text to evaluate.

        Returns:
           Probability.
        """
        tf_proba = 1
        s_del = '[ \t\n\r\f\v]'

        for i in range(self.n - 1):  # term count
            sub = ''
            for j in range(i + 1):
                if len(tokens) > 1 and tokens[j] not in self.d_puncwords:
                    sub += tokens[j] + ' '
            term_freq = self.get_tf(sub[:-1])
            split_text = self.split_text_by_space(sub)
            curr_size = len(split_text)

            if curr_size > 1:  # counts term frequencies
                prev_gram = ''
                for curr_word in re.split(s_del, sub)[:-1]:
                    prev_gram += curr_word + ' '
                i_size = self.get_tf(prev_gram[:-1])
            else:
                i_size = len(self.l_ngrams)

            value = term_freq / i_size
            tf_proba *= value  # normalized score

        return tf_proba

    def last_n_proba(self, tokens, tf_proba):
        """
        Returns probabilities for terms per n-gram window

        Args:
           tokens (list): Text to evaluate.
           tf_proba (int): Probabilty value to be returned

        Returns:
           Log Probability.
        """
        tf_log = 0
        s_del = '[ \t\n\r\f\v]'
        length = len(tokens) - self.n + 1
        for i in range(length):  # sequence by windows size to get the probability
            limit = i + self.n - 1
            sequence = ''
            for j in range(i, limit):
                if len(tokens) > 1 and tokens[j] not in self.d_puncwords:
                    sequence += tokens[j]
                    sequence += ' '
            sequence = sequence[:-1]
            i_next = i + self.n - 1
            next_term = tokens[i_next]

            curr_n_grams = self.defaultdict.keys()
            seq_count, seq_count_next = 0, 0
            # const = 0.9
            # const = 0.5
            const = 0.1
            # reward = 2
            reward = 10
            # reward = 100

            for n_gram in curr_n_grams:  # aggregates the n-grams for the seed
                if n_gram.startswith(sequence):
                    found = self.defaultdict[n_gram]
                    seq_count += found
                    s_split = re.split(s_del, n_gram)[-1]
                    if next_term == s_split:
                        found_next = self.defaultdict[n_gram]
                        seq_count_next += found_next
                # if not self.b_chars_only:
                #     if n_gram in self.d_stopwords:
                #         reward *= const
            if seq_count_next > 0:
                ngram_proba = seq_count_next / seq_count
            else:  # if new term, we apply laplace smoothing
                s_smooth = sequence + ' ' + next_term
                ngram_proba = self.smooth(s_smooth)

            tf_proba *= ngram_proba  # calculates the probability score
            # tf_proba2 = tf_proba * ngram_proba + reward
            tf_log = math.log(tf_proba)
            # tf_log = math.log(tf_proba2)

        return tf_log

    def evaluate(self, text):
        """
        Returns the log-likelihood of the specified text to be a product of the model.
        Laplace smoothing should be applied if necessary.

        Args:
           text (str): Text to evaluate.

        Returns:
           Float. The float should reflect the (log) probability.
        """
        s_del = '[ \t\n\r\f\v]'
        tokens = re.split(s_del, text)

        tf_proba = self.n_proba(tokens)

        tf_log = self.last_n_proba(tokens, tf_proba)

        return tf_log

    def smooth(self, ngram):
        """
        Returns the smoothed (Laplace) probability of the specified ngram.

        Args:
            ngram (str): the ngram to have it's probability smoothed

        Returns:
            float. The smoothed probability.
        """
        df_model = pd.DataFrame(self.model_dict_tf.items(), columns=['term', 'freq'])
        df_non_unique = df_model[df_model['freq'] > 1]  # dataframe of non-unique terms
        df_unique = df_model[df_model['freq'] == 1]  # dataframe of unique terms
        i_non_unique_terms = df_model.shape[0] - df_model[df_model['freq'] > 1].shape[0]
        
        # print(f'Number of Non-Unique Terms: {i_non_unique_terms}')
        # df_non_unique['freq'] += 1  # non-unique terms are added with 1
        
        df_non_unique.loc[:, 'freq'] += 1
        df_laplace = df_non_unique.copy()
        df_laplace = df_laplace.append(df_unique, ignore_index=False)
        nom = df_laplace['freq']
        v = len(self.model_dict_tf)
        n = self.corpus_size
        dom = n + v
        laplace1 = nom / dom  # calculates laplace formula

        df_laplace['freq'] = laplace1  # updates the dataframe
        pd.options.display.float_format = '{:.3f}'.format
        curr_n_grams = self.defaultdict.keys()
        s_del = '[ \t\n\r\f\v]'

        ngram = re.split(s_del, ngram)
        i_ngram = len(ngram) - 1
        sequence = ''
        for i in range(i_ngram):  # calculates by a window of n-1 terms
            sequence += ngram[i] + ' '
        sequence = sequence[:-2]
        seq_count = 0
        seq_count_next = 0
        next_term = ngram[-1]

        for n_gram in curr_n_grams:  # loops over n-grams
            if n_gram.startswith(sequence):
                seq_count += self.defaultdict[n_gram]
                if re.split(s_del, n_gram)[-1] == next_term:
                    seq_count_next += self.defaultdict[n_gram]

        nom = seq_count_next
        dom = seq_count
        n = len(self.l_ngrams)
        v = len(self.defaultdict)
        laplace2 = (nom + 1) / (dom + v)  # calculates laplace formula v2

        return laplace2

    def clean_format(self, term):
        """
        Returns a filtered term without invalid characters

        Args:
            term (str): the term to be filtered

        Returns:
            filtered term
        """
        new_term = ''
        for i in range(len(term)):
            char = term[i]
            if char not in self.d_puncwords:  # neglects invalid formats
                new_term = new_term + char
        return new_term


def set_stopwords():
    """
    Returns a dictionary of stopwords
    """
    p_project = os.path.dirname(os.path.dirname(__file__))
    p_project = p_project + r'\NLP_lang-model'
    p_resources = set_dir(p_project, 'resources')
    p_stopwords = p_resources + r'\stopwords.txt'
    d_sw = dict()
    l_sw = list()

    # with open(p_stopwords, 'r') as file:  # disabled - no attachments allowed for other resource files.
    #     data = file.read().replace('\n', ' ')
    # l_sw = data.split()

    l_sw = ['a', "a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after',
            'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already',
            'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow',
            'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate',
            'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away',
            'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand',
            'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both',
            'brief', 'but', 'by', 'c', "c'mon", "c's", 'came','can', "can't", 'cannot', 'cant', 'cause', 'causes',
            'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently',
            'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't",
            'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do',
            'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg',
            'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever',
            'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f',
            'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly',
            'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go',
            'goes', 'going', 'gone', 'got','gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly', 'has',
            "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here', "here's",
            'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither',
            'hopefully', 'how', 'howbeit', 'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored',
            'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar',
            'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just',
            'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter',
            'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking',
            'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more',
            'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly',
            'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine','no', 'nobody',
            'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously',
            'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other',
            'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p',
            'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably',
            'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably',
            'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw',
            'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems',
            'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she',
            'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something',
            'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying',
            'still', 'sub', 'such', 'sup', 'sure', 't', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank',
            'thanks', 'thanx', 'that', "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
            'thence', 'there', "there's", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon',
            'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough',
            'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too',
            'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under',
            'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses',
            'using', 'usually', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was',
            "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't",
            'what', "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas',
            'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's",
            'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't",
            'wonder', 'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're",
            "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']

    # for word in l_sw:
    #     d_sw[word] = ""
    # del d_sw
    return l_sw


def set_puncwords():
    """
    Returns a dictionary of punctuations
    """
    d_puncwords = {}
    l_punc = [' ', '', "\"", '\"', "\\", '\\\\', ',', '"', '|' '?', '-', '--', '_', '*', '"', '`', ':', '.', '/',
              ';', "'", '[', ']', '(', ')', '{', "}", '<', '>', '~', '^', '?', '&', '!', "=", '+', "#"]
    for word in l_punc:
        d_puncwords[word] = ""
    del l_punc
    return d_puncwords


def set_dir(path, folder_name):
    """
    Creates a directory

    Args:
        path (str): string of full parent path
        folder_name (str): string of child directory

    Returns:
        p_new_dir (str): creates the directory and returns its path
    """
    p_new_dir = path + '\\' + folder_name
    if not os.path.exists(p_new_dir):
        os.makedirs(p_new_dir)
        return p_new_dir
    return p_new_dir


def set_file_list(path):
    """
    Loads a list of file paths for a given string directory

    Args:
        path (str): string of full parent path

    Returns:
        l_files (list): list of file paths
    """
    l_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            curr_file_path = os.path.join(root, file)
            l_files.append(curr_file_path)
    return l_files


def lower_case(text):
    """
    Returns a lower case string

        Args:
        text (str): string input
    """
    return text.lower()


def filter_punctuations(text):
    """
    Returns string without punctuations

        Args:
        text (str): string input
    """
    text_filtered = ' '.join(word.strip(string.punctuation) for word in text.split())
    # text_filtered = text.translate(str.maketrans('', '', text.punctuation))
    l_words_filtered = text_filtered.split()
    text_no_puncs = re.sub('([!"#$%&\'()*+, -./:;<=>?@[\]^_`{|}~])', r' \1 ', text)
    text_no_puncs = re.sub('\s{2,}', ' ', text)
    return text_no_puncs


def normalize_text(text):
    """
    Returns a normalized version of the specified string.
    You can add default parameters as you like (they should have default values!)

    Args:
        text (str): the text to normalize

    Returns:
        string. the normalized text.
    """
    text = filter_punctuations(text)
    text = lower_case(text)
    return text


def load_corpus():
    """
    Returns:
        text (str): full text of all corpuses
    """
    p_project = os.path.dirname(os.path.dirname(__file__))
    p_project = p_project + r'\NLP_lang-model'
    p_corpus = set_dir(p_project, 'corpus')
    l_corpus = set_file_list(p_corpus)
    text = read_corpus(l_corpus)

    # sample
    s_sample = 'A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat on the rat that sat on the cat on the mat.'
    # text += s_sample
    text = s_sample

    return text


def read_corpus(p_curr_corpus):
    """
    Reads corpuses to the 'text' variable
    Corpus used: Norvig, Trump Tweets, Obama Speeches and William Shakespeare literature.
    Args:
        p_curr_corpus (str or list): file paths

    Returns:
        text (str): full text of all corpuses
    """
    text = ''
    if isinstance(p_curr_corpus, str):  # loads one corpus
        with open(p_curr_corpus, 'r', encoding='utf8') as file:
            text = file.read().replace('\n', ' ')
    else:
        for curr_corpus in p_curr_corpus:  # loads multiple corpuses
            with open(curr_corpus, 'r', encoding='utf8') as file:
                text += file.read().replace('\n', ' ')
    return text
    

if __name__ == '__main__':
    TC = Text_Classification()
    i_run_start = time.time()

    text_train = TC.load_text('trump_train')
    text_test = TC.load_text('trump_test')

    text_train = TC.preprocess_text(text_train)
    text_test = TC.preprocess_text(text_test)

    text_train = TC.parse_text(text_train)
    text_test = TC.parse_text(text_test)

    TC.analysis_text(text_train)

    text_train = TC.set_train_features(text_train)
    text_test = TC.set_test_features(text_test)

    TC.train_models(text_train, text_test)

    i_run_end = time.time()
    run_time = i_run_end - i_run_start
    run_time = float("{:.2f}".format(run_time))
    print(f'Done training and testing models. Completed in time: {run_time} seconds.')

    m, fn = TC.train_best_model()

    TC.predict(m, fn)
