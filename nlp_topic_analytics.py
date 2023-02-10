# -*- coding: utf-8 -*-
"""NLP-Topic-Analytics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VHaCHZQTrOOH7_V5oKc2ZbGVlfDdR3Yi

# Housekeeping Setup
## Prerequisite: install and import libraries
"""

pip install catboost

pip install transformers

pip install sentence_transformers

pip install optuna

pip install shap

!pip install ktrain

pip install unidecode

pip install textstat

import os
import string
import re
import abc
import cv2
import datetime
import torch
import numpy as np 
import pandas as pd
from pandas_profiling import ProfileReport
from tqdm import tqdm
from matplotlib import style
style.use('default')
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
from statistics import mean

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
import unidecode
import textstat
import unicodedata
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
import plotly.graph_objects as go

import tensorflow as tf
import tensorflow.keras as keras
import ktrain
from ktrain import text

from transformers import AutoModel, BertTokenizerFast, DistilBertTokenizer, TFDistilBertModel, DistilBertModel
import transformers
from sentence_transformers import SentenceTransformer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Dropout, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model

from sklearn.utils import shuffle
from tensorflow.keras import regularizers
from transformers import BertTokenizer, TFBertModel, BertConfig

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.feature_extraction import stop_words
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import optuna
import shap
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.decomposition import NMF
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier, Pool

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"; 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

"""# 1. EDA"""

train=pd.read_csv('/content/drive/MyDrive/public_data.csv', index_col='id')

train.shape

train.head()

"""### 1.1 Label"""

pd.DataFrame(train['label'].value_counts()).plot(kind='bar', figsize=(18, 8), color='k')

"""### 1.2 Corpus"""

from gensim.models import Word2Vec, FastText
from gensim.test.utils import common_texts
from sklearn.decomposition import PCA
import plotly.graph_objects as go

corpus = []
for col in train.message:
   word_list = col.split(" ")
   corpus.append(word_list)

#show first value
corpus[0:1]

#generate vectors from corpus
model = Word2Vec(corpus, min_count=1, size = 56)


#pass the embeddings to PCA
X = model[model.wv.vocab] 
pca = PCA(n_components=2)
result = pca.fit_transform(X)

#create df from the pca results
pca_df = pd.DataFrame(result, columns = ['x','y'])

N = 1000000
words = list(model.wv.vocab)

#add the words for the hover effect
pca_df['word'] = words
pca_df.head()

fig = go.Figure(data=go.Scattergl(
   x = pca_df['x'],
   y = pca_df['y'],
   mode='markers',
   marker=dict(
       color=np.random.randn(N),
       colorscale='Viridis',
       line_width=1
   ),
   text=pca_df['word'],
   textposition="bottom center"
))

fig.show()

"""# 2. Modeling:

## 2.1. Shallow NLP -- Word Enbedding (TF-IDF) + Boosting (SVM, DT, RF, Boosting...)
"""

X = train['message']
y = train['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import unidecode
import textstat
import string  

lemmer = WordNetLemmatizer()

# Simple preprocessor.

def my_preprocess(doc):
    
    # Lowercase
    doc = doc.lower()
    
    # Replace URL with URL string
    doc = re.sub(r'http\S+', 'URL', doc)
    
    # Remove punctuation
    doc = re.sub(r'[^\w\s]', '', doc)
    
    # Replace AT with AT string
    doc = re.sub(r'@', 'AT', doc)
    
    # Replace all numbers/digits with the string NUM
    doc = re.sub(r'\b\d+\b', 'NUM', doc)
    
    # Remove numbers
    doc = re.sub(r'\d+', '', doc)
    
    # Lemmatize each word.
    doc = ' '.join([lemmer.lemmatize(w) for w in doc.split()])

    return doc

# These functions will calculate additional features on the document.
# They will be put into the Pipeline, called via the FunctionTransformer() function.
# Each one takes an entier corpus (as a list of documents), and should return
# an array of feature values (one for each document in the corpus).
# These functions can do anything they want; I've made most of them quick
# one-liners Hopefully the names of the functions will make them self explanitory.

def doc_length(corpus):
    return np.array([len(doc) for doc in corpus]).reshape(-1, 1)

def lexicon_count(corpus):
    return np.array([textstat.lexicon_count(doc) for doc in corpus]).reshape(-1, 1)

def _get_punc(doc):
    return len([a for a in doc if a in string.punctuation])

def punc_count(corpus):
    return np.array([_get_punc(doc) for doc in corpus]).reshape(-1, 1)

def _get_caps(doc):
    return sum([1 for a in doc if a.isupper()])

def capital_count(corpus):
    return np.array([_get_caps(doc) for doc in corpus]).reshape(-1, 1)

# See if the document ends with something like "Love Steve XXX"
def has_lovexxx(corpus):
    return np.array([bool(re.search(r"l[ou]+ve?.{0,10}x{2,5}\.? ?$", doc.lower())) for doc in corpus]).reshape(-1, 1)
##this might be useful

def has_money(corpus):
    return np.array([bool(re.search("[\$£]|\bpence\b|\bdollar\b", doc.lower())) for doc in corpus]).reshape(-1, 1)
##this might be useful

def has_sexy_phrase(corpus):
    return np.array([bool(re.search("sexy single|\bfree sexy\b|\bsexy pic\b|\blive sex\b", doc.lower())) for doc in corpus]).reshape(-1, 1)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.feature_extraction import stop_words
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF
from sklearn.neural_network import MLPClassifier

# Need to preprocess the stopwords, because scikit learn's TfidfVectorizer
# removes stopwords _after_ preprocessing
stop_words = [my_preprocess(word) for word in stopwords.words('english')]
# This vectorizer will be used to create the BOW features
vectorizer = TfidfVectorizer(preprocessor=my_preprocess, 
                             max_features = 1000, 
                             ngram_range=[1,4],
                             stop_words=None,
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.25, min_df=0.001, use_idf=True)

# This vectorizer will be used to preprocess the text before topic modeling.
# (I _could_ use the same vectorizer as above- but why limit myself?)
vectorizer2 = TfidfVectorizer(preprocessor=my_preprocess, 
                             max_features = 1000, 
                             ngram_range=[1,2],
                             stop_words=None,
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.25, min_df=0.001, use_idf=True)

nmf = NMF(n_components=25, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)
rf = RandomForestClassifier(criterion='entropy', random_state=223)
mlp = MLPClassifier(random_state=42, verbose=2, max_iter=200)



feature_processing =  FeatureUnion([ 
    ('bow', Pipeline([('cv', vectorizer), ])),
    ('topics', Pipeline([('cv', vectorizer2), ('nmf', nmf),])),
    ('length', FunctionTransformer(doc_length, validate=False)),
    ('words', FunctionTransformer(lexicon_count, validate=False)),
    ('punc_count', FunctionTransformer(punc_count, validate=False)),
    ('capital_count', FunctionTransformer(capital_count, validate=False)),   
    ('has_lovexxx', FunctionTransformer(has_lovexxx, validate=False)),    
    ('has_money', FunctionTransformer(has_money, validate=False)),
    ('has_sexy_phrase', FunctionTransformer(has_sexy_phrase, validate=False)),
])

steps = [('features', feature_processing)]

pipe = Pipeline([('features', feature_processing), ('clf', mlp)])

param_grid = {}

# You - yes you! Manually choose which classifier run you'd like to tryout latter.
# You can set this to either:
#
# "RF" - Random Forest
# "MLP" - NN
# or something else!
#
# and then re-run the entire notebook
which_clf = "RF"

if which_clf == "RF":

    steps.append(('clf', rf))

    param_grid = {
        'features__bow__cv__preprocessor': [None, my_preprocess],
        'features__bow__cv__max_features': [200, 500, 1000],
        'features__bow__cv__use_idf': [False],
        'features__topics__cv__stop_words': [None],
        'features__topics__nmf__n_components': [25, 75],
        'clf__n_estimators': [100, 500],
        'clf__class_weight': [None],
    }
    
elif which_clf == "MLP":
    
    steps.append(('clf', mlp))
    param_grid = {
        'features__bow__cv__preprocessor': [my_preprocess],
        'features__bow__cv__max_features': [1000, 3000],
        'features__bow__cv__min_df': [0],
        'features__bow__cv__use_idf': [False],
        'features__topics__nmf__n_components': [300],
        'clf__hidden_layer_sizes': [(100, ), (50, 50), (25, 25, 25)],
    }

pipe = Pipeline(steps)

search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=3, scoring='f1_macro', return_train_score=True, verbose=2)

search = search.fit(X_train, y_train)

print("Best parameter (CV scy_train%0.3f):" % search.best_score_)
print(search.best_params_)

pred_val = search.predict(X_val)
print(confusion_matrix(y_val, pred_val))
print(classification_report(y_val, pred_val))

ari = adjusted_rand_score(y_val, pred_val)
ami = adjusted_mutual_info_score(y_val, pred_val, average_method='arithmetic')

print("ARI: {}".format(ari))
print("AMI: {}".format(ami))

"""## 2.2 Deep Learning NLP:

2.2.1 Sentence Embedding (word2vec,BERT) + Deep NN(such as CNN, RNN, LSTM, GRU, Transformer...)

Word2Vector model
"""

df = pd.read_csv('/content/drive/MyDrive/public_data.csv')
print(df.info())
df = df.drop('id', axis=1)
df.head()

class Word2VecModel():
    def __init__(self, train_df):
        self.df = train_df
        X = train_df['message']
        y = pd.get_dummies(train_df.astype(str),columns=['label'], prefix='', prefix_sep='').drop('message', axis=1)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.MAX_VOCAB_SIZE = 4096
        self.EMBEDDING_DIM=100
        self.MAX_SEQUENCE_LENGTH= 100
        self.VALIDATION_SPLIT =0.2
        self.BATCH_SIZE =128
        self.EPOCHS =10
        self.model = ''

    
    def create_tokenizer(self, x, y ):
        sentences = x.values
        possible_labels = y.columns
        targets = y.values
        tokenizer= Tokenizer(num_words=self.MAX_VOCAB_SIZE)
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)
        data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        print("max sequence length:", max(len(s) for s in sequences))
        print("min sequence length:", min(len(s) for s in sequences))
        s = sorted(len(s) for s in sequences)
        print("median sequence length:", s[len(s) // 2])
        print("max word index:", max(max(seq) for seq in sequences if len(seq) > 0))
        word2idx = tokenizer.word_index
        print('Found %s unique tokens.' % len(word2idx))
        return word2idx, data, targets, possible_labels
        
    def load_embedding(self):
        word2vec = {}
        with open(os.path.join('glove.6B.100d.txt')) as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
        print('Found %s word vectors.' % len(word2vec))
        return word2vec
    
        
    def embedding_data(self, word2idx, word2vec):
        num_words = min(self.MAX_VOCAB_SIZE, len(word2idx) + 1)
        embedding_matrix = np.zeros((num_words, self.EMBEDDING_DIM))
        for word, i in word2idx.items():
            if i < self.MAX_VOCAB_SIZE:
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return num_words, embedding_matrix
                    
    def create_model(self, num_words, possible_labels, embedding_matrix):
        embedding_layer = Embedding(
            num_words,
            self.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=self.MAX_SEQUENCE_LENGTH,
            trainable=False
        )
        
        input_ = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        x=embedding_layer(input_)
        x=Conv1D(128,3,activation="relu")(x)
        x=GlobalMaxPooling1D()(x)
        x=Dense(128,activation="relu")(x)
        output = Dense(len(possible_labels), activation='sigmoid')(x)

        self.model = Model(input_, output)
        self.model.compile(
          loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy']
        )
    
    def train_model(self, data, targets):
        print('Training model...')
        r = self.model.fit(
          data,
          targets,
          batch_size=self.BATCH_SIZE,
          epochs=self.EPOCHS,
          validation_split=self.VALIDATION_SPLIT
        )
        return self.model
        
    def run(self):
        word2idx, data, targets, possible_labels = self.create_tokenizer(self.X_train, self.y_train)
        word2vec = self.load_embedding()
        num_words, embedding_matrix = self.embedding_data(word2idx, word2vec)
        self.create_model(num_words, possible_labels, embedding_matrix)
        self.train_model(data, targets)
        return self.model
    
    def eval_model(self):
        word2idx, data, targets, possible_labels = self.create_tokenizer(self.X_train, self.y_train)
        pred_val = self.model.predict(data)
        ari = adjusted_rand_score(self.y_val, pred_val)
        ami = adjusted_mutual_info_score(self.y_val, pred_val, average_method='arithmetic')

        print("ARI: {}".format(ari))
        print("AMI: {}".format(ami))

word2vec_model = Word2VecModel(df)
model = word2vec_model.run()

"""max sequence length: 76 min sequence length: 2 median sequence length: 10 max word index: 2203 Found 2203 unique tokens. Found 400000 word vectors. Training model... Epoch 1/10 53/53 [==============================] - 3s 42ms/step - loss: 0.1126 - accuracy: 0.0288 - val_loss: 0.0688 - val_accuracy: 0.0585 Epoch 2/10 53/53 [==============================] - 2s 34ms/step - loss: 0.0618 - accuracy: 0.1643 - val_loss: 0.0581 - val_accuracy: 0.2161 Epoch 3/10 53/53 [==============================] - 2s 36ms/step - loss: 0.0487 - accuracy: 0.3985 - val_loss: 0.0449 - val_accuracy: 0.4006 Epoch 4/10 53/53 [==============================] - 2s 36ms/step - loss: 0.0375 - accuracy: 0.5617 - val_loss: 0.0369 - val_accuracy: 0.5110 Epoch 5/10 53/53 [==============================] - 2s 33ms/step - loss: 0.0306 - accuracy: 0.6551 - val_loss: 0.0319 - val_accuracy: 0.5869 Epoch 6/10 25/53 [=============>................] - ETA: 0s - loss: 0.0269 - accuracy: 0.7097

2.2.2 Transfer learning: Few-shot, zero-shot + Pipeline building in Deep Learning

Distillbert with ktrain learner
"""

X = df['message']
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

model_name = 'distilbert-base-uncased'

classes = y.unique().tolist()
#model = DistilBertModel.from_pretrained("distilbert-base-uncased")
trans = text.Transformer(model_name,maxlen=512, class_names=classes)

train_data = trans.preprocess_train(np.array(X_train), np.array(y_train))
valiation_data = trans.preprocess_train(np.array(X_val), np.array(y_val))

model = trans.get_classifier()
learner = ktrain.get_learner(model, train_data = train_data, val_data = valiation_data, batch_size = 16)
learner.lr_find( start_lr=1e-07,
    lr_mult=1.01,
    max_epochs=10,
    class_weight=None,
    stop_factor=4,
    show_plot=True,
    suggest=True,
    restore_weights_only=False,
    verbose=1,)

learner.fit_onecycle(lr=2.36E-05,epochs=10)

learner.validate(class_names=classes)

learner.view_top_losses(n = 10, preproc = trans)

predictor = ktrain.get_predictor(learner.model, preproc=trans)

"""TFDistilBertModel"""

# class MetaService(type):
#     def __new__(cls, name:str, bases, namespace, **kwargs):
#         home = os.environ.get('HOME')
#         if not name.startswith('Srv'):
#             raise TypeError('[Fatal] Invalid service statement')
#         elif not home.endswith('Jiayu'):
#             raise TypeError('[Fatal] Invalid device')
#         return super().__new__(cls, name, bases, namespace, **kwargs)
    

class SrvModelTrainer():
# (metaclass=MetaService):
    def __init__(self, data, batch_size, epochs):
        self.data = data
        self.max_len = 32
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes=len(self.data.label.unique())
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dbert_inp = ''

    def clean_stopwords_shortwords(self, w):
        stop = nltk.download('stopwords', quiet=True)
        stopwords_list=set(stopwords.words('english'))
        words = w.split() 
        clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
        return " ".join(clean_words) 

    def preprocess_sentence(self, w):
        w = self.clean_stopwords_shortwords(w)
        w = re.sub(r'@\w+', '',w)
        return w

    def clean_data(self):
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)
        print('File has {} rows and {} columns'.format(self.data.shape[0], self.data.shape[1]))
        self.data = shuffle(self.data)
        self.data.head()
        self.data['message'] = self.data['message'].map(self.preprocess_sentence)
    
    def encoding_label(self):
        label_dict = {}
        for i in range(len(self.data['label'].unique())):
            label_dict[self.data['label'].unique()[i]] = i
        self.data['label_num'] = self.data['label'].apply(lambda x: label_dict.get(x))
        self.sentences = self.data['message']
        self.labels = self.data['label_num']

    def create_model(self):
        inps = Input(shape=(self.max_len,), dtype='int64')
        masks= Input(shape=(self.max_len,), dtype='int64')
        dbert_layer = self.model(inps, attention_mask=masks)[0][:,0,:]
        dense = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01))(dbert_layer)
        dropout = Dropout(0.5)(dense)
        pred = Dense(self.num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)
        model = tf.keras.Model(inputs=[inps,masks], outputs=pred)
        print(model.summary())
        return model   
    
    def spliting_data(self):
        input_ids=[]
        attention_masks=[]
        for msg in self.data['message']:
            dbert_inps=self.tokenizer.encode_plus(
                msg,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                truncation=True
                )
            input_ids.append(dbert_inps['input_ids'])
            attention_masks.append(dbert_inps['attention_mask'])
        input_ids=np.asarray(input_ids)
        attention_masks=np.array(attention_masks)
        labels=np.array(self.labels)
        train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,labels,attention_masks,test_size=0.2)
        return train_inp,val_inp,train_label,val_label,train_mask,val_mask

    def train_model(self):
        self.clean_data()
        self.encoding_label()
        model = self.create_model()
        train_inp,val_inp,train_label,val_label,train_mask,val_mask = self.spliting_data()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
        history=model.fit(
            [train_inp,train_mask],
            train_label,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([val_inp,val_mask],val_label))
        return history

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
trainer = SrvModelTrainer(df, 16, 10)
trainer.train_model()

"""## 2.3. Hybrid NLP:

### 2.3.1 Sentence Embedding (word2vec) + Boosting (KNN, RF, LGBM, XGBoost)
"""







"""### 2.3.2 Sentence Embedding (BERT) + Boosting (KNN, RF, LGBM, CATBoost, MLP, SGD)

1. distilbert-base-nli-mean-tokens
2. xlm-r-bert-base-nli-stsb-mean-tokens
3. bert-base-wikipedia-sections-mean-tokens
4. paraphrase-mpnet-base-v2
5. bert-base-nli-mean-tokens

### GridSearchCV HPT
"""

class hybrid_bert(metaclass= abc.ABCMeta):

  def initialize(self, df, emb_pkg='distilbert-base-nli-mean-tokens', clf='RF'):
    self.df = df
    self.X = df.drop(['label'],axis=1)
    self.y = df['label']
    self.emb_pkg = emb_pkg
    self.clf = clf
    self.emb = SentenceTransformer(self.emb_pkg)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

  def clean_text(self, text):
    table = text.maketrans(
        dict.fromkeys(string.punctuation))
    
    words = word_tokenize(
        text.lower().strip().translate(table))
    words = [word for word in words if word not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]    
    return " ".join(lemmed)

  def get_sentence_lengths(self, text):
    tokened = sent_tokenize(text)
    lengths = []
    
    for idx,i in enumerate(tokened):
        splited = list(i.split(" "))
        lengths.append(len(splited))
        
    return (max(lengths), min(lengths), round(mean(lengths), 3))

  def create_features(self, df):
    df_f = pd.DataFrame(index=df.index)
    df_f['text_len'] = df['message'].apply(len)
    df_f['text_clean_len' ]= df['clean_message'].apply(len)
    df_f['text_len_div'] = df_f['text_clean_len' ] / df_f['text_len']
    df_f['text_word_count'] = df['clean_message'].apply(lambda x : len(x.split(' ')))
    df_f[['max_len_sent','min_len_sent','avg_len_sent']] = df.apply(lambda x: self.get_sentence_lengths(x['clean_message']),axis=1, result_type='expand')
    
    return df_f

  def cleansed(self,df):
    df['clean_message'] = df['message'].apply(self.clean_text)
    df_cleaned = pd.concat([df, self.create_features(df)], axis=1, copy=False, sort=False).drop(['message'],axis=1)
    return df_cleaned

  def convert(self):
    df_clean=self.cleansed(self.X)
    num = df_clean.select_dtypes(include=['int64', 'float64']).columns
    cat = df_clean.select_dtypes(include=['object', 'bool']).columns
    embedder = FunctionTransformer(lambda item:self.emb.encode(list(item), convert_to_tensor=True, show_progress_bar=False).detach().cpu().numpy())
    preprocessor = ColumnTransformer(transformers=[('embedder', embedder, 'clean_message')], remainder='passthrough' )
    trans = [('cat', preprocessor, cat), ('num', MinMaxScaler(), num)]
    col_transform = ColumnTransformer(transformers=trans)
    return col_transform


  def train_model(self):
    self.col_trans=self.convert()
    self.para(clf=self.clf)
    data=self.cleansed(self.X_train)
    search = GridSearchCV(self.pipe, self.grid, cv=5, n_jobs=1, scoring='f1_macro', return_train_score=True, verbose=2)
    self.search = search.fit(data, self.y_train)
    print("Best parameter (CV scy_train: %0.3f)" % self.search.best_score_)
    print(self.search.best_params_)

  def predict(self):
    data=self.cleansed(self.X_val)
    pred_val=self.search.predict(data)
    print(confusion_matrix(self.y_val, pred_val))
    print(classification_report(self.y_val, pred_val))

    ari = adjusted_rand_score(self.y_val, pred_val)
    ami = adjusted_mutual_info_score(self.y_val, pred_val, average_method='arithmetic')

    print("ARI: {}".format(ari))
    print("AMI: {}".format(ami))

    df_test = pd.read_csv('/content/drive/MyDrive/input_data.csv', index_col='id')
    df_test.info()
    df_test.head()

    input_data=self.cleansed(df_test)
    pred_test = self.search.predict(input_data)

    my_submission = pd.DataFrame({'Id': df_test.index, 'label': pred_test})
    print(my_submission.head())

    my_submission.to_csv('/content/drive/MyDrive/answers.csv', encoding='utf-8', index=False)

  @abc.abstractmethod
  def para(self):
    pass

class parameter_tuning(hybrid_bert):

  def para(self, clf='RF'):

    steps = [('prep', self.col_trans)]
    param_grid = {}

    par = {
    'leaf_estimation_method': 'Gradient',
    'learning_rate': 0.001,
    'max_depth': 8,
    'bootstrap_type': 'Bernoulli',
    'objective': 'MultiClass',
    'random_state': 42,
    'verbose': 0,
    'early_stopping_rounds' : 100,
    'iterations': 25,
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'l2_leaf_reg': 10,
    'subsample': 0.7,
    'thread_count': 20
    }

    cat = CatBoostClassifier(**par) 
    sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001,n_jobs=1, random_state=0, max_iter=2000) 
    rf = RandomForestClassifier(criterion='entropy', random_state=223)
    knn = KNeighborsClassifier()
    lgbm = lgb.LGBMClassifier( objective='multiclass', boosting_type='gbdt', learning_rate=0.01, num_leaves=31) 
    nmf = NMF(n_components=25, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)
    mlp = MLPClassifier(random_state=42, verbose=2, max_iter=600) 

    if clf == 'KNN':
      steps.append(('clf', knn))
      param_grid = { 'clf__n_neighbors' : [5,13],  'clf__weights' : ['uniform','distance'],  'clf__metric' : ['euclidean','manhattan']} 

    if clf == 'CAT':
      steps.append(('clf', cat))
      param_grid={'clf__learning_rate':[0.01, 0.001], 'clf__l2_leaf_reg':[10, 5], 'clf__thread_count':[20, 10] }

    if clf == "RF":
      steps.append(('clf', rf))
      param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [20, 100] , 'clf__min_samples_split': [2, 4]} 
        
    if clf == "MLP":
      steps.append(('clf', mlp))
      param_grid = {'clf__hidden_layer_sizes': [(100, ), (50, 50), (25, 25, 25), (15, 15, 15, 15), (10, 10, 10, 10, 10), (8, 8, 8, 8, 8, 8), (5, 5, 5, 5, 5, 5, 5), (3, 3, 3, 3, 3, 3, 3, 3)] }

    if clf == "NMF":
      steps.append(('clf', nmf))
      param_grid = {'clf__hidden_layer_sizes': [(100, ), (50, 50), (25, 25, 25)], }

    if clf == "LGBM":
      steps.append(('clf', lgbm))
      param_grid = {'clf__reg_alpha': [0.5, 1.5],  'clf__reg_lambda':[0, 1], 'clf__num_leaves': [50, 100]  } 

    if clf == 'SGD':
      steps.append(('clf', sgd))
      param_grid = {"clf__loss" : ["hinge", "log", "squared_hinge", "modified_huber"], "clf__alpha" : [0.0001, 0.001]}

    self.pipe = Pipeline(steps)
    self.grid = param_grid

"""CAT & bert-base-nli-mean-tokens"""

prmt=parameter_tuning()
prmt.initialize(train, emb_pkg='bert-base-nli-mean-tokens', clf='CAT')
prmt.train_model()
prmt.predict()

"""RF & distilbert-base-nli-mean-tokens"""

prmt=parameter_tuning()
prmt.initialize(train, emb_pkg='distilbert-base-nli-mean-tokens', clf='RF')
prmt.train_model()
prmt.predict()

"""LGBM & bert-base-wikipedia-sections-mean-tokens"""

prmt=parameter_tuning()
prmt.initialize(train, emb_pkg='paraphrase-mpnet-base-v2', clf='LGBM')
prmt.train_model()
prmt.predict()

"""KNN & xlm-r-bert-base-nli-stsb-mean-tokens"""

prmt=parameter_tuning()
prmt.initialize(train, emb_pkg='xlm-r-bert-base-nli-stsb-mean-tokens', clf='KNN')
prmt.train_model()
prmt.predict()

"""MLP & paraphrase-mpnet-base-v2"""

prmt=parameter_tuning()
prmt.initialize(train, emb_pkg='paraphrase-mpnet-base-v2', clf='MLP')
prmt.train_model()
prmt.predict()

"""SGD & paraphrase-mpnet-base-v2"""

prmt=parameter_tuning()
prmt.initialize(train, emb_pkg='paraphrase-mpnet-base-v2', clf='SGD')
prmt.train_model()
prmt.predict()

"""SGD & stsb-roberta-base-v2"""

prmt=parameter_tuning()
prmt.initialize(train, emb_pkg='stsb-roberta-base-v2', clf='SGD')
prmt.train_model()
prmt.predict()

"""Optuna HPT"""

class hybrid_bert():

  def __init__(self, df, emb_pkg='distilbert-base-nli-mean-tokens'):
    self.df = df
    self.X = df.drop(['label'],axis=1)
    self.y = df['label']
    self.emb_pkg = emb_pkg
    self.emb = SentenceTransformer(self.emb_pkg)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

  def clean_text(self, text):
    table = text.maketrans(
        dict.fromkeys(string.punctuation))
    
    words = word_tokenize(
        text.lower().strip().translate(table))
    words = [word for word in words if word not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]    
    return " ".join(lemmed)

  def get_sentence_lengths(self, text):
    tokened = sent_tokenize(text)
    lengths = []
    
    for idx,i in enumerate(tokened):
        splited = list(i.split(" "))
        lengths.append(len(splited))
        
    return (max(lengths), min(lengths), round(mean(lengths), 3))

  def create_features(self, df):
    df_f = pd.DataFrame(index=df.index)
    df_f['text_len'] = df['message'].apply(len)
    df_f['text_clean_len' ]= df['clean_message'].apply(len)
    df_f['text_len_div'] = df_f['text_clean_len' ] / df_f['text_len']
    df_f['text_word_count'] = df['clean_message'].apply(lambda x : len(x.split(' ')))
    df_f[['max_len_sent','min_len_sent','avg_len_sent']] = df.apply(lambda x: self.get_sentence_lengths(x['clean_message']),axis=1, result_type='expand')
    
    return df_f

  def cleansed(self,df):
    df['clean_message'] = df['message'].apply(self.clean_text)
    df_cleaned = pd.concat([df, self.create_features(df)], axis=1, copy=False, sort=False).drop(['message'],axis=1)

    return df_cleaned

  def convert(self):
    df_clean=self.cleansed(self.X)
    num = df_clean.select_dtypes(include=['int64', 'float64']).columns
    cat = df_clean.select_dtypes(include=['object', 'bool']).columns
    embedder = FunctionTransformer(lambda item:self.emb.encode(list(item), convert_to_tensor=True, show_progress_bar=False).detach().cpu().numpy())
    preprocessor = ColumnTransformer(transformers=[('embedder', embedder, 'clean_message')], remainder='passthrough' )
    trans = [('cat', preprocessor, cat), ('num', MinMaxScaler(), num)]
    col_trans = ColumnTransformer(transformers=trans)
    data_xtr = col_trans.fit_transform(self.cleansed(self.X_train))
    data_xte = col_trans.fit_transform(self.cleansed(self.X_val))
    data_ytr = self.y_train
    data_yte = self.y_val
    df_test = pd.read_csv('/content/drive/MyDrive/input_data.csv', index_col='id')
    X_test = col_trans.fit_transform(self.cleansed(df_test))
    return data_xtr, data_xte, data_ytr, data_yte, X_test

def objective(trial):

  classifier = trial.suggest_categorical('classifier', ['SVR', 'RF', 'KNN', 'SGD'])

  if classifier == 'RF':
      n_estimators = trial.suggest_int('n_estimators', 100, 200)
      max_depth = int(trial.suggest_float('max_depth', 1, 32, log=True))
      min_samples_split = trial.suggest_int('min_samples_split', 2, 4) 
      clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, criterion='entropy', random_state=223)
      
  elif classifier == 'SVR':
      c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
      clf = SVC(C=c, gamma='auto')

  elif classifier == 'KNN':
      n_neighbors = trial.suggest_int('n_neighbors', 5, 13)
      weights = trial.suggest_categorical('weights', ['uniform','distance'])
      metric = trial.suggest_categorical('metric', ['euclidean','manhattan'])
      clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

  else: 
      alpha = trial.suggest_float('alpha', 0.0001, 0.001)
      loss = trial.suggest_categorical('loss', ["hinge", "log", "squared_hinge", "modified_huber"])
      clf = SGDClassifier(alpha=alpha, loss=loss, penalty='l2', n_jobs=1, random_state=0, max_iter=2000)


  clf.fit(data_xtr, data_ytr)
  pred_val = clf.predict(data_xte)

  ari = adjusted_rand_score(data_yte, pred_val)
  ami = adjusted_mutual_info_score(data_yte, pred_val, average_method='arithmetic')

  print("ARI: {}".format(ari)), print(" AMI: {}".format(ami))

  return cross_val_score(clf, data_xtr, data_ytr, n_jobs=-1, cv=5, scoring = 'f1_macro').mean()

hbd=hybrid_bert(train)
data_xtr, data_xte, data_ytr, data_yte, X_test=hbd.convert()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial)

# Predict label of input data with optimal hyper-parameters identified from last step
hbd=hybrid_bert(train)
data_xtr, data_xte, data_ytr, data_yte, X_test=hbd.convert()
df_test = pd.read_csv('/content/drive/MyDrive/input_data.csv', index_col='id')

clf=  SVC(C=14.344570558281074, gamma='auto')

clf.fit(data_xtr, data_ytr)
pred_test = clf.predict(X_test)

my_submission = pd.DataFrame({'Id': df_test.index, 'label': pred_test})
print(my_submission.head())

my_submission.to_csv('/content/drive/MyDrive/answers.csv', encoding='utf-8', index=False)

"""Plotting the optimization history of the study."""

optuna.visualization.plot_optimization_history(study)

"""Plotting the accuracies for each hyperparameter for each trial."""

optuna.visualization.plot_slice(study)

"""Plotting the accuracy surface for the hyperparameters involved in the random forest model."""

optuna.visualization.plot_contour(study, params=['n_estimators', 'max_depth'])