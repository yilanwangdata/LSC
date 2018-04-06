part of code of NLP from several Jupyters and Eclipse
```
import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import nltk
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
datafile = os.path.join('..', 'data', 'labeledTrainData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df.head()
def display(text, title):
    print(title)
    print("\n------------------\n")
    print(text) 
raw_example = df['review'][1]
display(raw_example, 'original_data_quickshow')
example = BeautifulSoup(raw_example, 'html.parser').get_text()
display(example, 'no_HTML')
example_letters = re.sub(r'[^a-zA-Z]', ' ', example)
display(example_letters, 'no_dot')
words = example_letters.lower().split()
display(words, 'pure_words')
stopwords = {}.fromkeys([ line.rstrip() for line in open('../stopwords.txt')])
words_nostop = [w for w in words if w not in stopwords]
display(words_nostop, 'after_stopwords')
eng_stopwords = set(stopwords)

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)
clean_text(raw_example)
df['clean_review'] = df.review.apply(clean_text)
df.head()
vectorizer = CountVectorizer(max_features = 5000) 
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()
train_data_features.shape
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, df.sentiment)
confusion_matrix(df.sentiment, forest.predict(train_data_features))
del df
del train_data_features
datafile = os.path.join('..', 'data', 'testData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df['clean_review'] = df.review.apply(clean_text)
df.head()
test_data_features = vectorizer.transform(df.clean_review).toarray()
test_data_features.shape
result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id, 'sentiment':result})
output.to_csv(os.path.join('..', 'data', 'Bag_of_Words_model.csv'), index=False)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data_features,df.sentiment,test_size = 0.2, random_state = 0)
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
LR_model = LogisticRegression()
LR_model = LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)
cnf_matrix = confusion_matrix(y_test,y_pred)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

print("accuracy metric in the testing dataset: ", (cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[1,1]+cnf_matrix[1,0]+cnf_matrix[0,1]))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

import warnings
warnings.filterwarnings("ignore")

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def split_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [clean_text(s) for s in raw_sentences if s]
    return sentences
sentences = sum(review_part.apply(split_sentences), [])
print('{} reviews -> {} sentences'.format(len(review_part), len(sentences)))

sentences_list = []
for line in sentences:
    sentences_list.append(nltk.word_tokenize(line))
num_features = 300    
min_word_count = 40  
num_workers = 4       
context = 10      
model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)
from gensim.models.word2vec import Word2Vec
model = Word2Vec(sentences_list, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context)
model.init_sims(replace=True)
model.save(os.path.join('..', 'models', model_name))
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

def to_review_vector(review):
    global word_vec
    
    review = clean_text(review, remove_stopwords=True)
   
    word_vec = np.zeros((1,300))
    for word in review:
      
        if word in model:
            word_vec += np.array([model[word]])
    
    return pd.Series(word_vec.mean(axis = 0))

train_data_features = df.review.apply(to_review_vector)
train_data_features.head()
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data_features,df.sentiment,test_size = 0.2, random_state = 0)
LR_model = LogisticRegression()
LR_model = LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)
cnf_matrix = confusion_matrix(y_test,y_pred)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

print("accuracy metric in the testing dataset: ", (cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[1,1]+cnf_matrix[1,0]+cnf_matrix[0,1]))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

from gensim.models.word2vec import Word2Vec
def load_dataset(name, nrows=None):
    datasets = {
        'unlabeled_train': 'unlabeledTrainData.tsv',
        'labeled_train': 'labeledTrainData.tsv',
        'test': 'testData.tsv'
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('..', 'data', datasets[name])
    df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)
    print('Number of reviews: {}'.format(len(df)))
    return df

eng_stopwords = {}.fromkeys([ line.rstrip() for line in open('../stopwords.txt')])

def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def print_call_counts(f):
    n = 0
    def wrapped(*args, **kwargs):
        nonlocal n
        n += 1
        if n % 1000 == 1:
            print('method {} called {} times'.format(f.__name__, n))
        return f(*args, **kwargs)
    return wrapped

@print_call_counts
def split_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [clean_text(s) for s in raw_sentences if s]
    return sentences
%time sentences = sum(df.review.apply(split_sentences), [])
print('{} reviews -> {} sentences'.format(len(df), len(sentences)))
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
num_features = 300   
min_word_count = 40 
num_workers = 4    
context = 10          
downsampling = 1e-3  

model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)
print('Training model...')
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

model.save(os.path.join('..', 'models', model_name))
model.most_similar("inflation")

from gensim.models.word2vec import Word2Vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

def load_dataset(name, nrows=None):
    datasets = {
        'unlabeled_train': 'unlabeledTrainData.tsv',
        'labeled_train': 'labeledTrainData.tsv',
        'test': 'testData.tsv'
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('..', 'data', datasets[name])
    df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)
    print('Number of reviews: {}'.format(len(df)))
    return df
eng_stopwords = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words
model_name = '300features_40minwords_10context.model'
model = Word2Vec.load(os.path.join('..', 'models', model_name))
def to_review_vector(review):
    words = clean_text(review, remove_stopwords=True)
    array = np.array([model[w] for w in words if w in model])
    return pd.Series(array.mean(axis=0))
train_data_features = df.review.apply(to_review_vector)
train_data_features.head()
forest = RandomForestClassifier(n_estimators = 100, random_state=42)
forest = forest.fit(train_data_features, df.sentiment)
confusion_matrix(df.sentiment, forest.predict(train_data_features))
test_data_features = df.review.apply(to_review_vector)
test_data_features.head()
result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id, 'sentiment':result})
output.to_csv(os.path.join('..', 'data', 'Word2Vec_model.csv'), index=False)
output.head()
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] // 10
%%time

kmeans_clustering = KMeans(n_clusters = num_clusters, n_jobs=4)
idx = kmeans_clustering.fit_predict(word_vectors)
word_centroid_map = dict(zip(model.index2word, idx))
import pickle

filename = 'word_centroid_map_10avg.pickle'
with open(os.path.join('..', 'models', filename), 'bw') as f:
    pickle.dump(word_centroid_map, f)
for cluster in range(0,10):
    print("\nCluster %d" % cluster)
    print([w for w,c in word_centroid_map.items() if c == cluster])
wordset = set(word_centroid_map.keys())

def make_cluster_bag(review):
    words = clean_text(review, remove_stopwords=True)
    return (pd.Series([word_centroid_map[w] for w in words if w in wordset])
              .value_counts()
              .reindex(range(num_clusters+1), fill_value=0))
train_data_features = df.review.apply(make_cluster_bag)
train_data_features.head()
forest = RandomForestClassifier(n_estimators = 100, random_state=42)
forest = forest.fit(train_data_features, df.sentiment)
confusion_matrix(df.sentiment, forest.predict(train_data_features))
df = load_dataset('test')
df.head()
test_data_features = df.review.apply(make_cluster_bag)
test_data_features.head()
result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id, 'sentiment':result})
output.to_csv(os.path.join('..', 'data', 'Word2Vec_BagOfClusters.csv'), index=False)
output.head()

import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
if __name__ == '__main__':
    
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 4:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.model.wv.save_word2vec_format(outp2, binary=False)
#python word2vec_model.py dataft.txt dataft.model dataft.vector
from gensim.models import Word2Vec

dataft_model = Word2Vec.load('wiki.zh.text.model')

testwords = ['inflation','china','president','bitcoin','tax']
for i in range(5):
    res = dataft.most_similar(testwords[i])
    print (testwords[i])
    print (res)
import tensorflow as tf
import numpy as np
import math
import collections
import pickle as pkl
from pprint import pprint
from pymongo import MongoClient
import re
import os.path as path
import os

class word2vec():
    def __init__(self,
                 vocab_list=None,
                 embedding_size=200,
                 win_len=3,
                 num_sampled=1000,
                 learning_rate=1.0,
                 logdir='/tmp/simple_word2vec',
                 model_path= None
                 ):
        self.batch_size     = None
        if model_path!=None:
            self.load_model(model_path)
        else:
          
            assert type(vocab_list)==list
            self.vocab_list     = vocab_list
            self.vocab_size     = vocab_list.__len__()
            self.embedding_size = embedding_size
            self.win_len        = win_len
            self.num_sampled    = num_sampled
            self.learning_rate  = learning_rate
            self.logdir         = logdir

            self.word2id = {}  
            for i in range(self.vocab_size):
                self.word2id[self.vocab_list[i]] = i
            self.train_words_num = 0 
            self.train_sents_num = 0
            self.train_times_num = 0 
            self.train_loss_records = collections.deque(maxlen=10)
            self.train_loss_k10 = 0

        self.build_graph()
        self.init_op()
        if model_path!=None:
            tf_model_path = os.path.join(model_path,'tf_vars')
            self.saver.restore(self.sess,tf_model_path)

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.train.SummaryWriter(self.logdir, self.sess.graph)

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.embedding_dict = tf.Variable(
                tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0)
            )
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                              stddev=1.0/math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs) 
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights = self.nce_weight,
                    biases = self.nce_biases,
                    labels = self.train_labels,
                    inputs = embed,
                    num_sampled = self.num_sampled,
                    num_classes = self.vocab_size
                )
            )

            tf.scalar_summary('loss',self.loss) 

           
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss) 

            self.test_word_id = tf.placeholder(tf.int32,shape=[None])
            vec_l2_model = tf.sqrt( 
                tf.reduce_sum(tf.square(self.embedding_dict),1,keep_dims=True)
            )

            avg_l2_model = tf.reduce_mean(vec_l2_model)
            tf.scalar_summary('avg_vec_model',avg_l2_model)

            self.normed_embedding = self.embedding_dict / vec_l2_model
        
            test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
            self.similarity = tf.matmul(test_embed, self.normed_embedding, transpose_b=True)

            
            self.init = tf.global_variables_initializer()

            self.merged_summary_op = tf.merge_all_summaries()

            self.saver = tf.train.Saver()

    def train_by_sentence(self, input_sentence=[]):
       
        sent_num = input_sentence.__len__()
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence:
            for i in range(sent.__len__()):
                start = max(0,i-self.win_len)
                end = min(sent.__len__(),i+self.win_len+1)
                for index in range(start,end):
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])
                        label_id = self.word2id.get(sent[index])
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        if len(batch_inputs)==0:
            return
        batch_inputs = np.array(batch_inputs,dtype=np.int32)
        batch_labels = np.array(batch_labels,dtype=np.int32)
        batch_labels = np.reshape(batch_labels,[batch_labels.__len__(),1])

        feed_dict = {
            self.train_inputs: batch_inputs,
            self.train_labels: batch_labels
        }
        _, loss_val, summary_str = self.sess.run([self.train_op,self.loss,self.merged_summary_op], feed_dict=feed_dict)

      
        self.train_loss_records.append(loss_val)
       
        self.train_loss_k10 = np.mean(self.train_loss_records)
        if self.train_sents_num % 1000 == 0 :
            self.summary_writer.add_summary(summary_str,self.train_sents_num)
            print("{a} sentences dealed, loss: {b}"
                  .format(a=self.train_sents_num,b=self.train_loss_k10))
        self.train_words_num += batch_inputs.__len__()
        self.train_sents_num += input_sentence.__len__()
        self.train_times_num += 1

    def cal_similarity(self,test_word_id_list,top_k=10):
        sim_matrix = self.sess.run(self.similarity, feed_dict={self.test_word_id:test_word_id_list})
        sim_mean = np.mean(sim_matrix)
        sim_var = np.mean(np.square(sim_matrix-sim_mean))
        test_words = []
        near_words = []
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.vocab_list[test_word_id_list[i]])
            nearst_id = (-sim_matrix[i,:]).argsort()[1:top_k+1]
            nearst_word = [self.vocab_list[x] for x in nearst_id]
            near_words.append(nearst_word)
        return test_words,near_words,sim_mean,sim_var

    def save_model(self, save_path):

        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

     
        model = {}
        var_names = ['vocab_size',     
                     'vocab_list',      
                     'learning_rate',   
                     'word2id',       
                     'embedding_size',  
                     'logdir',         
                     'win_len',         
                     'num_sampled',    
                     'train_words_num',      
                     'train_sents_num', 
                     'train_times_num', 
                     'train_loss_records',  
                     'train_loss_k10', 
                     ]
        for var in var_names:
            model[var] = eval('self.'+var)

        param_path = os.path.join(save_path,'params.pkl')
        if os.path.exists(param_path):
            os.remove(param_path)
        with open(param_path,'wb') as f:
            pkl.dump(model,f)

        tf_path = os.path.join(save_path,'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess,tf_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise RuntimeError('file not exists')
        param_path = os.path.join(model_path,'params.pkl')
        with open(param_path,'rb') as f:
            model = pkl.load(f)
            self.vocab_list = model['vocab_list']
            self.vocab_size = model['vocab_size']
            self.logdir = model['logdir']
            self.word2id = model['word2id']
            self.embedding_size = model['embedding_size']
            self.learning_rate = model['learning_rate']
            self.win_len = model['win_len']
            self.num_sampled = model['num_sampled']
            self.train_words_num = model['train_words_num']
            self.train_sents_num = model['train_sents_num']
            self.train_times_num = model['train_times_num']
            self.train_loss_records = model['train_loss_records']
            self.train_loss_k10 = model['train_loss_k10']

if __name__=='__main__':


    stop_words = []
    with open('stop_words.txt',encoding= 'utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('now words'.format(n=len(stop_words)))

   
    raw_word_list = []
    sentence_list = []
    with open('datawj.txt',encoding='gbk') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n','')
            while ' ' in line:
                line = line.replace(' ','')
            if len(line)>0:
                raw_words = list(line)
                dealed_words = []
                for word in raw_words:
                    if word not in stop_words and word not in ['www','com','http']:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line = f.readline()
    word_count = collections.Counter(raw_word_list)
    print('choose our words'
          .format(n1=len(raw_word_list),n2=len(word_count)))
    word_count = word_count.most_common(30000)
    word_list = [x[0] for x in word_count]
    w2v = word2vec(vocab_list=word_list,    
                   embedding_size=200,
                   win_len=2,
                   learning_rate=1,
                   num_sampled=100,         
                   logdir='/tmp/280')       
 num_steps = 10000
    for i in range(num_steps):
        #print (i%len(sentence_list))
        sent = sentence_list[i%len(sentence_list)]
        w2v.train_by_sentence([sent])
    w2v.save_model('model')
    
    w2v.load_model('model') 
    test_word = ['president','china']
    test_id = [word_list.index(x) for x in test_word]
    test_words,near_words,sim_mean,sim_var = w2v.cal_similarity(test_id)
    print (test_words,near_words,sim_mean,sim_var)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
 
#word_tokens = word_tokenize(example_sent)
 
#filtered_sentence = [w for w in word_tokens if not w in stop_words]
summary_word = []
for i in range(len(data.index)):
    #word_tokens=word_tokenize(data['summary'].iloc[i])
    words=data['summary'].iloc[i].split(' ')
    summary_word.append(words)
#print summary_word
type(summary_word)
summary_word[0:5]
summary_lower=[]
for line in summary_word:
    line_lower=[]
    for w in line:
        w=w.lower()
        line_lower.append(w)
    summary_lower.append(line_lower)
print summary_lower[0:3]       
summary_word2=[]
all_words=[]
for line in summary_lower:
    line_clean = []
    for w in line:
        if w not in stop_words:
            line_clean.append(w)
            all_words.append(str(w))
    summary_word2.append(line_clean)
print summary_word2[0:3]
type(summary_word2)    
summary_word2=pd.Series(summary_word2)
summary_word2[0:3]
data['summary_word']=summary_word2

print len(summary_word2)
print data.shape
data.iloc[5:18]
headline_word = []
for i in range(len(data.index)):
    #word_tokens=word_tokenize(data['summary'].iloc[i])
    words=data['headline'].iloc[i].split(' ')
    headline_word.append(words)
#print summary_word
type(headline_word)
headline_word[0:5]
headline_lower=[]
for line in headline_word:
    line_lower=[]
    for w in line:
        w=w.lower()
        line_lower.append(w)
    headline_lower.append(line_lower)
print headline_lower[0:3] 
headline_word2=[]
all_words_headline=[]
for line in headline_lower:
    line_clean = []
    for w in line:
        if w not in stop_words:
            line_clean.append(w)
            all_words_headline.append(str(w))
    headline_word2.append(line_clean)
print headline_word2[0:3]
headline_word2=pd.Series(headline_word2)
data['headline_word']=headline_word2
data.iloc[19:30]
dates=[]
for line in range(len(data.index)):
    date=re.findall(r'(?:Jan|Feb|March|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\.]* (?:\d{1,2},) ?\d{4}', data['info'].iloc[line])
    dates.append(date)
print dates[0:5]
print len(dates)
type(dates)
data['dates']=dates
data.head()
data=data.drop('headline',axis=1)
data=data.drop('info',axis=1)
data=data.drop('summary',axis=1)
import nltk
from nltk.probability import FreqDist
dist=FreqDist(all_words)
vocabl=dist.keys()
print type(dist)
print len(dist)
print vocabl[:10]
print dist['$1.93']
print dist['increase']
all_words=pd.DataFrame({'all_words':all_words})
all_words.head()
words_count=all_words.groupby(by=['all_words'])['all_words'].agg({"count":np.size})
words_count=words_count.reset_index().sort_values(by=["count"],ascending=False)
words_count.head()
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

wordcloud=WordCloud(font_path="./data/simhei.ttf",background_color="white",max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
words_headline=pd.DataFrame({'all_words_headline':all_words_headline})
words_headline.head()
words_count_headline=words_headline.groupby(by=['all_words_headline'])['all_words_headline'].agg({"count":np.size})
words_count_headline=words_count_headline.reset_index().sort_values(by=["count"],ascending=False)
words_count_headline.head()
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

wordcloud=WordCloud(font_path="./data/simhei.ttf",background_color="white",max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
WNlemma=nltk.WordNetLemmatizer()
summary_word3=data['summary_word'].tolist()
print data['summary_word'].iloc[3]
summary_word4=[]
for line in summary_word3:
    line_clean=[]
    for w in line:
        w=unicode(w, errors='replace')
        words=WNlemma.lemmatize(w)
        line_clean.append(words)
    summary_word4.append(line_clean)
print summary_word4[0:3]
headline_word3=data['headline_word'].tolist()
#print data['summary_word'].iloc[3]
headline_word4=[]
for line in headline_word3:
    line_clean=[]
    for w in line:
        w=unicode(w, errors='replace')
        words=WNlemma.lemmatize(w)
        line_clean.append(words)
    headline_word4.append(line_clean)
print headline_word4[0:3]
pairs=list()
for line in data['summary_meaning']:
    for i in range(len(line)-1):
        pair=line[i:i+2]
        pairs.append(pair)
for pair in pairs:
    print ' '.join(pair)
import sys
pairs_index = dict()

for line in data['summary_meaning']:
    for i in range(len(line)-1):
        pair=tuple(line[i:i+2])
        if pair in pairs_index:
            pairs_index[pair]+=1
        else:
            pairs_index[pair]=1
for pair in pairs_index.keys():
    count=pairs_index[pair]
    if count>1:
        print ' '.join(pair)+' '+str(count)
def count_ngrams(tokens,n):
    ngrams=dict()
    if len(tokens)<n:
        return ngrams
    for i in range(len(tokens)-n+1):
        ngram=tuple(tokens[i:i+1])
        if ngram not in ngrams:
            ngrams[ngram]=0
        else:
            ngrams[ngram] +=1
    return ngrams
def m_counts(ngrams_list):
    merged=dict()
    for ngrams in ngrams_list:
        for key, val in ngrams.iteritems():
            if key not in merged:
                merged[key] = 0
            else:
                merged[key] += val
    return merged
if __name__ == '__main__':
    import sys
    n = sys.argv[1]
    counts = list()
    for line in data['summary_meaning']:
        counts.append(count_ngrams(list(line), 3))
    combined_counts = m_counts(counts)
    sorted_counts = sorted(combined_counts.items(), key=lambda x: x[1],reverse=True)
    for key, val in sorted_counts:
        print ''.join(key) + ": " + str(val)
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
count_ngram=[]
for line in data['summary_meaning']:
    n_grams = ngrams(line,3)
    count=Counter(n_grams)
    count_ngram.append(count)
print count_ngram
from nltk.collocations import *
bigram_measures=nltk.collocations.BigramAssocMeasures()
summary_list=data['summary_meaning'].tolist()
summary_list[0]
all_words_list=[]
for line in summary_list:
    for w in line:
        all_words_list.append(w)
all_words_list
finder=BigramCollocationFinder.from_words(all_words_list)
finder.nbest(bigram_measures.pmi,10)
import gensim
from gensim import corpora, models, similarities
dictionary = corpora.Dictionary(summary_list)
corpus = [dictionary.doc2bow(sentence) for sentence in summary_list]
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)
print (lda.print_topic(2, topn=5))
for topic in lda.print_topics(num_topics=5, num_words=5):
    print (topic[1])
```




