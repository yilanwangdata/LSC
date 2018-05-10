### Use WIKI to build our word vector
Download the WIKI Chinese 
Transform the XML file to txt
```
import logging
import os.path
import sys
from gensim.corpora import WikiCorpus
if __name__ == '__main__':
    
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 3:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = b' '
    i = 0
    output = open(outp, 'w',encoding='utf-8')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        s=space.join(text)
        s=s.decode('utf8') + "\n"
        output.write(s)
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
#python process.py zhwiki-latest-pages-articles.xml.bz2 wiki.zh.text
```
the last sentence is to let the command to transform the download version the txt version, using the name we want

use opencc to transform the Tradition to Simple
```
opencc -i wiki_texts.txt -o test.txt -c t2s.json
```

Using Jieba to split Chinese words
```
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs,sys
def cut_words(sentence):
    #print sentence
    return " ".join(jieba.cut(sentence)).encode('utf-8')
f=codecs.open('wiki.zh.jian.text','r',encoding="utf8")
target = codecs.open("zh.jian.wiki.seperate.txt", 'w',encoding="utf8")
print ('open files')
line_num=1
line = f.readline()
while line:
    print('---- processing ', line_num, ' article----------------')
    line_seg = " ".join(jieba.cut(line))
    target.writelines(line_seg)
    line_num = line_num + 1
    line = f.readline()
f.close()
target.close()
exit()
while line:
    curr = []
    for oneline in line:
        #print(oneline)
        curr.append(oneline)
    after_cut = map(cut_words, curr)
    target.writelines(after_cut)
    print ('saved',line_num,'articles')
    exit()
    line = f.readline1()
f.close()
target.close()

python Testjieba.py
```

Use Word2vec to create a model to get word vector
```
import logging
import os.path
import sys
import multiprocessing
from gensim.corpora import WikiCorpus
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
#python word2vec_model.py zh.jian.wiki.seperate.txt wiki.zh.text.model wiki.zh.text.vector
```
Choose any words you want to test the model
```
from gensim.models import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('wiki.zh.text.model')

testwords = ['word1','word2','word3','word4','word5']
for i in range(5):
    res = en_wiki_word2vec_model.most_similar(testwords[i])
    print (testwords[i])
    print (res)
```
use the model we build already to get the vector
```
df = load_dataset('labeled_train')
df.head()
def to_review_vector(review):
    words = clean_text(review, remove_stopwords=True)
    array = np.array([model[w] for w in words if w in model])
    return pd.Series(array.mean(axis=0))
train_data_features = df.review.apply(to_review_vector)
train_data_features.head()
forest = RandomForestClassifier(n_estimators = 100, random_state=0)
forest = forest.fit(train_data_features, df.label)
test_data_features = df.review.apply(to_review_vector)
test_data_features.head()

```
Cluster
```
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
test_data_features = df.review.apply(make_cluster_bag)
test_data_features.head()
```
use  CountVectorizer
```
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)
words = []
for line_index in range(len(x_train)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))
    except:
        print (line_index,word_index)
words[0] 
text=[]
from sklearn.feature_extraction.text import CountVectorizer
for i in len(words):

    texts=words[i]
    text.append(texts)
cv = CountVectorizer()
cv_fit=cv.fit_transform(text)

cv = CountVectorizer(ngram_range=(1,4))
cv_fit=cv.fit_transform(texts)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)
```
Tfidf
```
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vectorizer.fit(words)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)
```
