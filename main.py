# -*- coding: utf-8 -*-
pip install nltk

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from google.colab import drive
drive.mount('/content/drive')

# 匯入imdb的資料
import numpy as np
import pandas as pd
imdb = r'/content/drive/MyDrive/Colab Notebooks/IMDB Dataset.csv'
data = pd.read_csv(imdb)
print(data.head())

"""# 新增區段"""

# 自己建字典
  # 先找使用頻率高的字詞
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))  # stopwords
string.punctuation  # 標點符號

review = data['review'].values
sentiment = data['sentiment'].values
all_words = []
sentences = []
for i in range(len(review)):
    token = word_tokenize(review[i])
    a_sentence = []
    for word in token:
        if (word not in stop_words) and (word not in string.punctuation) and (word != 'br'):
            all_words.append(word)
            a_sentence.append(word)
    sentences.append(a_sentence)
bag_of_words = Counter(all_words)
print(bag_of_words.most_common(100))
print(sentences[0])

print(sentences[:2])

max_unique_tokens = 10000
w2v_dimension = 300
max_DocLen = 100
most_common_words = bag_of_words.most_common(max_unique_tokens)
  # 建常用辭彙的字典-->我用list儲存
dict = []
frequency = []
for i in range(len(most_common_words)):
    dict.append(most_common_words[i][0])
    frequency.append(i + 1)
print(dict)

path = r'/content/drive/MyDrive/Colab Notebooks/metrix_embedding.csv'
f = open(path, 'r', encoding = 'utf_8')
matrix_lines = f.readlines()
for i in range(len(matrix_lines)):
    matrix_lines[i] = matrix_lines[i].split(',')
    for number in matrix_lines[i]:
      number = float(number)
    matrix_lines[i] = np.array(matrix_lines[i])
matrix_lines = np.array(matrix_lines)
print(matrix_lines.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add(Embedding(max_unique_tokens, w2v_dimension, input_length=max_DocLen))
model.add(Flatten())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.layers[0].set_weights([matrix_lines])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])


model.summary()

# 處理輸入文件
  # 1.評論裡面有哪些字在字典裏面,這個要跑40分鐘...，測試階段我先改200就好了...
data = []
for i in range(len(review)):
    token = word_tokenize(review[i])
    serial_number = []
    for word in token:
      if word in dict:
        serial_number.append(dict.index(word))
    data.append(serial_number)

print(data[0])
for i in data[0]:
  print(dict[i])

from keras.utils import pad_sequences
from keras import preprocessing
data = pad_sequences(data, maxlen=max_DocLen, truncating='post', padding='post')
print(data[1])

# 處理label, postive = 1, negative = 0
label = []
for i in sentiment:
    if i == 'positive':
        label.append(1)
    else:
        label.append(0)
label = np.array(label)

# 切割train/test
train_data = data[:25000]
test_data = data[25000:]

train_labels = label[:25000]
test_labels = label[25000:]


print(type(train_data))
print(type(train_labels))

# 執行囉
history = model.fit(train_data, train_labels, epochs=4, batch_size = 256)

testing_result = model.evaluate(test_data, test_labels)

# 可以把評論拿進來處理了~
  # 上面都執行完畢了 -->準確率大概在75%到80%中間

"""# 下面處理飯店資料

"""

import pandas as pd

# 讀取並整理資料格式
df = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/reviews.csv")  # 我這邊用的是芝加哥評論的資料(可以改用其他地方)
df2 = df.drop(["listing_id", "id", "reviewer_id", "reviewer_name", "date"], axis=1)  # 只有comments的
df2

# 偵測語言
!pip install langdetect
from langdetect import detect
from langdetect import detect_langs
from langdetect import DetectorFactory
DetectorFactory.seed = 0

docs = []
for i in range(len(df2["comments"])):  # 因為資料蠻多筆的所以大概要跑30幾分鐘，看到時候要不要把範圍縮小或者改地區的資料，不改也可
    comment = df2["comments"][i]
    try:
        if detect(comment) == "en":  # 如果是英文資料
            docs.append(comment)
    except:
        pass

docs[0:10]

import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))  # stopwords
string.punctuation  # 標點符號

# 將評論句子切成單字，並且去掉stop_words, punctuation, 'br'後存成list
airbnb_sentences = []
all_words = []
for i in range(len(docs)):
    token = word_tokenize(docs[i])
    a_sentence = []
    for word in token:
        if (word not in stop_words) and (word not in string.punctuation) and (word != 'br'):
            all_words.append(word)
            a_sentence.append(word)
    airbnb_sentences.append(a_sentence)
print(airbnb_sentences[0:2])

# 將剛剛的單字list對照先前建好的dict轉成word ID sequence 這個要10幾20分鐘
airbnb_data = []
airbnb_serial_number = []
for i in range(len(docs)):
    airbnb_token = (airbnb_sentences[i])
    airbnb_serial_number = []
    for word in airbnb_token:
        if word in dict:
            airbnb_serial_number.append(dict.index(word))
    airbnb_data.append(airbnb_serial_number)
print(airbnb_data[0:1])

from keras.utils import pad_sequences
from keras import preprocessing
airbnb_data = pad_sequences(airbnb_data, maxlen=max_DocLen, truncating='post', padding='post')

predict_output = model.predict(airbnb_data)
print(predict_output)

# 如果要跑這邊以下的程式可以直接帶現成資料進去
positive_review = []
negative_review = []
observe = []  # 拿來比較參數改變後，改變分類的評論有哪些(如果沒有要測數值的話就放著他也不影響結果)
boundry = 0.5  # 直接在這裡改變參數吧
for j in range(len(predict_output)):
  predict_num = predict_output[j]
  if predict_num >= boundry:
    positive_review.append(docs[j])
    if predict_num < 0.13:  # 填入比較之測試參數
      observe.append(docs[j])
  else:
    negative_review.append(docs[j])

negative_review

len(observe)
observe

len(negative_review)

len(positive_review)

"""這邊以下做cluster處理  --> 正面負面都要"""

# TFIDF  --> 正面的 
  # 文件還沒帶，我先隨便放，測試一下-->正常要把review改成飯店資料(正向的)
from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF_vectorizer = TfidfVectorizer(min_df = 2, stop_words = 'english')
TFIDF_vectors = TFIDF_vectorizer.fit_transform(positive_review)
print(TFIDF_vectors[0])

from sklearn.cluster import KMeans

n_cluster = 7  # 參數再調
cost = []
for i in range(2,n_cluster):
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(TFIDF_vectors)
  cost.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(cost,'bx-')
plt.show()

# 最後n -->我先亂放

final_n_clusters = 5
final_kmeans = KMeans(final_n_clusters)
final_kmeans.fit(TFIDF_vectors)

order_centroids = final_kmeans.cluster_centers_.argsort()[:,::-1]

for i in range(final_n_clusters):
  print("\n\nCluster {} :".format(i))
  for ind in order_centroids[i, :15]:
    print(TFIDF_vectorizer.get_feature_names()[ind])

"""這邊以下做負面的"""

from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF_vectorizer = TfidfVectorizer(min_df = 2, stop_words = 'english')
TFIDF_vectors = TFIDF_vectorizer.fit_transform(negative_review)
print(TFIDF_vectors[0])

n_cluster = 7  # 參數再調
cost = []
for i in range(2,n_cluster):
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(TFIDF_vectors)
  cost.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(cost,'bx-')
plt.show()

final_n_clusters = 5
final_kmeans = KMeans(final_n_clusters)
final_kmeans.fit(TFIDF_vectors)

order_centroids = final_kmeans.cluster_centers_.argsort()[:,::-1]

for i in range(final_n_clusters):
  print("\n\nCluster {} :".format(i))
  for ind in order_centroids[i, :15]:
    print(TFIDF_vectorizer.get_feature_names()[ind])

selected_negative_voc = []
for i in selected_negative_voc:
