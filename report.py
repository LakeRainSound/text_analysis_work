# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 情報システム論実習課題 テキスト分析
#
# 生物圏情報学講座M1  
# 6930323723  
# 山本凱  
#
# ## 利用する文書
# 様々な国のWikipediaにおけるabstractを取り出したデータセット．
#
# ## 利用する表現手法
#
# ここでは利用する表現手法として，BoWとTF-IDFを利用する．
#
# ## 利用する距離
# BoWについては，マンハッタン距離とユークリッド距離，コサイン類似度によって計算する．
# TF-IDFについては，コサイン類似度を計算する．

# ## 実際の処理

# ### 必要なパッケージのインストール

# !pip install -q nltk
# !pip install -q gensim

# ### 必要なパッケージのインポート・ダウンロード

# +
import nltk
import numpy as np
import pandas as pd
import re
from nltk.corpus import wordnet as wn #lemmatize関数のためのimport

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
# -

# ### データのデータフレームへの格納

df = pd.read_csv("./nlp_country.csv")
df

# ### 前処理

# stop wordsで冠詞、代名詞、前置詞など一般的な語を削除するための関数を定義する．今回は英語であるので英語のstop wordsのリストを利用する．

en_stop = nltk.corpus.stopwords.words('english')
def remove_stopwords(word, stopwordset):
  if word in stopwordset:
    return None
  else:
    return word


# 具体的に，前処理をしていく．

# +
def preprocessing_text(text):
  def cleaning_text(text):
    # @の削除
    pattern1 = '@|%'
    text = re.sub(pattern1, '', text)    
    pattern2 = '\[[0-9 ]*\]'
    text = re.sub(pattern2, '', text)    
    # <b>タグの削除
    pattern3 = '\([a-z ]*\)'
    text = re.sub(pattern3, '', text)    
    pattern4 = '[0-9]'
    text = re.sub(pattern4, '', text)
    return text
  
  def tokenize_text(text):
    text = re.sub('[.,]', '', text)
    return text.split()

  def lemmatize_word(word):
    # make words lower  example: Python =>python
    word=word.lower()
    
    # lemmatize  example: cooked=>cook
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
      return lemma
    
  text = cleaning_text(text)
  tokens = tokenize_text(text)
  tokens = [lemmatize_word(word) for word in tokens]
  tokens = [remove_stopwords(word, en_stop) for word in tokens]
  tokens = [word for word in tokens if word is not None]
  return tokens
  
docs = df["Abstract"].values
pp_docs = [preprocessing_text(text) for text in docs]


# -

# ### 前処理したドキュメントで計算
# 前処理したdocumentを用いて，BoWとTF-IDFを計算する．

# #### Cosine類似度を計算する関数の定義

def cosine_similarity(list_a, list_b):
  inner_prod = np.array(list_a).dot(np.array(list_b))
  norm_a = np.linalg.norm(list_a)
  norm_b = np.linalg.norm(list_b)
  try:
      return inner_prod / (norm_a*norm_b)
  except ZeroDivisionError:
      return 1.0


# #### ミンコフスキー距離を計算する関数の定義

def minkowski_distance(list_a, list_b, p):
  diff_vec = np.array(list_a) - np.array(list_b)
  return np.linalg.norm(x=diff_vec, ord=p)


# ### Bag of Words
# まずはBag of Wordsから計算する．bowのベクトルを計算する関数を定義する．

def bow_vectorizer(docs):
  word2id = {}
  for doc in docs:
    for w in doc:
      if w not in word2id:
        word2id[w] = len(word2id)
        
  result_list = []
  for doc in docs:
    doc_vec = [0] * len(word2id)
    for w in doc:
      doc_vec[word2id[w]] += 1
    result_list.append(doc_vec)
  return result_list, word2id


# 実際に計算する．

bow_vec, word2id = bow_vectorizer(pp_docs)


# 今回は日本のドキュメントと最もよく似ている国を示す．まず，コサイン類似度による計算を行う．

# +
def calc_cosine(vector, vector_list):
  result = {}
  for i, x in enumerate(vector_list):
    result[i] = cosine_similarity(vector, vector_list[i])
    
  return result

print("BoW: cosine類似度")
res = calc_cosine(bow_vec[0],bow_vec)
res


# -

# 結果から日本を除いて最も数値が大きいのは14の国であり，その国はインドネシアである．

# 次に日本のドキュメントと各ドキュメントのミンコフスキー距離を計算するための関数を示す．
# 今回はミンコフスキー距離の$p$は，$p=1$， $p=2$の2通りで計算する．

def calc_minkowski(vector, vector_list, p):
    result = {}
    result = {}
    for i, x in enumerate(vector_list):
        result[i] = minkowski_distance(vector, x, p)
    
    return result


# マンハッタン距離の場合($p=1$)を示す．

print("BoW: マンハッタン距離")
res = ｃalc_minkowski(bow_vec[0],bow_vec, 1)
res   

# 日本を除くマンハッタン距離が最も小さいのは4でありその国はインドである．

# 次にユークリッド距離の場合を示す．

print("BoW: ユークリッド距離")
res = calc_minkowski(bow_vec[0],bow_vec, 2)
res   


# 日本を除くユークリッド距離が最も小さいのは4でありその国はインドである．

# ### 考察

# 結果から，ユークリッド距離とcosine類似度の結果が一致しており，マンハッタン距離が異なるということが分かる．
# これはマンハッタン距離は単純な出現文字数の差を取る一方で，その二乗和をとってルートを取るという点で異なることからこのような結果の違いとして現れたと考える．また，cosine類似度はそのベクトルの角度が近いすなわち，文書の内容の方向性が近いものが抽出できる．こうした計算方法の違いで実際に近い文書として出力されることが結果からわかった．

# ### TF-IDF
# つぎにTF-IDFを計算する．TF-IDFのベクトルを計算する関数を定義する．

def tfidf_vectorizer(docs):
  def tf(word2id, doc):
    term_counts = np.zeros(len(word2id))
    for term in word2id.keys():
      term_counts[word2id[term]] = doc.count(term)
    tf_values = list(map(lambda x: x/sum(term_counts), term_counts))
    return tf_values
  
  def idf(word2id, docs):
    idf = np.zeros(len(word2id))
    for term in word2id.keys():
      idf[word2id[term]] = np.log(len(docs) / sum([bool(term in doc) for doc in docs]))
    return idf
  
  word2id = {}
  for doc in docs:
    for w in doc:
      if w not in word2id:
        word2id[w] = len(word2id)
  
  return [[_tf*_idf for _tf, _idf in zip(tf(word2id, doc), idf(word2id, docs))] for doc in docs], word2id


# 実際に前処理をした文書で計算する．

tfidf_vec, word2id = tfidf_vectorizer(pp_docs)

# BoWと同様，日本の文書と似ている文書を見つける．まずは，cosine類似度の計算をする．

print("TF-IDF: cosine類似度")
res = calc_cosine(tfidf_vec[0],tfidf_vec)
res

# 結果から，最も近い文書は5の韓国であることが分かる．  
# 次に，マンハッタン距離を求める．

print("TF-IDF: マンハッタン距離")
res = ｃalc_minkowski(tfidf_vec[0],tfidf_vec, 1)
res   

# マンハッタン距離が最も小さいのは8のフランスであることが分かる．  
# 次に，ユークリッド距離を求める．

print("TF-IDF: ユークリッド距離")
res = ｃalc_minkowski(tfidf_vec[0],tfidf_vec, 2)
res   

# ユークリッド距離が最も小さいのは1のアメリカであることが分かる．  

# ### 考察
# TF-IDFの場合，結果が全ての距離尺度で異なる結果となった．また，BoWの結果と比較しても各尺度で一致している結果はなく，TF-IDFとBoWでは異なるアルゴリズムの結果が現れていると考えられる．一般的に用いられている尺度であるcosine類似度は韓国の文書が最も似ているという結果となった．TF-IDFは単語の出現頻度と，逆文書頻度を利用するため，単純な出現頻度を見る，BoWとは結果が異なると考える．また，利用する尺度や表現手法によってこうした結果の違いとして現れるため，利用する尺度は慎重に選定する必要があると考える．

# ## 感想
# 利用する尺度，表現手法によって違いが実際に現れることが分かった．また，TF-IDFやBoWはコードを見ると案外簡単にかけることがわかった．一方難しいと感じたのは，前処理の仕方であり，この部分がしっかりしていないと精度の高い文書比較は出来なさそうであると感じた．
