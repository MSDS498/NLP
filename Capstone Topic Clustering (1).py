#!/usr/bin/env python
# coding: utf-8

# In[84]:


import nltk
nltk.download('stopwords')
#stopwords=stopwords.words('Portuguese')


# In[145]:


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#conda install gensim
import gensim
from gensim import corpora, models


# In[110]:


import csv
import pandas as pd
import re
import string


# In[332]:


#Upload data
reviews_df=pd.read_csv('/Users/neeramajumdar/Desktop/Capstone/Data/olist_order_reviews_dataset.csv')


# In[333]:


reviews_df.head()


# In[334]:


reviews_df['review_comment_message'][0]


# In[14]:


#Count how many null values
NoReviews=reviews_df['review_comment_message'].isnull().sum(axis=0)
NoReviews
#There are 58,247 customer that did not provide a review; only 41,753 out of 100,001 customers

ReviewsProvided=reviews_df['review_comment_message'].notnull().sum(axis=0)


# In[15]:


import matplotlib.pyplot as plt
names=['Reviews Provided','No Reviews']
values=[41753,58247]
plt.bar(names,values)
plt.show()


# In[16]:


reviews_df.info()


# In[335]:


#Create dataframe with just review_comment_message and then remove nulls from it for processing
allreviews_df=reviews_df['review_comment_message']

#remove nulls
allreviews_df.dropna

#Keep records after nulls are dropped (Save changes)
allreviews_df.dropna(inplace=True)


# In[336]:


pd.DataFrame(allreviews_df)


# In[188]:


allreviews_df.notnull().sum(axis=0)


# In[337]:


import string
def remove_punctuation(text):
    no_punct ="".join([c for c in text if c not in string.punctuation])
    return no_punct


# In[338]:


allreviews_df=allreviews_df.apply(lambda x: remove_punctuation(x))


# In[339]:


allreviews_df.head()


# In[340]:


#Tokenize

tokenizer=RegexpTokenizer(r'\w+')
allreviews_df=allreviews_df.apply(lambda x: tokenizer.tokenize(x.lower()))


# In[341]:


#Take out stopwords
#stopwords=set(stopwords.words('Portuguese'))
#text = [[x for x in text if x not in stopwords] for document in allreviews_df]

def remove_stopwords(text):
    words=[w for w in text if w not in stopwords.words('Portuguese')]
    return words
allreviews_df=allreviews_df.apply(lambda x: remove_stopwords(x))


# In[342]:


allreviews_df


# In[114]:


#texts = [[word for word in document.lower().split() if word not in stopwords] for document in allreviews_df]


# In[195]:


#Take out less frequent words
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
         frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for document in allreviews_df]

#Remove numbers  ## this doesnt work because the numbers are treated like text

#allreviews_df = [word for word in text if not any(c.isdigit() for c in word)]


# In[212]:


allreviews_df.to_excel(r'/Users/neeramajumdar/Desktop/Capstone/Data/Book1.xlsx')


# In[78]:


#######Decided not to do stemming so we don't lose the root of the word since we are unfamiliar with Portuguese
#Stemming
#from nltk.stem import RSLPStemmer
#stemmer=nltk.stem.RSLPStemmer()
#def word_stemmer(text):
#    stem_text="".join([stemmer.stem(i) for i in text])
#    return stem_text
#allreviews_df=allreviews_df.apply(lambda x: word_stemmer(x))


# In[197]:


#convert df to list

allreviews=[]

allreviews.append(allreviews_df)


# In[319]:


allreviews


# In[200]:


#Text to Dictionary


dictionary=corpora.Dictionary(allreviews_df)
print(dictionary)


# In[202]:


#count the # of occurences  of each distinct word and convert word to integer with doc2bow (like tf-idf)
corpus=[dictionary.doc2bow(text) for text in allreviews_df]


# In[203]:


corpus


# In[204]:


#LDA Modeling (Topic Modeling)
from gensim.models import LdaModel
NUM_TOPICS=10
ldamodel=LdaModel(corpus,num_topics=NUM_TOPICS,id2word=dictionary,passes=50)


# In[205]:


topics=ldamodel.show_topics()
for topic in topics:
    print(topics)


# In[206]:


word_dict = {};
for i in range (NUM_TOPICS):
    words=ldamodel.show_topic(i,topn=10)
    word_dict['Topic #'+ '{:02d}'.format(i+1)]=[i[0] for i in words]
top10=pd.DataFrame(word_dict)
top10


# In[207]:


top10.to_excel(r'/Users/neeramajumdar/Desktop/Capstone/Data/allreviewsexport.xlsx')


# In[425]:


#Visualization
import pyLDAvis.gensim
lda_display=pyLDAvis.gensim.prepare(ldamodel,corpus,dictionary,sort_topics=False)
pyLDAvis.display(lda_display)


# In[85]:


conda install -c conda-forge pyldavis


# In[272]:


##Analysis on Reviews & Review Score

#Extract review_score and review_comment_message

review_w_score_df=reviews_df[['review_score','review_comment_message']]


# In[273]:


review_w_score_df


# In[274]:


#Add column for good review or bad review. Anything less than 4 is considered bad review (0) and anything
#greater than 4 is considered good review

review_w_score_df['is_bad_review']=review_w_score_df['review_score'].apply(lambda x:1 if x >= 4 else 0)


# In[275]:


review_w_score_df


# In[276]:


#Fill in null values in review_comment_message
review_w_score_df['review_comment_message'].fillna('', inplace=True)


# In[277]:


review_w_score_df


# In[278]:


tokenizer=RegexpTokenizer(r'\w+')
review_w_score_df['review_comment_message']=review_w_score_df['review_comment_message'].apply(lambda x: tokenizer.tokenize(x.lower()))


# In[279]:


review_w_score_df


# In[280]:


#Clean up this dataframe

def remove_stopwords(text):
    words=[w for w in text if w not in stopwords.words('Portuguese')]
    return words
review_w_score_df=review_w_score_df.apply(lambda x: remove_stopwords(x))

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
         frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for document in review_w_score_df]


# In[281]:


review_w_score_df


# In[282]:


#Positive and Negative Review Score distribution
review_w_score_df['is_bad_review'].value_counts(normalize=True)


# In[248]:


#Word Cloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud (used top 10 clusters generated earlier)
show_wordcloud(top10)


# In[304]:


#After splitting text in each row in review_comment_message, the words are in a list. This needs to be converted
#back to string to apply other functions

review_w_score_df['review_comment_message']=review_w_score_df['review_comment_message'].apply(lambda x:','.join(str(x)))


# In[306]:


# add sentiment anaylsis columns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
review_w_score_df['sentiments'] = review_w_score_df['review_comment_message'].apply(lambda x: sid.polarity_scores(x))
review_w_score_df = pd.concat([review_w_score_df.drop(['sentiments'], axis=1), review_w_score_df['sentiments'].apply(pd.Series)], axis=1)


#####This was done to create polarity scores but it did not work out since the text was in portguese.


# In[316]:


review_w_score_df


# In[309]:


# add number of characters column
review_w_score_df['nb_chars'] = review_w_score_df['review_comment_message'].apply(lambda x: len(x))

# add number of words column
review_w_score_df['nb_words'] = review_w_score_df['review_comment_message'].apply(lambda x: len(x.split(',')))


# In[424]:


#TFIDF Model from corpus created in the beginning
tfidf=models.TfidfModel(corpus)
corpus_tfidf=tfidf[corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    


# In[ ]:




