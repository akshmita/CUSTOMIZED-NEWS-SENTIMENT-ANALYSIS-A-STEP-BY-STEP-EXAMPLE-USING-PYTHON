# CUSTOMIZED-NEWS-SENTIMENT-ANALYSIS-A-STEP-BY-STEP-EXAMPLE-USING-PYTHON
CUSTOMIZED NEWS SENTIMENT ANALYSIS: A STEP-BY-STEP EXAMPLE USING PYTHON
In this article, I will take you through a step-by-step process of building a foreboding index using a customized dictionary with python tools. The customized dictionary has 102 positive words representing foreboding, anxiousness, or uncertainty and 102 negative words representing antonyms of foreboding, anxiousness or uncertainty. The methodology for creating the dictionary is found in Roy Trivedi (2021). For TFI only the words "foreboding, uncertainty, fear, worry" were taken into account, which we will call as root words.
1. Let us now, jump to the coding, with the usual imports. Data used for developing the index is taken from Refinitiv, Eikon.
Input:
import eikon as ek
from collections import Counter
import pandas as pd
from pandas import DataFrame
import numpy as np
from bs4 import BeautifulSoup
from textblob import TextBlob
import datetime
from datetime import time
import warnings
warnings.filterwarnings("ignore")
ek.set_app_key('****')
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
import re

2. Now let's import the dictionary, the list of words and its classification:
Input:
import csv
labels = {}
with open('forebroding_dict.csv', mode='r') as inp:
reader = csv.reader(inp)
labels = {rows[0]:rows[1] for rows in reader}
print(labels)
Output: (part of output being shown)
{'achievement': 'negative', ….. 'apprehensive': 'positive',}

3. Next, we define a function to match text in the news to sentiment, in accordance to our dictionary:
Input:
def matcher(k):
x = (i for i in labels if i in k)
return ' | '.join(map(labels.get, x))

4. Get news from Refinitiv Eikon. I am taking intra-day news on Covid 19 (COVID), In English (LEN), sourced from Reuters (RTRS). See article on news sentiment analysis with eikon data Jason Ramchandani (2019).
Input:
df = ek.get_news_headlines('Topic:COVID AND Language:LEN AND Source:RTRS', date_from='2021–04–01T09:00:00',date_to='2021–04–02T09:00:00',count=50)
5. Next the news story has to be obtained using the get_news story function of eikon and we use beautiful soup to create a BeautifulSoup object from our HTML news article. We clean the text , tokenize and obtain the filtered words, and store it in column 'filtered words'. As we will be using regex, we are converting it to strings.
Input:
for idx, storyId in enumerate(df1['storyId'].values):
newsText = ek.get_news_story(storyId)
if newsText:
soup = BeautifulSoup(newsText,"lxml")
sentence = newsText.lower()
sentence=sentence.replace('{html}',"")
cleanr = re.compile('<.*?>')
cleantext = re.sub(cleanr, '', sentence)
rem_url=re.sub(r'http\S+', '',cleantext)
rem_num = re.sub('[0–9]+', '', rem_url)
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(rem_num)
filtered_words = [w for w in tokens if len(w) > 2 ]
df1['filtered_words'].iloc[idx] = filtered_words
df1['filtered_words'] = df1['filtered_words'].astype(str,'ignore')

6. Next we apply the function matcher, already defined, on words, and pick up words of our dictionary and the corresponding sentiment. In line 5, we find only the words in foreboding, uncertainty, fear, worry, which are our 'root words'.
Three sentiment indices are calculated. The Foreboding Index for a news report is calculated as
FI = Number of Positive Words / (Number of Negative + Positive Words) ….(1)
The TFI for a news report is calculated as
TFI= Frequency of Root Words / Total Number of Words ……..(2)
Both the Foreboding Index and TFI marks words related to foreboding as positive and therefore a higher FI and TFI represents greater foreboding. It may be noted that while FI can have a value between 0 (no positive words) to infinity (no positive or negative words), TFI takes a value between 0 (when none of the words appear in a news) to 1 (if only these words appear in a news). While TFI looks at whether foreboding sentiment appears in a news, FI sees it in the context of both positive and negative words relating to foreboding appearing giving a more nuanced sentiment classification.
Input:
df1['flag_list']=df1['filtered_words'].apply(matcher)
df1['flag_list'] = df1['flag_list'].astype(str,'ignore')
df1['number']=df1['filtered_words'].str.count(r'[a-z]')
df1['number2']=df1['flag_list'].str.count(r'[a-z]')
df1['fore']=df1['filtered_words'].str.findall(r'\'\b(forebroding|uncertianty|fear|worry)\b\'')
df1['fore'] = df1['fore'].astype(str,'ignore')
df1['count_flag']=df1['fore'].str.count(r'[a-z]')
df1['term_frequency']=df1['count_flag']/df1['number']
df1['flag_list'] = df1['flag_list'].astype(str,'ignore')
df1['count_flag_pos']=df1['flag_list'].str.findall(r'(pos)')
df1['count_flag_pos'] = df1['count_flag_pos'].astype(str,'ignore')
df1['count_flag_pos1']=df1['count_flag_pos'].str.count(r'[a-z]')
df1['count_flag_neg']=df1['flag_list'].str.findall(r'neg')
df1['count_flag_neg'] = df1['count_flag_neg'].astype(str,'ignore')
df1['count_flag_neg1']=df1['count_flag_neg'].str.count(r'[a-z]')
df1['sentiment']=(df1['count_flag_pos1']+1)/(df1['count_flag_neg1']+1)
df1['sentiment2']=(df1['count_flag_pos1'])/(df1['count_flag_neg1']+df1['count_flag_pos1'])
 
References:
· Jason Ramchandani (2019), News Sentiment Analysis with Eikon, https://www.refinitiv.com/perspectives/future-of-investing-trading/news-sentiment-analysis-with-eikon-data-apis/
· Regular Expression How to, https://docs.python.org/3/howto/regex.html#performing-matches
