
#streamlit run app.py

import streamlit as st
import sys
import os

# my_path = os.getcwd()
# my_path = my_path.split('/')[:-1]
# my_path = "/".join(my_path)
# sys.path.insert(0, my_path)

from src.data_objects import *
import json
import pandas as pd
import nltk
import numpy as np
from more_itertools import split_after

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk import Tree
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

import pandas_gpt
import openai

st.set_page_config(
    page_title="Insync Analytics GPT App",
    page_icon="🧊",
    layout="wide",
)

with open(f"data_sources.json") as f:
    data = f.read()

js = json.loads(data)
key = js["openai"]['api_key']

if key == '':
    print(f"get an api key at https://platform.openai.com/account/api-keys")
else:
    openai.api_key = key

focus_list = ["ADBE", "AMZN", "LMT", "LOW", "ORLY"]
pm = Portfolio(focus_list)
pm.get_fundamental_data()
df = pm.fundamentals

def clean_df(df):

    replacers = {'Q/Q': 'quarter over quarter growth in ',
    'Y/Y': 'year over year growth in ',
    '(Y/Y)':'year over year growth in',
    'TTM': 'trailing twelve months',
    '/': ' per ',
    'Incr.' : 'incremental',
    '$': 'dollars of ',
    "IS":'income statement',
    'BS':'balance sheet',
    'KM':'key metrics'}

    df['filing_date']=pd.to_datetime(df['filing_date'])

    df['lineitem'] = (
        df.lineitem.str.replace('[...…]','')
        .str.split()
        .apply(lambda x: ' '.join([replacers.get(e, e) for e in x]))
    )
    df['category'] = (
        df.category.str.replace('[...…]','')
        .str.split()
        .apply(lambda x: ' '.join([replacers.get(e, e) for e in x]))
    )

    df = df[['ticker', 'year', 'quarter',
       'numberonly', 'category',
       'lineitem','filing_date']]
    
    def get_noun_phrases(text):
        blob = TextBlob(text).noun_phrases
        return ','.join(blob)

    df['key_phrases'] = df['lineitem'].apply(lambda sent: get_noun_phrases((sent)))
    
    return df

df = clean_df(df)

st.title('Dataframe Prompt Query')

prompt_string  = st.text_input("Enter prompt:")

if prompt_string == "":
    df = df.head()
else:
    df = df.ask(prompt_string)

df['year'] = df['year'].astype(str)

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


csv = convert_df(df)

st.download_button(
   "Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)

st.dataframe(df,hide_index=True)  # Same as st.write(df)
#column_config={"year": st.column_config.NumberColumn(format="{0:.0f}")}