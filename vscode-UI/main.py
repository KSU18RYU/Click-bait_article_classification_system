import pandas as pd
import json
import glob
import os
import konlpy
from konlpy.tag import Okt #형태소 추출
import re # regular expressions
from collections import Counter

import numpy as np
import tensorflow as tf
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout, LSTM, Bidirectional, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from function import text_prepro, data_return, data_return_title_content, model_predict

import streamlit as st
from io import StringIO

st.markdown("<h1 style='text-align: center;'>낚시성 기사 분류 시스템</h1>", unsafe_allow_html=True)

number = st.number_input('분류할 기사의 수를 입력해주세요. (최대10개)', max_value= 10, min_value=1, value=1, step=1)

cols = st.columns(number)

for i in range(0, number):
    globals()[f'title{i}'] = cols[i].text_input(f'{i+1}번 기사 제목')
    globals()[f'content{i}'] = cols[i].text_area(f'{i+1}번 기사 본문')

if st.button('분류 시작'):
    for i in range(0, number):
        globals()[f'data{i}'] = data_return(globals()[f'title{i}'], globals()[f'content{i}'])
        if model_predict(globals()[f'data{i}']) == 1:
            cols[i].write(f"{i+1}번 기사 분류 결과: 일반 기사입니다.")
        else :
            cols[i].write(f"{i+1}번 기사 분류 결과: 낚시성 기사입니다.")

uploaded_file = st.file_uploader("분류할 기사의 텍스트 파일을 업로드해주세요.")
if uploaded_file is not None:
    # # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)
    string_data = data_return_title_content(string_data)

if st.button('텍스트 파일 분류'):
    if model_predict(string_data) == 1:
        st.write("분류 결과: 일반 기사입니다.")
    else :
        st.write("분류 결과: 낚시성 기사입니다.")
# streamlit run main.py  #cmd에 입력 #ctrl + c 종료

