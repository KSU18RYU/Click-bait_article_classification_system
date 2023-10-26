import re
import tensorflow as tf
from konlpy.tag import Okt #형태소 추출
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def text_prepro(text):
    sentences = re.split(r'[\.\?\!]+', text)
    # 한글만 추출 (한글이 아닌 케이스를 패턴으로 컴파일)
    filter_ = re.compile('[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+')
    
    # 문장 단위로 추출
    filtered_sentences = []
    for sentence in sentences:
        filtered_sentences.append(filter_.sub(' ', sentence))
    
    # # 불용어 리스트
    stopwords = ['의', '부', '일', '면서', '차', '그간', '만', '약', '년', '전', '얼마나', '그', '등', '또', '것', '며', '를', '때문', '라며', '당시',
                '모든', '이', '측은', '수', '번', '개', '고', '아야', '월', '지난', '내년', '앞', '료', '건', '스', '때', '더', '회', '분', '날', '쪽',
                '잔', '자', '좀', '각', '간', '초', '말', '경우', '곳', '중', '내', '관련', '바로', '한편', '앞서', '이번', '자기', '억', '주로',
                '뒤', '낼', '및', '조', '중', '위해', '억원', '확', '앞서', '바', '다', '계', '첫', '사', '로', '현', '회', '각각', '률', '폭',
                '처', '측', '단', '뚝', '기', '호', '속', '건', '두', '코', '로', '한편', '걸', '안', '이번', '류', '데', '것이므', '례', '과', '로서',
                '율', '온', '즉', '익', '여기', '와', '는', '역시', '다해', '끼', '만큼', '세', '매우', '천', '꽤', '어가', '중이', '거나', '해',
                '기', '여개', '제', '다다', '나', '네', '형', '므', '민', '주', '기', '위핸', '타', '채', '설', '단', '팅', '더', '번', '심지어', 
                '것일', '뿐', '온', '유', '대해', '서도', '켜', '간의']
    
    # # okt를 활용하여 명사만 추출하고, 불용어는 제거하여 리스트 생성
    okt = Okt()
    
    # # 불용어는 제외한 명사 리스트
    cleaned_noun_list = []
    
    # # 불용어 제거한 명사로 이루어진 문장 리스트
    cleaned_sentence_list = []
    for sentence in filtered_sentences:
        nouns = okt.nouns(sentence)
        clean_contents = [noun for noun in nouns if not noun in stopwords]
        cleaned_noun_list.extend(clean_contents)
        # cleaned_sentence_list.append(clean_contents)

    return cleaned_noun_list

def data_return(title, content): # 제목과 내용 입력
    data = pd.DataFrame()
    title_content = title + content
    data.loc[0, 'title_content'] = title_content
    
    data.title_content[0] = text_prepro(data.title_content[0])
    return data

def data_return_title_content(title_content): # 제목+내용 입력 ## 오버라이딩
    data = pd.DataFrame()
    data.loc[0, 'title_content'] = title_content
    
    data.title_content[0] = text_prepro(data.title_content[0])
    return data

def model_predict(data):
    with open('tokenizer_fake_content.pickle', 'rb') as handle:
        token = pickle.load(handle)
    model = tf.keras.models.load_model('val_evaluate_0.90_model.h5')

    re_docs = data.title_content
    rx = token.texts_to_sequences(re_docs)
    re_padded_x = pad_sequences(rx, maxlen = 330, padding='post', truncating='post')

    predictions = model.predict(re_padded_x)
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1
    return predictions[0].astype(np.int32)