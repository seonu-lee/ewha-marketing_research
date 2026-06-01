'''
1. dtm 
1.1 개념: 문석(리뷰) 속 단어(term)들의 등장횟수 또는 가중치를 행렬로 표현한 것. 행: 문서, 열: 단어
1.2 특징
    - 희소행렬
    - 단어 순서, 구조 무시 (bag of words)
1.3 용도
    - 텍스트를 수치 벡터로 변환해 연산, 기계학습에 사용
    - 단어 순서를 잃는 단점이 있으나, 계산 효율성과 해석 용이성이 높아, 자연어처리, 마케팅 분석의 출발점이 됨 

'''

import sys
from importlib import reload
sys.path.append('')
from s04_dtm.dtm_perBrand import a3_tfidf_perBrand as tfidf
reload(tfidf)


dtm_file_name='reviews_restaurants_az_perUserBrand_chipotlemexicangrill_0.05_0.9_dtm'
df_tfidf = tfidf.create_and_save_tfidf_perBrand(dtm_file_name=dtm_file_name, apply_l2=False)
df_tfidf_l2 = tfidf.create_and_save_tfidf_perBrand(dtm_file_name=dtm_file_name, apply_l2=True)
    
    
