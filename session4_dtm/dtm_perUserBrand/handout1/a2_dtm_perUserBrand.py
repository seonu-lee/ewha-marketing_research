'''

'''

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


#=================================
# 공통 설정
#=================================
PATH_to_data = ""
PATH_to_save = ""

#=================================
# User, brand별 리뷰 -> dtm 만들기
#=================================
def create_dtm_perUserBrand(input_file_name, brand_slted, min_df_rate, max_df_rate, manual_stopwords):
    #=================================
    # 0. 데이터 불러오기
    #=================================
    ub_reviews = pd.read_csv(f"{PATH_to_data}/{input_file_name}.csv")

    ### 선택 브랜드에 대한 리뷰만 추출
    ub_reviews = ub_reviews[ub_reviews['name']==brand_slted]

    ### cleaning
    ub_reviews['pooled_text_clean'] = ub_reviews['pooled_text_clean'].fillna("")

    #============================================
    # 문서-단어 행렬 (document-term freq matrix) 생성
    #============================================
    ### 불용어 정의
    nltk_stop = stopwords.words('english')
    initial_manual_stopwords = [] # 제거할 stopping words 여기에 추가
    stop_words = nltk_stop + initial_manual_stopwords
    stop_words = sorted(list(set(stop_words)))

    ### CountVectorizer를 사용한 문서-단어 행렬 생성
    vectorizer = CountVectorizer(stop_words=stop_words) # CountVectorizer: 텍스트 데이터를 DTM으로 변환하는 클래스
    dtm = vectorizer.fit_transform(ub_reviews['pooled_text_clean']) # pooled_text_clean 컬럼으로 부터 DTM 생성
    terms = vectorizer.get_feature_names_out() # 단어이름 NumPy 배열 - 전체 단어 목록

    ### dtm 확인
    # dtm.shape 
    # 단어수가 매우 큰 희소행렬이이어서 df로 변환하면 파일사이즈가 너무 커짐. 메모리 초과로 다운될 수 있음.
    # dtm_df = pd.DataFrame(dtm_array, columns=terms) # DataFrame으로 구성, 50*11455, 5241*3042
    # dtm_df['doc_id'] = brand_reviews['doc_id']

    #============================================
    # 의미중복 통합
    #============================================
    # 희소 단어 제거 하기 전에 의미 중복 통합해야함.

    #============================================
    # 희소단어 제거
    #============================================

    #=================================
    # 2.1 전체 문서(브랜드)에서의 희소단어 제거 (공통단어도 제거하는 옵션 추가) - 비율 기준
    #=================================
    ### 1) 희소 단어 제거: 
    # 전체 문서에서 각 단어가 등장하는 문서 비율(document freq rate) 기준

    ## 각 단어의 등장 문서 비율 계산
    term_frequencies = np.asarray((dtm > 0).sum(axis=0)).flatten() # 각 단어가 등장한 문서 수 계산 (dtm > 0인 경우를 counting)
    doc_count = dtm.shape[0] # 전체 문서 수
    df_rate = term_frequencies / doc_count # document freq rate, 각 단어의 등장빈도 비율 계산 (등장 문서 수 / 전체 문서 수)

    word_mask = (df_rate >= min_df_rate) & (df_rate < max_df_rate) # 최소, 최대 df rate 기준값 적용하여 단어 필터링
    # sum(word_mask) # 필터링하고 남은 단어수

    ## 희소/공통 단어 제거된 DTM과 단어 배열 추출
    dtm_reduced = dtm[:, word_mask]  # 모든 행에 대해서 열(단어들)을 Boolean 조건으로 필터링 - 문서-단어 행렬(DTM)에서 조건을 만족하는 단어(열)만 남김
    terms_reduced = terms[word_mask] # 단어 이름 NumPy 배열 - 전체 단어 목록에서, 조건을 만족하는 단어만 골라서 terms_reduced에 저장

    ### 2) pandas DataFrame으로 변환 (문서 × 단어 형태)
    dtm_df = pd.DataFrame(dtm_reduced.toarray(), columns=terms_reduced) # 희소행렬을 밀집행렬로 변환하고 DataFrame으로 다시 변환
    dtm_df['doc_id'] = ub_reviews['doc_id'].values # 원래 문서 ID(doc_id)를 함께 붙이기

    ### 3) 제거된 단어 출력 (확인용)
    sparse_words_deleted = terms[df_rate < min_df_rate]
    common_words_deleted = terms[df_rate >= max_df_rate]
    print (f"제거된 희소단어(총 {len(sparse_words_deleted)}개): {sparse_words_deleted}") # 제거하는 것이 적절한지 검토할 것
    print (f"제거된 공통단어(총 {len(common_words_deleted)}개): {common_words_deleted}") # 제거하는 것이 적절한지 검토할 것

    #============================================
    # 불용어 제거 (수동)
    #============================================
    dtm_df_cleaned = dtm_df.copy()
    # dtm_df_cleaned.columns.to_list() # 키워드 리스트

    to_drop = [w for w in manual_stopwords if w in dtm_df_cleaned.columns] # manual_stopwords중 dtm 컬럼에 존재하는 단어만 추출
    dtm_df_cleaned = dtm_df_cleaned.drop(to_drop, axis=1)


    #============================================
    # 선정된 단어에 대해서 문서-단어 행렬 만들기 (값은 빈도수)
    #============================================
    # dtm_f = dtm_df[words_features].copy() # 최종 선정된 word 컬럼만 남기기  

    dtm_df_cleaned = dtm_df_cleaned.loc[~(dtm_df_cleaned == 0).all(axis=1)] # 모든 값이 0인 행 제거
    dtm_df_cleaned = dtm_df_cleaned.loc[:, ~(dtm_df_cleaned == 0).all(axis=0)] # 모든 값이 0인 열 제거


    #============================================
    # 최종 분석용 데이터 결합 및 저장 
    #============================================
    # 문서별 데이터에 dtm 붙이기
    ub_reviews_dtm = pd.merge(ub_reviews.drop(columns=['pooled_text_clean']), dtm_df_cleaned.reset_index(drop=True), on='doc_id', how='inner')

    # 저장
    ub_reviews_dtm = ub_reviews_dtm.set_index('doc_id')
    ub_reviews_dtm.to_csv(f"{PATH_to_save}/{input_file_name}_{brand_slted}_{min_df_rate}_{max_df_rate}_dtm.csv", index=False, encoding='utf-8-sig')

    return ub_reviews_dtm


if __name__ == "__main__":

    manual_stopwords = [
        'al', "also", "alway", 'anoth', 'area', 'around', 'ask', 
        'back', 'bite', 'box', 
        'come', 'could', 'came', 
        'dont', 'day', 'de', 'didnt', 
        'even', 'ever', 'el', 
        'get', 'give', 'got', 
        'im', 'ive', 
        'let', 'la', 'last', 
        'make', 'made', 'mayb', 
        'name', 
        'one', 
        'round', 
        'someth', 'still', 'seem', 'sinc', 'sub', 'said', 
        'told', 'that', 'think', 'two', 'though', 'thought', 'took', 
        'us', 
        'want', 'way', 'went', 'would', 'wasnt', 
        'your', 'year',
    ]

    # 조건 1
    input_file_name = "reviews_restaurants_az_perUserBrand"
    brand_slted = 'chipotlemexicangrill'
    min_df_rate = 0.05 # 최소 df(document freq) rate, 각 단어의 등장빈도 비율 (등장 문서 수 / 전체 문서 수)
    max_df_rate = 0.9 # 최대 df(document freq) rate, 각 단어의 등장빈도 비율 (등장 문서 수 / 전체 문서 수)

    ub_reviews_dtm = create_dtm_perUserBrand(input_file_name, brand_slted, min_df_rate, max_df_rate, manual_stopwords)




