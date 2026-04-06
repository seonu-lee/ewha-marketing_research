'''
1. Document-Term Matrix (DTM)
1.1 개념
    - 문서(리뷰) 속 단어(term)들의 등장횟수 또는 가중치를 행렬로 표현한 것 
    - 행(document)=문서, 열(term)=단어, 값=해단 단어의 빈도 또는 가중치
1.2 특징
    - 희소행렬 (대부분의 값이 0)
    - 단어 순서, 구조 무시 (bag of words)
1.3 용도
    - 텍스트(비정형)를 수치 벡터(정형)로 변환해 연산, 기계학습에 사용
    - 단어 순서를 잃는 단점이 있으나, 계산 효율성과 해석 용이성이 높아, 자연어처리, 마케팅 분석의 출발점이 됨 


2. 단어 분류
2.1 빈도 기준
2.1.1 희소 단어
    - 극히 일부 문서에만 등장
    - 제거 or 포함? 제거하는 이유? 포함하는 이유?
    - 최근 트렌드를 반영하는 단어
2.1.2 공통 단어
    - 대부분의 문서에 등장
    - 제거 or 포함? 제거하는 이유? 포함하는 이유?
    - 대부분의 문서에 등장 할지라도 빈도수에서 차이가 있을 수 있음
2.1.3 핵심 단어

2.2 의미 기준
    - 예) 맛에 대한 단어, 음식 종류에 대한 단어, 분위기에 대한 단어
    - 구분 이유?


2. 희소 단어 필터링

2.1 전체 문서(브랜드)에서의 희소단어 제거 - 비율 기준
2.1.1 목적: 일부 브랜드에만 등장하는 단어 제거
2.1.2 방법: 
    - 각 단어가 등장하는 문서(브랜드)의 비율이 기준값보다 낮으면 제거 
    - 예: 전체 문서(브랜드) 5000건 중 "machine"가 등장하는 문서 50건 --> 0.001(50/5000) 즉 0.1%의 문서에 만 등장 --> 기준값이 10%이면, 제거

2.2 각 문서(브랜드)별 희소단어 제거
2.2.1 목적: 각 브랜드별로, 해당 브랜드의 일부 리뷰글에만 등장하는 단어를 제거
2.2.2 방법: 
    - 각 브랜드의 리뷰들 중 해당 단어가 등장하는 리뷰의 비율이 기준값 보다 낮으면 제거
    - 예: 맥도날드 리뷰글 1000개 중 "fresh"가 등장하는 리뷰글 150개 --> 0.15(150/1000), 즉 15%의 맥도날드 리뷰글에 등장 --> 기준값과 비교하여 제거여부 결정
    - 각 브랜드별로 희소단어 제거 후 남은 단어 리스트들 합치기, 즉 모든 브랜드에서 희소한 단어들은 제거함

2.3 각 문서(브랜드)별 희소단어 제거 후 남은 단어들 중, 전체 문서(브랜드)에서의 희소단어 제거 - 횟수 기준
2.3.1 목적: 등장하는 문서(브랜드) 수가 너무 낮은 단어 제거 (2.2 수행 후 추가 필터링)
2.3.2 방법:
    - 각 단어가 등장하는 문서(브랜드)의 수가 기준값보다 낮으면 제거 
    - 예: "background"를 단어로 가지고 있는 문서(브랜드) 9건 --> 기준값이 10건이면, 제거

3. 추가 불용어 제거
3.2 내용: dtm 생성 및 회소 단어 필터링 후, 추가 불용어 제거
3.1 이유: 
    - cleaning, dtm생성 단계에서 불용어 제거했으나, 사전에(pre-) 모든 불용어를 정의하기는 어려움
    - 불용어가 발견될 때마다 처음부터 다시 시작하는 것은 시간이 많이 들고, 번거로움 
    - context나 분석 목적에 따라 제거해야할 불용어가 달라질 수 있음
3.3 방법:
    - 불용어 리스트 만들어 적용

4. 연습 & 토론
- 여러 가지 희소단어 제거 조건에 대해 dtm 생성 후, 결과 비교


'''

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#=================================
# 공통 설정
#=================================
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\session4_dtm\results"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\session4_dtm\results"

#=================================
# 브랜드별 리뷰 -> dtm 만들기
#=================================
def create_dtm_perBrand(input_file_name, min_df_rate, max_df_rate, min_review_prop_in_doc, min_doc_count_finally, manual_stopwords):

    '''
    Parameters
    ----------
    input_file_name: dtm 계산할 파일명
    min_df_rate, max_df_rate: 최소, 최대 df(document freq) rate, 각 단어의 등장빈도 비율 (등장 문서 수 / 전체 문서 수)
    min_review_prop_in_doc: 문서내에서 각 단어 최소 등장 비율  (해당 문서내에서 각 단어 등장 횟수/해당 문서의 리뷰 수)
    min_doc_count_finally, 
    manual_stopwords: 추가적으로 제거할 불용어 리스트
    
    
    biz_cat_slted: 데이터 추출할 business 카테고리
    state_slted: 데이터 추출할 지역(state)

    Returns
    -------
    브랜드별 cleaning된 리뷰 데이터
    '''

    #---------------------------------
    # 0. 데이터 불러오기
    #---------------------------------
    brand_reviews = pd.read_csv(f"{PATH_to_data}/{input_file_name}.csv")

    #---------------------------------
    # 1. 문서-단어 행열 (document-term freq matrix) 생성
    #---------------------------------
    ### 불용어 정의
    nltk_stop = stopwords.words('english')
    initial_manual_stopwords = [] # 제거할 stopping words 여기에 추가
    stop_words = nltk_stop + initial_manual_stopwords
    stop_words = sorted(list(set(stop_words)))

    ### CountVectorizer를 사용한 문서-단어 행렬 생성
    vectorizer = CountVectorizer(stop_words=stop_words) # CountVectorizer: 텍스트 데이터를 DTM으로 변환하는 클래스
    dtm = vectorizer.fit_transform(brand_reviews['pooled_text_clean']) # pooled_text_clean 컬럼으로 부터 DTM 생성
    terms = vectorizer.get_feature_names_out() # 단어이름 NumPy 배열 - 전체 단어 목록

    ### dtm 확인
    dtm # NumPy 희소행열 dtm.shape - (5151, 305676)
    len(terms) # 305676

    ### 단어수가 매우 큰 희소행렬이이어서 df로 변환하면 파일사이즈가 너무 커짐. 메모리 초과로 다운될 수 있음.
    # dtm_df = pd.DataFrame(dtm_array, columns=terms) # DataFrame으로 구성, 50*11455, 5241*3042
    # dtm_df['doc_id'] = brand_reviews['doc_id']

    #---------------------------------
    # 의미중복 통합
    #---------------------------------
    # 희소 단어 제거 하기 전에 의미 중복 통합


    #---------------------------------
    # 2. 희소단어 제거
    #---------------------------------

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
    dtm_df['doc_id'] = brand_reviews['doc_id'] # .values # 원래 문서 ID(doc_id)를 함께 붙이기

    ### 3) 제거된 단어 출력 (확인용)
    sparse_words_deleted = terms[df_rate < min_df_rate]
    common_words_deleted = terms[df_rate >= max_df_rate]
    print (f"제거된 희소단어(총 {len(sparse_words_deleted)}개): {sparse_words_deleted}") # 제거하는 것이 적절한지 검토할 것
    print (f"제거된 공통단어(총 {len(common_words_deleted)}개): {common_words_deleted}") # 제거하는 것이 적절한지 검토할 것

    #=================================
    # 2.2 각 문서(브랜드)별 희소단어 제거
    #=================================

    ### 1) 각 문서에서 각 단어가 등장하는 리뷰수 비율 계산
    dtm_df = dtm_df.set_index('doc_id').astype(float)  # 정수형 나눗셈 위해 float
    review_count = brand_reviews.set_index('doc_id')['review_count']
    dtm_normalized = dtm_df.div(review_count, axis=0) # 문서(브랜드)별로 각 단어가 등장하는 횟수를 해당 문서(브랜드)에 대한 전체 리뷰수로 나눔

    ### 2) 각 문서에서 리뷰수 비율이 특정값 이상인 단어들을 리스트로 추출
    high_freq_terms_per_doc = list()

    # row = dtm_normalized.iloc[0] # 확인용
    for _, row in dtm_normalized.iterrows(): # 각 행이 series 데이터로 input으로 들어감
        # print(row)
        high_terms = row[row >= min_review_prop_in_doc].index.tolist() # min_doc_prop 이상인 단어만 필터링 -> 단어명 추출
        high_freq_terms_per_doc.append(high_terms)

    ### 3) 자주 등장한 단어들의 전체 집합
    high_freq_terms = [item for words_batch in high_freq_terms_per_doc for item in words_batch] # 리스트의 리스트를 리스트로 변환
    high_freq_terms = sorted(list(set(high_freq_terms))) # 중복 제거
    # len(high_freq_terms) # 확인용

    #=================================
    # 2.3 각 문서(브랜드)별 희소단어 제거 후 남은 단어들 중, 전체 문서(브랜드)에서의 희소단어 제거 - 횟수 기준
    # (2.2 수행 후 추가 필터링)
    #=================================

    ### 1) 각 단어가 얼마나 많은 문서(브랜드)에서 등장했는지 계산
    term_doc_counts = dict() # 각 단어가 등장한 문서(브랜드) 수 저장할 dict
    for term in high_freq_terms:
        # 각 문서 리스트(high_freq_terms_per_doc)에서 해당 단어가 등장하는 문서 수 계산
        count = sum(term in term_list for term_list in high_freq_terms_per_doc)
        term_doc_counts[term] = count

    term_doc_df = pd.DataFrame(list(term_doc_counts.items()), columns=['term', 'doc_count']) # 단어 빈도 사전 --> dataframe으로 변환

    ### 2) 기준값 이상인 경우만 선택
    term_doc_df_slted = term_doc_df[term_doc_df['doc_count'] > min_doc_count_finally] # min_doc_count_finally(예, 10개) 초과 문서에 등장하는 경우만 남김
    words_features = term_doc_df_slted['term'].to_list() # 단어 리스트

    ### 3) dtm에 적용 - 선택된 단어 컬럼만 남김
    dtm_df = dtm_df[words_features] 


    #---------------------------------
    # 3. 불용어 제거 (수동)
    #---------------------------------
    dtm_df_cleaned = dtm_df.copy()
    # dtm_df_cleaned.columns.to_list() # 키워드 리스트

    to_drop = [w for w in manual_stopwords if w in dtm_df_cleaned.columns] # manual_stopwords중 dtm 컬럼에 존재하는 단어만 추출
    dtm_df_cleaned = dtm_df_cleaned.drop(to_drop, axis=1) # .drop(columns=to_drop) 과 동일


    #---------------------------------
    # 최종 분석용 데이터 결합 및 저장 
    #---------------------------------
    ### 정리
    dtm_df_cleaned = dtm_df_cleaned.loc[~(dtm_df_cleaned == 0).all(axis=1)] # 모든 값이 0인 행 제거, .all(axis=1) 각 행에 대해 각 열의 값이 전부 T이면 T, 하나라도 F이면 F 반환.  
    dtm_df_cleaned = dtm_df_cleaned.loc[:, ~(dtm_df_cleaned == 0).all(axis=0)] # 모든 값이 0인 열 제거, .loc[:, mask]에서 : 는 모든 행, mask는 살아남은 열

    # brand별 데이터에 dtm 붙이기
    brand_reviews_dtm = pd.merge(brand_reviews.drop(columns=['pooled_text_clean']), dtm_df_cleaned.reset_index(), on='doc_id', how='inner')

    # 저장
    brand_reviews_dtm = brand_reviews_dtm.set_index("doc_id") # doc_id는 데이터 프로세싱과정에서 임시로 사용한 것이므로 index로 설정하여 저장할때 삭제되도록 함
    brand_reviews_dtm.to_csv(f"{PATH_to_save}/{input_file_name}_{min_df_rate}_{max_df_rate}_{min_review_prop_in_doc}_{min_doc_count_finally}_dtm.csv", index=False, encoding='utf-8-sig')

    return brand_reviews_dtm


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

    ## 조건 1
    input_file_name = 'reviews_restaurants_az_perBrand'
    min_df_rate = 0.1 # 최소 df(document freq) rate, 각 단어의 등장빈도 비율 (등장 문서 수 / 전체 문서 수)
    max_df_rate = 0.9 # 최대 df(document freq) rate, 각 단어의 등장빈도 비율 (등장 문서 수 / 전체 문서 수)
    min_review_prop_in_doc = 0.3 # 단어가 주어진 문서(브랜드)의 리뷰들 중에서 등장하는 최소 비율 (0.5 - kim et al. 2024 JMR)
    min_doc_count_finally = 10 # 각 단어가 최종 단어 리스트에 포함되기 위해, 해당 단어가 등장해야하는 최소 문서수(브랜드 수)
    brand_reviews_dtm = create_dtm_perBrand(input_file_name, min_df_rate, max_df_rate, min_review_prop_in_doc, min_doc_count_finally, manual_stopwords)

    ## 조건 2
    input_file_name = 'reviews_restaurants_az_perBrand'
    min_df_rate = 0.1
    max_df_rate = 0.9
    min_review_prop_in_doc = 0.1 # 단어가 주어진 문서(브랜드)의 리뷰들 중에서 등장하는 최소 비율 (0.5 - kim et al. 2024 JMR)
    min_doc_count_finally = 10 # 각 단어가 최종 단어 리스트에 포함되기 위해, 해당 단어가 등장해야하는 최소 문서수(브랜드 수)
    brand_reviews_dtm = create_dtm_perBrand(input_file_name, min_df_rate, max_df_rate, min_review_prop_in_doc, min_doc_count_finally, manual_stopwords)


    # stw, term 카테고리 분류
    brand_reviews_dtm.columns.to_list()


    # ## 조건 1
    # input_file_name = 'reviews_restaurants_az_perBrand'
    # min_df_rate = 0.1
    # max_df_rate = 0.9
    # min_review_prop_in_doc = 0.3 # 단어가 주어진 문서(브랜드)의 리뷰들 중에서 등장하는 최소 비율 (0.5 - kim et al. 2024 JMR)
    # min_doc_count_finally = 10 # 각 단어가 최종 단어 리스트에 포함되기 위해, 해당 단어가 등장해야하는 최소 문서수(가게수)
    # brand_reviews_dtm = create_dtm_perBrand(input_file_name, min_df_rate, max_df_rate, min_review_prop_in_doc, min_doc_count_finally, manual_stopwords)

    # ## 조건 2
    # input_file_name = 'reviews_restaurants_az_perBrand'
    # min_df_rate = 0.1
    # max_df_rate = 0.9
    # min_sparsity_among_all_doc = 0.1 # 단어가 전체 문서(브랜드) 중 등장하는 문서(브랜드) 최소 비율
    # min_review_prop_in_doc = 0.1 # 0.5 - kim et al. 2024 JMR 
    # min_doc_count_finally = 10 # 각 단어가 최종 단어 리스트에 포함되기 위해, 해당 단어가 등장해야하는 최소 문서수(가게수)
    # brand_reviews_dtm = create_dtm_perBrand(input_file_name, min_df_rate, max_df_rate, min_review_prop_in_doc, min_doc_count_finally, manual_stopwords)


# 결과 분석

# 제거된 희소단어 (302,627개)

# 전체 305,676개 단어 중 99%가 희소단어로 제거됨. 
#  `'__'`, `'___'` 같은 특수문자 잔재랑 `'麻辣鍋'`, `'點心'` 같은 중국어/한자 단어들이 포함되어 있음.
# 전처리 단계에서 완전히 걸러지지 않은 노이즈들이 여기서 제거된 것으로 보임.

# ---

# 제거된 공통단어 (112개)

# 90% 이상의 브랜드에서 등장하는 단어들인데, 보면 `'food'`, `'good'`, `'great'`, `'delici'`, `'order'`, `'servic'` 같이 레스토랑 리뷰에서 거의 항상 나오는 단어들임.
# 모든 브랜드에 공통으로 나오니까 브랜드 간 변별력이 없어서 제거된 것으로 보임.

# ---

# 조건1, 2 결과가 동일한 이유

# 2.1 단계(df_rate 필터링)는 두 조건이 완전히 같은 파라미터(`min=0.1, max=0.9`)라서 출력이 동일한 것으로 보임.
# 조건1, 2의 차이는 2.2 단계(`min_review_prop_in_doc`: 0.3 vs 0.1)에서 나타나는 거라 여기선 동일하게 보이는 게 정상.



