'''
1. DTM 문제점
1.1 빈도 vs 중요도
    - 빈도만으로는 단어 중요도를 비교하기 어려움. 
    - 예: good, food 같은 보편적인 단어는 거의 모든 식당 리뷰에 등장하여 빈도가 높지만, 각 식당을 특성을 설명하지는 못함.
    - 즉, 고빈도 공통어가 단순 빈도로는 중요도가 과대평가됨
1.2 길이 편향
    - 텍스트를 dtm으로 바꾸면 길이가 긴 문서일수록 각 단어를 더 많이 포함하게 됨. 즉 단어 벡터의 크기가 커짐
    - 벡터 크기가 커면 거리, 분산, 가중치 계산에서 더 많이 반영됨
    - 즉, 텍스트가 길다는 이유만으로 더 중요하다는 편향이 발생할 수 있음

2. TF-IDF
2.1 개념
    - TF (Term Frequency): “문서 안에서 얼마나 자주 나오나?”
    - IDF (Inverse Document Frequency): “전체 코퍼스에서 얼마나 희귀한가?”
    - TF*IDF: 둘을 곱해 자주 등장하면서도 희귀한 단어에 높은 가중치를 줌
2.2 역할
    - 보편적 단어에 대한 가중치를 낮추고, 특정 문서에만 많이 등장하는 단어의 가중치를 높임으로써 문서(브랜드)간의 차별적 특징을 잘 반영함
    - 즉, 희소단어 부각 + 보편단어 억제 --> 문서(브랜드)간 차별성 증폭
2.3 방법    
    - idf = log((N+1)/(df+1))+1
    - N 문서(브랜드)수, df 해당 단어가 포함된 문서(브랜드)수

3. 문서별 정규화
3.1 개념: 
    - 벡터 또는 변수 스케일 맞추기, 행(문서) 벡터의 길이를 1로 통일
3.2 역할: 
    - 길이 편향 제거 --> 내용 (즉, 행 벡터의 방향)만 비교
3.2 방법
    - L2 정규화: 행 벡터의 유클리드 길이가 1인 벡터로 만듦, v/||v||2
    - L1 정규화: 행 벡터 각 원소의 절대값의 합이 1인 벡터로 만듦, v/||v||1

4. 사용
- 대부분의 벡터 공간 기법 (PCA, 군집분석, 회귀 등)에서 tfidf + L2정규화 조합 적용이 일반적임
- 토픽모델 LDA는 다항분포(정수 count)를 가정하므로 tfidf, 정규화 모두 적용하면 안됨
- 토픽모델 NMF에서 모두 가능: tfidf + L2 정규화 (해석, 토픽 선명도 개선됨)

'''

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 

#=================================
# 공통 설정
#=================================
PATH_to_data = ""
PATH_to_save = ""


#=================================
# tfidf 변환 함수
#=================================
def create_and_save_tfidf_perBrand(dtm_file_name, apply_l2):

    #---------------------------------
    # 0. 데이터 불러오기 
    #---------------------------------
    df = pd.read_csv(f"{PATH_to_data}/{dtm_file_name}.csv") # dtm 데이터 불러오기
    # df = df.set_index('name')


    #---------------------------------
    # 1. 전처리 
    #---------------------------------
    ### 메타데이터와 dtm 분리
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    dtm_cols = [col for col in df.columns if col not in meta_cols]

    df_meta = df[meta_cols] # 메타데이터 분리
    df_dtm = df[dtm_cols] # 단어 빈도 DTM만 추출


    #---------------------------------
    # 2. tf-idf 변환 
    #---------------------------------
    ### 변환 객체 생성 & 변환
    if apply_l2:
        tfidf = TfidfTransformer(norm='l2') # TF-IDF 변환기 생성, l2 norm 적용
    else:
        tfidf = TfidfTransformer(norm=None) # TF-IDF 변환기 생성, norm 미적용
    dtm_tfidf = tfidf.fit_transform(df_dtm) # TF-IDF 행렬 계산 (sparse matrix 반환)

    ### dataframe으로 변환
    df_tfidf = pd.DataFrame(dtm_tfidf.toarray(), columns=dtm_cols) # TF-IDF 행렬을 pandas df로 변환
    # np.linalg.norm(df_tfidf, axis=1) # normalize 확인용
    df_tfidf = df_tfidf.round(5) # 자릿수 조정

    df_tfidf = pd.concat([df_meta, df_tfidf], axis=1) # 메타데이터와 TF-IDF 데이터 결합, axis=1 옆으로 나란히 결합

    
    #---------------------------------
    # 3. 저장하기 
    #---------------------------------
    if apply_l2:
        df_tfidf.to_csv(f"{PATH_to_save}/{dtm_file_name}_tfidf_l2.csv", encoding='utf-8-sig', index=False)
    else:
        df_tfidf.to_csv(f"{PATH_to_save}/{dtm_file_name}_tfidf.csv", encoding='utf-8-sig', index=False)

    return df_tfidf


if __name__ == "__main__":
    dtm_file_name = 'reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm'
    df_tfidf = create_and_save_tfidf_perBrand(dtm_file_name=dtm_file_name, apply_l2=False)
    df_tfidf_l2 = create_and_save_tfidf_perBrand(dtm_file_name=dtm_file_name, apply_l2=True)

    dtm_file_name = 'reviews_restaurants_az_perBrand_0.1_0.9_0.1_10_dtm'
    df_tfidf = create_and_save_tfidf_perBrand(dtm_file_name=dtm_file_name, apply_l2=False)
    df_tfidf_l2 = create_and_save_tfidf_perBrand(dtm_file_name=dtm_file_name, apply_l2=True)




