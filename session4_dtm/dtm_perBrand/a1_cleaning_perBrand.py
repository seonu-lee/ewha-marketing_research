'''
1. brand 이름 cleaning
1.1 브랜드 이름 역할: 브랜드 이름을 식별자로 사용함
1.2 문제: 대소문자, 띄어쓰기, 특수문자 등 섞여 있는 경우 서로 다른 브랜드로 처리될 수 있음
1.3 전처리: 소문자화, 앞뒤/내부 공백제거, 특수문자제거(알파벳, 숫자만 남김)

2. 데이터 선별 
    - 산업 카테고리, 지역, 빈도 수 등 분석 목적에 맞게 추출

3. 리뷰 텍스트 cleaning
3.1 대소문자, 특수문자, 공백, 구두점 변환/제거 
3.2 어간(stem)
    - 파생어, 복합어 만들때 변하지 않는 부분
    - "맛있는 음식을 먹었다" --> 어간: ["맛있", "음식", "먹"]
    - "The children are playing happily in the playground." 
        --> ['the', 'children', 'are', 'play', 'happili', 'in', 'the','playground']
3.3 형태소
    - 의미를 가지는 최소 단위
    - 한국어처럼 1어절에 의미, 문법요소가 연쇄적으로 붙는 경우, 의미단위로 구분
    - "맛있는 음식을 먹었다" --> 형태소: ["맛있", "는", "음식", "을", "었", "다"]
3.3 오타

'''

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
tqdm.pandas() # pandas에 progress_apply 기능 등록

#=================================
# 공통 설정
#=================================
PATH_to_data = "C:\\Users\\seonu\\Documents\\ewha-marketing_research\\datasets\\yelp_dataset"
PATH_to_save = "C:\\Users\\seonu\\Documents\\ewha-marketing_research\\session4_dtm\\results"

#=================================
# 리뷰 텍스트 클리닝
#=================================
def cleaning_review_text_perBrand(biz_cat_slted, state_slted):
    '''
    Parameters
    ----------
    biz_cat_slted: 데이터 추출할 business 카테고리
    state_slted: 데이터 추출할 지역(state)

    Returns
    -------
    브랜드별 cleaning된 리뷰 데이터
    '''

    #---------------------------------
    # 0. 데이터 불러오기
    #---------------------------------
    business_raw = pd.read_csv(f"{PATH_to_data}/yelp_business.csv")
    reviews_raw = pd.read_csv(f"{PATH_to_data}/yelp_review.csv")
    # users_raw = pd.read_csv(f"{PATH_to_data}/yelp_user.csv")
    # hours_raw = pd.read_csv(f"{PATH_to_data}/yelp_business_hours.csv")

    business = business_raw.copy()
    reviews = reviews_raw.copy()
    # users = users_raw.copy()
    # hours = hours_raw.copy()

    #---------------------------------
    # 1. 브랜드 이름 cleaning
    #---------------------------------
    # 이름 cleaning
    business['name_ori'] = business['name']
    business['name'] = business['name'].str.lower() # 소문자화
    # business['name'] = business['name'].str.strip() # 앞뒤 공백 제거
    # business['name'] = business['name'].str.replace(r'\s+', '', regex=True) # 공백 제거 (정규표현식 이용 r'')
    business['name'] = business['name'].str.replace('[^a-z0-9]', '', regex=True) # 특수문자 제거 (알파벳/숫자만 남김)

    #---------------------------------
    # 2. 분석대상 business 선정
    #---------------------------------
    business_slted = business.copy() # 원본과 분리된 작업임을 명시적으로 표시함. copy() 사용하면 복사하여 별도의 id가 부여됨.

    # state 필터링
    business_slted = business_slted[business_slted['state']==state_slted] # 애리조나(AZ) 주의 음식점 데이터만 추출

    # category 필터링
    business_slted = business_slted[business_slted['categories'].str.contains(biz_cat_slted)] # 카테고리가 Restaurant인 가게만 추출

    # 리뷰수 필터링
    business_slted = business_slted[business_slted['review_count'] > 10] # 리뷰 수 기준 필터링

    #---------------------------------
    # 3. 리뷰 데이터 브랜드별 aggregation
    #---------------------------------
    ### 리뷰 데이터(reviews)에 브랜드 이름 정보(name) 추가하기
    reviews_slted = pd.merge(reviews, business_slted[['business_id', 'name']], how='inner', on='business_id')

    ### brand name 중복 확인
    # len(reviews_slted['name'].unique())
    # len(reviews_slted['business_id'].unique())

    ### 그룹 요약 (브랜드별 리뷰 요약) 
    brand_reviews = reviews_slted.groupby('name').agg({'review_id': 'count', 'stars': 'mean', 'text': ' '.join, 'useful': 'sum', 'funny': 'sum', 'cool': 'sum'}).reset_index() # name 기준 agg
    brand_reviews = brand_reviews.rename(columns={'review_id': 'review_count', 'stars': 'avg_stars', 'text': 'pooled_text', 'useful': 'useful_count', 'funny': 'funny_count', 'cool': 'cool_count'}) # 컬럼이름 수정

    ### 각 브랜드의 category 정보 추가
    brand_reviews = pd.merge(brand_reviews, business_slted[['name', 'categories']].drop_duplicates(subset=['name'], keep='first'), on='name', how='left')
    brand_reviews = brand_reviews.sort_values(by='review_count', ascending=False).reset_index(drop=True).reset_index().rename(columns={'index': 'doc_id'}) # 데이터 처리과정에서 이름대신 사용하기 위해 임시 doc_id 생성


    #---------------------------------
    # 4. 텍스트 전처리: 소문자, 숫자/구두점 제거, 불용어 및 어간 추출
    #---------------------------------
    ### 테스트용 미니 sample 사용할 경우
    # brand_reviews = brand_reviews.head(50) # 리뷰수 기준 50개만 추출 (테스트용)

    ### 단어 토큰에서 제거할 불용어 설정
    nltk.download('stopwords') # NLTK stopwords 다운로드 (최초 실행 시 필요)
    stop_words = set(stopwords.words('english')) # 기본 불용어 리스트 
    custom_remove = {"does", "not", "thing"} # 사용자 정의 불용어

    ### 소문자 + 숫자 제거 + 구두점 제거
    brand_reviews['pooled_text_clean'] = brand_reviews['pooled_text'].str.lower()
    # brand_reviews['pooled_text_clean'] = brand_reviews['pooled_text_clean'].str.replace('[^a-zA-Z_ ]+', '', regex=True) # 이렇게 하면 영어외의 언어 글자 삭제됨
    brand_reviews['pooled_text_clean'] = brand_reviews['pooled_text_clean'].str.replace(r'\d+', '', regex=True) # 숫자 제거
    brand_reviews['pooled_text_clean'] = brand_reviews['pooled_text_clean'].str.replace(r'[^\w\s]', '', regex=True) # \w는 문자, 숫자, 밑줄, 공백 이외의 모든 문자(즉, 구두점 등)를 제거
    # brand_reviews[['pooled_text', 'pooled_text_clean']].sample(50) # 확인용

    ### 토큰화 + 어간추출 -> 다시 문장으로 합치기 (시간 많이 걸림)
    stemmer = PorterStemmer() # 어간 추출기 초기화
    def tokenize_filter_stem(text):
        tokens = text.split() # 공백 기준으로 쪼개기
        tokens = [w for w in tokens if w not in stop_words and w not in custom_remove] # 불용어 제거
        tokens = [stemmer.stem(w) for w in tokens] # 어간 추출하여 리스트로 나열
        return ' '.join(tokens) # 추출된 어간 리스트를 다시 합치기
    brand_reviews['pooled_text_clean'] = brand_reviews['pooled_text_clean'].progress_apply(tokenize_filter_stem) # Pandas .apply()에 tqdm 기반 진행률 바를 붙인 함수


    #---------------------------------
    # 저장
    #---------------------------------
    ### 정리
    brand_reviews = brand_reviews.drop(["pooled_text"], axis=1) # 원본 pooled text 삭제

    ### 저장
    brand_reviews.to_csv(f"{PATH_to_save}/reviews_{biz_cat_slted.lower()}_{state_slted.lower()}_perBrand.csv", index=False, encoding='utf-8-sig')

    return brand_reviews

if __name__ == '__main__':
    biz_cat_slted = 'Restaurants'
    state_slted = 'AZ'
    brand_reviews = cleaning_review_text_perBrand(biz_cat_slted, state_slted)


#__name__ 이란?

# 파일을 직접 실행하면 → __name__ == '__main__' → 조건 True, 실행됨
# 다른 파일에서 import하면 → __name__ == '파일명' → 조건 False, 실행 안 됨

#BOW(Bag of Words) — 개념. 단어 순서 무시하고 등장 횟수만 셈
# DTM — BOW를 여러 문서에 대해 행렬로 만든 것
# TF-IDF — DTM의 단순 빈도 대신 가중치로 교체한 버전