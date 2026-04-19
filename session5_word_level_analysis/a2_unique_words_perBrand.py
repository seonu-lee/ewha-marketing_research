'''
1. 브랜드별 고유 단어 찾기
1.1 주어진 브랜드의 각 단어의 고유도 계산
1.1 log_odds: 각 단어가 선택 브랜드에서 전체 브랜드와 비교하여 쓰이는 비율 (고유도)
    - Monroe, Burt L., Michael P. Colaresi, and Kevin M. Quinn. 2008. “Fightin’ Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict.” Political Analysis 16 (4): 372-403. https://doi.org/10.1093/pan/mpn018
1.2 계산방법: 
    - (브랜드 A 에서의 각 단어 등장 비율 - 나머지에서의 각 단어의 등장 비율)
    - 브랜드 A에서 등장 비율이 나머지 전체에서의 등장 비율보다 높은 단어는 브랜드 A의 고유단어로 더 적합    
    - z-score: log-odds를 표준오차로 나누어 정규화, 통계적 유의성 검증
    - 주의: 정수 count 벡터를 전제로 만든 통계임. tfidf에 적용하면 통계적 의미가 모호해짐

2. 브랜드간 공통점, 차이점 찾기
2.1 브랜드간 고유단어 비교 
    - A, B 브랜드 공틍으로 많이 쓰는 단어
    - A, B 브랜드 공통으로 적게 쓰는 단어
    - A 브랜드에 많이 쓰고, B 브랜드에 적게 쓰는 단어
    - A 브랜드에 적게 쓰고, B 브랜드에 많이 쓰는 단어

3. 주어진 단어에 대한 브랜드간 사용의 차이
3.1 예시
3.1.1 food
    - food 라는 단어를 많이 vs 적게 쓰는 브랜드들의 특징? 
    - 메뉴 폭/전문성?
3.1.2 delicious
3.1.3 fast

4. 경쟁 브랜드 추출
4.1 경쟁 정의: 브랜드별 단어 사용과 경쟁 관계?
4.2 특정 브랜드와 가장 언어적으로 유사한 브랜드
4.2.1 코사인 유사도
    - cosine similarity(a,b)=cosθ
    

'''



import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint

#=================================
# 조건 설정
#=================================
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\session4_dtm\results"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\session5_word_level_analysis\results"

#=================================
# 0. 데이터 불러오기 
#=================================
# df_dtm = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm.csv") # dtm 데이터 불러오기
# df_dtm_tfidf = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_0.9_0.1_10_dtm_tfidf.csv") # tfidf적용한 dtm 데이터 불러오기
# df_dtm_tfidf_l2 = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_0.9_0.1_10_dtm_tfidf_l2.csv") # tfidf적용한 dtm 데이터 불러오기

# 상위 10% 공통단어 제거하지 않은 데이터
df_dtm = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_1.0_0.3_10_dtm.csv") # dtm 데이터 불러오기
df_dtm_tfidf = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_1.0_0.3_10_dtm_tfidf.csv") # tfidf적용한 dtm 데이터 불러오기
df_dtm_tfidf_l2 = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_1.0_0.3_10_dtm_tfidf_l2.csv") # tfidf적용한 dtm 데이터 불러오기

### 사용할 데이터 선택
df_tf = df_dtm.copy()
df_tfidf = df_dtm_tfidf.copy()
df_tfidf_l2 = df_dtm_tfidf_l2.copy()

### 컬럼 구분 - meta, word 컬럼
meta_cols = ['name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories']
word_cols = [col for col in df_tf.columns if col not in meta_cols]

data_tf = df_tf[word_cols] # 단어 컬럼만 추출
data_tfidf = df_tfidf[word_cols]
data_tfidf_l2 = df_tfidf_l2[word_cols]


#=================================
# 1. 브랜드별 고유 단어 찾기
#=================================
### 고유도 계산 함수
def log_odds_dirichlet(count_row, count_corpus, alpha=0.01):
    """
    count_row : 1-D array (특정 브랜드의 단어 카운트)
    count_corpus : 1-D array (코퍼스 전체 카운트 = 모든 브랜드 합계)
    alpha : 0 빈도를 방지하고 극희소 단어에 대한 과도한 log-odds 발산을 완화
    returns : DataFrame - 컬럼 [log_odds, z_score]
    """
    # 1. 스무딩된 카운트
    cw_A = count_row + alpha # 브랜드 A에서 각 단어의 등장 횟수 리스트 (0이 되는 것을 방지하기 위해 alpha 더해줌)
    cw_B = (count_corpus - count_row) + alpha # 브랜드 A를 제외한 나머지 모든 브랜드들에서 각 단어의 등장 횟수 리스트

    # 2. 로그 오즈 차
    # 브랜드 A 에서의 각 단어 등장 비율 - 나머지에서의 각 단어의 등장 비율
    delta = np.log(cw_A) - np.log(np.sum(cw_A)) - (np.log(cw_B) - np.log(np.sum(cw_B)))

    # 3. 분산 & z-score
    var = 1/(cw_A) + 1/(cw_B)
    z = delta / np.sqrt(var)

    return pd.DataFrame({'log_odds': delta, 'z_score': z})

### 특정 브랜드에 대한 고유단어 추출 함수
def extracting_unique_words_for_brand(data_tf, brand_name):

    brand_idx = df_tf[df_tf['name']==brand_name].index[0] # 브랜드 idx 찾기

    counts_A  = data_tf.iloc[brand_idx].values # 브랜드 A 단어 TF
    counts_C  = data_tf.sum(axis=0).values # 전체 코퍼스 TF

    df_logodds = log_odds_dirichlet(count_row=counts_A, count_corpus=counts_C, alpha=0.1)
    df_logodds.index = data_tf.columns # 단어 라벨
    df_logodds = df_logodds.sort_values('z_score', ascending=False)
    return df_logodds

### 적용 --------------------
brand_name='subway'
brand_name='mcdonalds'
df_logodds = extracting_unique_words_for_brand(data_tf=data_tf, brand_name=brand_name)
df_logodds.head(10)
df_logodds.tail(10)

#=================================
# 2. 브랜드간 공통점, 차이점 찾기
#=================================
### 브랜드간 고유단어 비교함수
def compare_unique_words_for_two_brands(brands_to_compare):

    ### 브랜드 할당
    brandA = brands_to_compare[0]
    brandB = brands_to_compare[1]

    ### logodds 계산
    # brandA - logodds 계산
    df_logodds_brandA = extracting_unique_words_for_brand(data_tf=data_tf, brand_name=brandA)
    df_logodds_brandA = df_logodds_brandA.reset_index().rename(columns={'index': 'word'})
    df_logodds_brandA['brand'] = brandA

    # brandB - logodds 계산
    df_logodds_brandB = extracting_unique_words_for_brand(data_tf=data_tf, brand_name=brandB)
    df_logodds_brandB = df_logodds_brandB.reset_index().rename(columns={'index': 'word'})
    df_logodds_brandB['brand'] = brandB

    df_logodds_pooled = pd.concat([df_logodds_brandA, df_logodds_brandB], axis=0) # 합치기

    ### 데이터 배열 변경: brand, word, z_score --> word * brand
    pivot_z = df_logodds_pooled.pivot(index='word', columns='brand', values='z_score')

    ### High/Low 기준 테이블 만들기
    # 1) ‘High / Low’ 판정 기준 설정
    thr  = 1.96 # p-value 약 0.5
    high =  pivot_z  >  thr      # True/False 매트릭스
    low  =  pivot_z  < -thr

    # 2) 2×2 표(카운트) 만들기
    HH =  (high[brandA] & high[brandB]).sum()      # 공통 High
    HL =  (high[brandA] & low [brandB]).sum()      # 맥 High, 서브 Low
    LH =  (low [brandA] & high[brandB]).sum()      # 맥 Low, 서브 High
    LL =  (low [brandA] & low [brandB]).sum()      # 공통 Low

    hl_freq_table = pd.DataFrame(
        [[HH, HL],
        [LH, LL]],
        index=[f'{brandA} High', f'{brandA} Low'],
        columns=[f'{brandB} High', f'{brandB} Low']
    )
    # print(hl_freq_table) # 확인용

    # 3) 전용 키워드 리스트 추출
    hl_words_dic = dict()
    hl_words_dic[f'{brandA}_only'] = pivot_z.index[high[brandA] & low[brandB]].tolist()
    hl_words_dic[f'{brandB}_only'] = pivot_z.index[low[brandA]  & high[brandB]].tolist()
    hl_words_dic['common_high'] = pivot_z.index[high[brandA] & high[brandB]].tolist()
    hl_words_dic['common_low'] = pivot_z.index[low[brandA] & low[brandB]].tolist()
    # print(hl_words_dic) # 확인용

    return hl_freq_table, hl_words_dic

### 적용 --------------------
# ex. 맥도날드 vs 서브웨이
brands_to_compare = ['subway', 'mcdonalds']
hl_freq_table, hl_words_dic = compare_unique_words_for_two_brands(brands_to_compare=brands_to_compare)
print(hl_freq_table) # 공틍으로 많이 쓰는 단어, 공통으로 적게 쓰는 단어, 맥도날드에만 많이 쓰는 단어, 서브웨이에만 많이 쓰는 단어--2*2 테이블
pprint(hl_words_dic, width=80, compact=True)


#=================================
# 3. 주어진 단어에 대한 브랜드간 사용의 차이
#=================================
### 각 브랜드별로 고유단어 추출하여, 합치기 함수
def cal_logodds_for_all_brands(data_tf):
    
    # 1) 각 브랜드에 대해, 각 단어의 고유도 계산 (log_odds, z_score)
    df_logodds_list = list()

    # brand_idx_each = data_tf.index[0] # 확인용
    for brand_idx_each in data_tf.index:
        # print(brand_idx_each)   

        counts_A  = data_tf.iloc[brand_idx_each].values # 브랜드 A 단어 TF
        counts_C  = data_tf.sum(axis=0).values # 전체 코퍼스 TF

        df_logodds = log_odds_dirichlet(count_row=counts_A, count_corpus=counts_C, alpha=0.1) #  각 단어의 고유도 계산 (log_odds, z_score)
        df_logodds.index = data_tf.columns # 단어 라벨 추가

        df_logodds = df_logodds.sort_values('z_score', ascending=False)
        df_logodds = df_logodds.reset_index().rename(columns={'index': 'word'}) # word를 index에서 컬럼으로 가져옴
        df_logodds['brand']= df_tf['name'].values[brand_idx_each] # 해당 브랜드 이름 추가
        df_logodds =df_logodds.set_index("brand").reset_index() # 브랜드 컬럼을 맨 앞으로 가져오기

        df_logodds_list.append(df_logodds)

    # 2) 합치기
    df_logodds_pooled = pd.concat(df_logodds_list, axis=0) # 합치기
    return df_logodds_pooled

### 적용 ---------------
df_logodds_pooled = cal_logodds_for_all_brands(data_tf)
data_tf.sum().sort_values(ascending=False).index.to_list() # 컬럼에 있는 단어 리스트 확인 (내림차순)

### food 라는 단어를 많이 vs 적게 쓰는 브랜드들의 특징
# 메뉴 폭/전문성?
target_word = 'food'
df_logodds_pooled[df_logodds_pooled['word']==target_word].sort_values('z_score', ascending=False).head(50)
df_logodds_pooled[df_logodds_pooled['word']==target_word].sort_values('z_score', ascending=True).head(50)

### delicious
target_word = 'delici'
df_logodds_pooled[df_logodds_pooled['word']==target_word].sort_values('z_score', ascending=False).head(50)
df_logodds_pooled[df_logodds_pooled['word']==target_word].sort_values('z_score', ascending=True).head(50)

### fast
target_word = 'fast'
df_logodds_pooled[df_logodds_pooled['word']==target_word].sort_values('z_score', ascending=False).head(50)
df_logodds_pooled[df_logodds_pooled['word']==target_word].sort_values('z_score', ascending=True).head(50)

#=================================
# 4. 경쟁 브랜드 추출
#=================================
### 코사인 유사도 상위 브랜드 추출 함수
def nearest_brands(target_brand, X, topn=10):
    sim = cosine_similarity(X.loc[[target_brand]], X)[0] # X.loc[[]] -- 데이터프레임형태유지하기 위해 이중 대괄호사용. 이차원 df 형태 유지해야 cosine_similarity 인풋으로 사용가능함
    s = pd.Series(sim, index=X.index).sort_values(ascending=False)
    return s.drop(target_brand).head(topn)

### 적용
X_tfidf = df_tfidf.set_index('name')[word_cols]
nearest_brands(target_brand='mcdonalds', X=X_tfidf, topn=10)
nearest_brands('subway', X_tfidf, topn=10)
nearest_brands('chipotlemexicangrill', X_tfidf, topn=10)

X_tfidf_l2 = df_tfidf_l2.set_index('name')[word_cols]
nearest_brands(target_brand='mcdonalds', X=X_tfidf_l2, topn=10)
nearest_brands('subway', X_tfidf_l2, topn=10)
nearest_brands('chipotlemexicangrill', X_tfidf_l2, topn=10)

######################################

# **[인사이트] 브랜드 수준 분석 결과**

# ---

# **1. 맥도날드 vs 서브웨이 비교 (2×2 테이블)**

# 공통 High 50개, 공통 Low 84개로 두 브랜드가 **패스트푸드 체인**으로서 공유하는 언어 패턴이 뚜렷함. `fast`, `drive`, `locat`, `line`, `manag`, `employe` 등 **운영/효율성** 관련 단어를 공통으로 많이 쓰고, `delici`, `flavor`, `steak`, `server`, `atmospher` 등 **다이닝 경험** 관련 단어를 공통으로 적게 씀.

# 맥도날드 전용 단어(`burger`, `fri`, `egg`, `coffe`, `cream`, `bun`)는 **햄버거·아침식사** 메뉴 중심을 반영하고, 서브웨이 전용 단어(`fresh`, `healthi`, `ingredi`, `bread`, `veggi`, `turkey`, `tuna`)는 **건강·커스터마이징** 브랜드 정체성을 잘 반영함.

# ---

# **2. 단어별 브랜드 사용 차이**

# `food`를 많이 쓰는 브랜드는 `pandaexpress`, `truefoodkitchen`, `chickfila` 등 **다양한 메뉴를 제공하는 체인** 위주이고, 적게 쓰는 브랜드는 `pizzeriabianco`, `grimaldispizzeria`, `bosadonuts` 등 **특정 메뉴에 특화된 전문점** 위주임. 즉 `food`는 메뉴 폭이 넓을수록 더 자주 등장하는 단어임.

# `delici`를 많이 쓰는 브랜드는 `flowerchild`, `coladoscoffeecrepes`, `greennewamericanvegetarian` 등 **건강식·카페·브런치** 계열이고, 적게 쓰는 브랜드는 `mcdonalds`, `buffalowildwings`, `chipotlemexicangrill` 등 **패스트푸드·스포츠바** 계열임. 음식 품질보다 속도·편의를 강조하는 브랜드일수록 `delici` 사용이 낮음.

# `fast`를 많이 쓰는 브랜드는 `innoutburger`, `jimmyjohns`, `chickfila`, `wendys` 등 전형적인 **패스트푸드 체인**이고, 적게 쓰는 브랜드는 `culinarydropout`, `truefoodkitchen`, `pizzeriabianco` 등 **캐주얼 다이닝·피자·브런치** 계열임. 이는 `fast`가 브랜드 포지셔닝을 구분하는 핵심 단어임을 시사함.

# ---

# **3. 경쟁 브랜드 추출 (코사인 유사도)**

# 맥도날드의 경쟁 브랜드는 `jackinthebox`(0.939), `wendys`(0.938), `sonicdrivein`(0.904) 등 **드라이브스루 중심 패스트푸드 체인**으로 실제 시장 경쟁 구도와 일치함.

# 서브웨이의 경쟁 브랜드는 `quiznos`(0.929), `jerseymikessubs`(0.925), `firehousesubs`(0.908) 등 **샌드위치 전문 체인**으로 구성됨.

# 치폴레의 경쟁 브랜드는 `qdobamexicaneats`(0.921), `uberrito`(0.882) 등 **멕시칸 패스트캐주얼** 브랜드로 집중됨.

# TF-IDF와 TF-IDF+L2의 유사도 결과가 거의 동일한 것은 L2 정규화가 벡터의 **방향(각도)**은 바꾸지 않고 크기만 조정하기 때문임. 코사인 유사도는 방향만 비교하므로 L2 정규화 여부가 결과에 영향을 주지 않음.