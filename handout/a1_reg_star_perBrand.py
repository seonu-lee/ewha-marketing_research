'''
1. 회귀분석 개요
1.1 종속 변수
    - 만족도(star ratings)
1.2 독렵 변수
    - DTM(Document-Term Matrix)에서 추출한 단어 빈도
    - 토픽 모델링 결과, 요인분석(EFA) 기반 잠재 요인
    - 단어 기반 변수는 차원이 매우 크고 희소(sparse)하기 때문에 변수 선택 또는 차원 축소가 자주 필요함

1.3 데이터 전처리
1.3.1 tf vs tfidf
    - tfidf는 단어 빈도뿐 아니라 전체 문서 내 희소성을 반영한 가중치를 사용함
    - 따라서 회귀계수는 "실제 단어 빈도 증가"가 아니라 상대적 중요도 증가에 대한 효과로 해석됨
    - 해석의 직관성이 tf 대비 낮을 수 있음
    
1.3.2 열별 standard scaling
    - 표준화 계수, 독립변수 1SD 증가할때 종속변수의 변화량을 의미
    - 상호작용 term이 있을 경우, mean centering 효과가 있어 다중공선성을 줄임
1.3.3 행별 l2 정규화
    - 행별 정규화는 변수 간 공분산 구조를 변화시킬 수 있으므로 주의가 필요함
    - review count의 영향을 통제하려는 목적이라면 review count를 통제변수(control variable)로 추가하는 방법을 사용할 수 있음
    - 리뷰 수 차이에 따른 이분산(heteroscedasticity)을 보정하려는 경우에는 weighted least squares(WLS)를 적용할 수 있음

2. 이슈
2.1 다중공선성 (multicolinearity)
    - 단어들이 함께 등장(co-occurrence)할 경우 높은 상관관계를 가지기 쉬움
    - VIF(Variance Inflation Factor)
        : 특정 변수가 다른 독립변수들로 얼마나 설명되는지를 나타내는 지표
        : 일반적으로 VIF > 10이면 심한 다중공선성으로 간주    

    - 개별 계수(부호/크기)의 불안정: 데이터, 모델이 조금만 바뀌어도 부호가 뒤집히거나 크기가 요동칠 수 있음
    - 표준오차가 증가하여, p-값 증가: 중요한 변수가 유의미하지 않게 보일 수 있음
    - 변수 선별, 차원 축소 적용

    - (참조) vif 계산방법: 
        1) 특정 변수 X_j 를 종속변수로 두고 나머지 독립변수들로 회귀분석 수행
        2) 해당 회귀의 결정계수(R²_j) 계산
        3) VIF_j = 1 / (1 - R²_j)

2.2 이분산 (heteroscedasticity)
    - 리뷰 수가 많은 브랜드의 평균 평점은 신뢰도 높고 (오차 분산 작음), 리뷰 수가 적은 브랜드는 신뢰도가 낮음 (오차 분산 큼) 
        -> 전형적인 이분산 발생
    - 문제: 이분산이 존재하면 회귀계수 자체보다 t-value, p-value, 신뢰구간 해석에 문제가 발생함

    - WLS: 각 관측치별로 가중치(weight)를 다르게 주는 회귀분석을 함 (이분산을 발생시키는 요인을 알고 있을 때)
    - heteroscedasticity robust standard error 적용: 
        - 오차가 등분산이 아니더라도 robust한 방식으로 표준오차를 추정
        - 회귀계수 자체는 그대로 두고 이분산에 강건한 표준오차를 계산하는 방법

2.3 word frequency 분포
    - 대부부의 값이 0 또는 매우 작은 값으로 치우친 분포를 가짐 
    - tf (빈도) 로그 변환을 통해 극단값의 영향을 완화하고 분포 왜도를 줄여 선형회귀의 안정성을 높임 (일부 단어의 과도한 영향력을 감소시킬 수 있음)

'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import plotly.express as px

import sys
from importlib import reload
sys.path.append('')
from lib.lib_dtm import lib_filtering_dtm as lfd
from s10_reg.lib import lib_regression as lreg
reload(lfd); reload(lreg)

### 공통설정
PATH_to_save = ""
meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 

#------------------------------
# 1) 데이터 불러오기 
#------------------------------
input_data_filtering_conditions = dict(
    input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm",
    remove_brand_w_word_in_name = False,
    brand_categories_slted = [],
    words_to_delete = [],
    words_to_include_exclusively = [],
    )
data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

#------------------------------
# 2) 회귀 분석 전처리
#------------------------------
df = data_w_meta_cols.copy().set_index('name')    
meta_cols = [col for col in df.columns if col in meta_cols_pool]

# 독립변수 선정 (X)
variables_for_reg = [col for col in df.columns if col not in meta_cols_pool] # 단어변수
X = df[variables_for_reg] # 독립변수로 포함할 컬럼만 선택    

# X = X.drop(columns=['asada']) # 변수 제거

# X 스케일 변환 및 상수항 추가
X_scaled = X.copy() 
X_scaled = np.log(X_scaled+1) # 로그변환 - **NOTE:변환 하는 경우와 하지 않는 경우 결과 차이 시연할것**
X_scaled = StandardScaler().fit_transform(X_scaled) # 각 단어(열) 별로 표준화 (value-mean)/stderror **NOTE:변환 하는 경우와 하지 않는 경우 결과 차이 시연할것** 
X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns) # dataframe으로 변환(변수명 정보 위해)    
X_scaled = sm.add_constant(X_scaled) # 상수항 추가 (intercept)

# 다중공선성 확인
vif_df = lreg.calc_vif(X=X_scaled)
vif_df.head(20)

# (참고) x변수 log변환 전후 분포 확인
fig = px.histogram(x=X["atmospher"], nbins=30); fig.show()
fig = px.histogram(x=np.log(X["atmospher"]+1), nbins=30); fig.show()

# 3) 종속변수, 가중치변수 선정
y = df["avg_stars"] # 종속 변수
w = df["review_count"].clip(lower=1).astype(float)  # WLS 가중치, 하한값 1로 설정(이하 값은 모두 1로 변환)

#------------------------------
# 3) 회귀 분석
#------------------------------
reg_result, reg_result_df = lreg.reg_analysis(y=y, X_scaled=X_scaled, w=w)
reg_result.summary()
df_sorted = reg_result_df.reindex(reg_result_df['coef'].abs().sort_values(ascending=False).index) # coef 절대값 기준 정렬
df_sorted.head(50)

reg_result_df.sort_values(by=['p_value']).head(50)  # p_value, 


#------------------------------------
# 추가 분석: review count 를 독립변수로 사용
#------------------------------------
X_scaled['log_review_count'] = np.log(df["review_count"])
X_scaled['log_review_count_sq'] = X_scaled['log_review_count']**2

reg_result, reg_result_df = lreg.reg_analysis(y=y, X_scaled=X_scaled, w="")
reg_result.summary()
# df_sorted = reg_result_df.reindex(reg_result_df['coef'].abs().sort_values(ascending=False).index) # coef 절대값 기준 정렬
# df_sorted.head(50)


#------------------------------------
# 전체 브랜드 + 선택 키워드
#------------------------------------
## 키워드 분류
# 요리 카테고리 및 국적 (Cuisine & Ethnicity)
cuisine_tags = [
    'asian', 'chines', 'indian', 'japanes', 'thai', 'lo', 'teriyaki', 'pad',  # 아시아
    'mexican', 'asada', 'carn', 'chile', 'chili', 'enchilada', 'guacamol', 'salsa', 'taco', 'tortilla', 'quesadilla', # 멕시칸
    'italian', 'greek', 'bruschetta', 'hummu', 'pita', # 유럽/지중해
    'bbq', 'burger', 'diner', 'hawaiian', 'philli', 'countri' # 미국/기타
]
# 식재료 및 세부 메뉴 (Ingredients & Menu)
menu_ingredients = [
    'bacon', 'beef', 'chicken', 'crab', 'duck', 'fish', 'lamb', 'lobster', 'pork', 'rib', 'salmon', 'seafood', 'shrimp', 'steak', 'turkey', # 단백질
    'biscuit', 'bread', 'bun', 'crust', 'noodl', 'pasta', 'pizza', 'rice', 'toast', 'waffl', 'pancak', 'pie', # 탄수화물
    'bean', 'butter', 'chees', 'corn', 'garlic', 'mushroom', 'oliv', 'onion', 'potato', 'salad', 'sauc', 'sausag', 'tofu', 'tomato', 'veggi', 'vegan', 'vegetarian', # 채소 및 양념
    'cake', 'chocol', 'coffe', 'cooki', 'cream', 'fruit', 'ice', 'juic', 'pastri', 'smoothi', 'tea', 'yogurt' # 디저트/카페
]
# 주류 및 음료 (Alcohol & Drinks)
drinks_alcohol = [
    'beer', 'brew', 'cocktail', 'margarita', 'martini', 'wine', 'tap', 'glass', 'bottl'
]
# 식사 유형 및 서비스 형태 (Service & Format)
service_format = [
    'breakfast', 'brunch', 'lunch', 'dinner', 'happi', 'night', # 시간대
    'bakeri', 'bistro', 'buffet', 'cafe', 'club', 'deli', 'dive', 'pub', 'shop', 'store', 'truck', # 업종 형태
    'counter', 'deliv', 'deliveri', 'drive', 'order', 'reserv', 'select', 'waiter', 'waitress', 'carry', 'cart' # 서비스 방식
]
# 장소 및 분위기 (Location & Atmosphere)
location_atmosphere = [
    'airport', 'chandler', 'downtown', 'glendal', 'hill', 'mall', 'mesa', 'neighborhood', 'scottsdal', 'phoenix', # 지역 및 위치
    'atmospher', 'decor', 'insid', 'outsid', 'patio', 'room', 'view', 'fountain', 'park', 'garden', # 공간 특징
    'bar', 'hotel', 'kitchen', 'loung', 'market', 'offic', 'resort', 'station', 'tv', # 시설
    'band', 'danc', 'event', 'fun', 'game', 'golf', 'live', 'movi', 'music', 'play', 'pool', 'sport', 'watch' # 활동/문화
]
# 고객 경험 및 가치 평가 (Experience & Evaluation)
experience_eval = [
    'authent', 'beauti', 'cheap', 'clean', 'decent', 'excel', 'favorit', 'healthi', 'perfect', 'qualiti', 'special', 'super', 'tasti', 'wonder', 'worth', # 주관적 평가
    'card', 'cash', 'charg', 'coupon', 'free', 'groupon', 'pay', 'price', 'tip', # 비용 관련
    'actual', 'arriv', 'call', 'decid', 'employe', 'expect', 'famili', 'guy', 'husband', 'kid', 'manag', 'owner', 'person', 'server', 'wife', # 상황/관계
    'big', 'enough', 'half', 'huge', 'larg', 'portion', 'top', 'small' # 양/사이즈
]

## 0) 분석조건
words_to_include_exclusively = sorted(list(set(experience_eval)))

input_data_filtering_conditions = dict(
    input_file_name = "reviews_restaurants_az_perBrand_0.1_1.0_0.3_10_dtm",
    remove_brand_w_word_in_name = False,
    brand_categories_slted = [],
    words_to_delete = [],
    words_to_include_exclusively = words_to_include_exclusively,
    )
data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)





