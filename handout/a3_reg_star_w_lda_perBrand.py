'''
lda분석 - 브랜드별 토픽확률분포, 토픽별 주요단어 추출
회귀분석 - 토픽확률분포를 독립변수로 하여 회귀분석
결과해석 - 토픽별 주요단어, 회귀분석 바탕으로 해석

'''


import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import plotly.express as px

import sys
from importlib import reload
sys.path.append('')
from lib.lib_dtm import lib_filtering_dtm as lfd
from s08_topic_model import a1_lda
from s10_reg.lib import lib_regression as lreg
reload(lfd);reload(a1_lda);reload(lreg)

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
# 2) lda 분석
#------------------------------
### 최적 parameter로 lda 모델 학습
params_slted = dict(
    n_components_slted = 11, # 5, 7, 11
    doc_topic_prior_slted = None, 
    topic_word_prior_slted = None
    )
lda_best, topic_word_prob_df, document_topic_prob_w_meta_cols = a1_lda.traing_lda_best(data_w_meta_cols, params_slted)

### 토픽별 Top‑N 키워드 추출
topic_kws = a1_lda.extract_topic_keywords(topic_word_prob_df, method = "phi_excl", top_n=10)

#------------------------------
# 3) 회귀 분석 전처리
#------------------------------
### 독립변수 선정 및 전처리: pca score를 독립변수
df = document_topic_prob_w_meta_cols.copy()
meta_cols = [col for col in df.columns if col in meta_cols_pool]

# 독립변수 선정 (X)
# lda는 토픽들의 확률분포여서 합이 1임 - perfect multicollinearity 문제발생. 마지막 토픽을 삭제함
# 결과해석 - 제거된 토픽 대비 각 토픽이 미치는 영향
variables_for_reg = [col for col in df.columns if col not in meta_cols_pool+["main_topic"]]
X = df[variables_for_reg[:-1]] # 독립변수로 포함할 컬럼만 선택, 맨 마지막 토픽 컬럼은 제거   

# X 스케일 변환 및 상수항 추가
X_scaled = X.copy() 
X_scaled = sm.add_constant(X_scaled) # 상수항 추가 (intercept)

# 다중공선성 확인
vif_df = lreg.calc_vif(X=X_scaled)
vif_df.head(20)

# x 변수 분포확인
fig = px.histogram(x=X["Topic1"], nbins=30); fig.show()

### 종속변수, 가중치변수 선정
y = df["avg_stars"] # 종속 변수
w = df["review_count"].clip(lower=1).astype(float)  # WLS 가중치, 하한값 1로 설정(이하 값은 모두 1로 변환)

#------------------------------
# 4) 회귀 분석
#------------------------------
reg_result, reg_result_df = lreg.reg_analysis(y=y, X_scaled=X_scaled, w=w)
reg_result.summary()
df_sorted = reg_result_df.reindex(reg_result_df['coef'].abs().sort_values(ascending=False).index) # coef 절대값 기준 정렬
df_sorted.head(50)

