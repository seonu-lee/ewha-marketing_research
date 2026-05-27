
'''
주성분분석 - 브랜드별 pca score, 주성분별 주요단어 추출
회귀분석 - 주성분을 독립변수로 사용하여 회귀분석
결과해석 - 군집별 상위키워드, 회귀분석 바탕으로 해석
'''

import pandas as pd
import statsmodels.api as sm
import plotly.express as px

import sys
from importlib import reload
sys.path.append('')
from lib.lib_dtm import lib_filtering_dtm as lfd
from s06_pca import a1_pca_perBrand
from s10_reg.lib import lib_regression as lreg
reload(lfd);reload(a1_pca_perBrand);reload(lreg)

### 공통설정
PATH_to_save = ""
meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 

#------------------------------
# 1) 데이터 불러오기 
#------------------------------
input_data_filtering_conditions = dict(
    input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2",
    remove_brand_w_word_in_name = False,
    brand_categories_slted = [],
    words_to_delete = [],
    words_to_include_exclusively = [],
    )
data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

#------------------------------
# 2) 주성분 분석
#------------------------------
### pca score 추출
pca_score_w_meta_cols, pca_coeff_df = a1_pca_perBrand.calculate_pca_coeff_score_perBrand(
    data_w_meta_cols=data_w_meta_cols, 
    apply_stdscaler=True, 
    apply_l2=True, 
    num_comp_to_extract=10, # 추출할 pca component 수
    )
### 각 주성분의 주요단어 추출
top_words_df = a1_pca_perBrand.get_top_words_per_pc(pca_coeff_df, top_n=10)

#------------------------------
# 3) 회귀 분석 전처리
#------------------------------
### 독립변수 선정 및 전처리: pca score를 독립변수
df = pca_score_w_meta_cols.copy().set_index('name')    
meta_cols = [col for col in df.columns if col in meta_cols_pool]

# 독립변수 선정 (X)
variables_for_reg = [col for col in df.columns if col not in meta_cols_pool] # 단어변수
X = df[variables_for_reg] # 독립변수로 포함할 컬럼만 선택    

# X 스케일 변환 및 상수항 추가
X_scaled = X.copy() 
X_scaled = sm.add_constant(X_scaled) # 상수항 추가 (intercept)

# 다중공선성 확인
vif_df = lreg.calc_vif(X=X_scaled)
vif_df.head(20)

# x 변수 분포확인
fig = px.histogram(x=X["x0"], nbins=30); fig.show()

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

