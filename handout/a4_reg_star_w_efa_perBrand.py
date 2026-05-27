'''
efa분석 - 브랜드별 efa scores, factor별 주요단어 추출
회귀분석 - efa score를 독립변수로 하여 회귀분석
결과해석 - factor별 주요단어, 회귀분석 바탕으로 해석
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
from s09_efa import a1_efa_perBrand
from s10_reg.lib import lib_regression as lreg
reload(lfd);reload(a1_efa_perBrand);reload(lreg)

### 공통설정
PATH_to_save = ""
meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 

#------------------------------
# 1) 데이터 불러오기 
#------------------------------
input_data_filtering_conditions = dict(
    input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm",
    remove_brand_w_word_in_name = False,
    brand_categories_slted = ['Italian'], # Mexican, Fast Food
    words_to_delete = [],
    words_to_include_exclusively = [],
    )
data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

#------------------------------
# 2) 요인분석
#------------------------------
### 요인분석 전처리 조건
apply_div_by_review_count=True; apply_l2=False; apply_stdscaler=True

### 요인분석 전처리
# 데이터 적합성 점검
a1_efa_perBrand.kmo_bartlett_test(data_w_meta_cols=data_w_meta_cols, apply_div_by_review_count=apply_div_by_review_count, apply_l2=apply_l2, apply_stdscaler=apply_stdscaler)

# 데이터 적합성이 문제가 있을 경우, dtm 에서 희소·저분산 단어 제거  
data_w_meta_cols = a1_efa_perBrand.filtering_data_via_sparse_variability(data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=50)

### 요인 수(k) 결정
k_kaiser, scree_fig = a1_efa_perBrand.determine_n_factors(data_w_meta_cols=data_w_meta_cols, apply_div_by_review_count=apply_div_by_review_count, apply_l2=apply_l2, apply_stdscaler=apply_stdscaler)
scree_fig.show()

### 요인모델 적합 & 회전
n_factors = 10   # 위 단계에서 결정한 k 값 # 5, 10
method = "principal" # 'principal', 'uls'/'minres', 'ml'/'mle'
rotation_method = 'oblimin' # varimax, promax, oblimin
factor_corr_matrix, factor_loadings, brand_factor_scores = a1_efa_perBrand.traing_factor_model(
    data_w_meta_cols=data_w_meta_cols, n_factors=n_factors, rotation_method=rotation_method, 
    apply_div_by_review_count=apply_div_by_review_count, apply_l2=apply_l2, apply_stdscaler=apply_stdscaler, method=method
    )

### 유의미한 factor loading 추출
factor_loadings_sig, factor_words = a1_efa_perBrand.extracting_sig_loadings(loadings=factor_loadings, loading_cutoff_value=0.3)
factor_words

#------------------------------
# 3) 회귀 분석 전처리
#------------------------------
### 독립변수 선정 및 전처리: pca score를 독립변수
df = brand_factor_scores.copy()   
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
fig = px.histogram(x=X["F1_score"], nbins=30); fig.show()

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

    