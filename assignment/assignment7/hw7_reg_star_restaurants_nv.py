#=================================
# HW7: Regression Analysis
# - 종속변수: avg_stars (브랜드 평균 평점)
# - 분석 단위: 브랜드
#
# [분석 조건]
# - Part 1: NV 전체 레스토랑 (3,891개 브랜드), 단어(TF DTM) 독립변수
#   → NV Steakhouses(204개)는 변수 수(323개) > 관측치로 과적합 발생
#   → 관측치 확보를 위해 NV 전체 브랜드 사용
# - Part 2: NV Steakhouses (204개 브랜드), EFA 요인점수 독립변수
#   → 요인점수 5개로 변수 수 << 관측치 조건 충족
#   → HW6 EFA 분석 조건 동일하게 적용
#=================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from factor_analyzer import FactorAnalyzer
from sklearn.feature_selection import VarianceThreshold
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"

import sys
sys.path.append(r"C:\Users\seonu\Documents\ewha-marketing_research")
from lib.lib_dtm import lib_filtering_dtm as lfd
from lib.lib_regression import calc_vif, reg_analysis

#=================================
# 공통 설정
#=================================
meta_cols_pool = ['name', 'review_count', 'avg_stars', 'useful_count',
                  'funny_count', 'cool_count', 'categories']
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment7\results"


#=================================
# Part 1. 단어를 독립변수로 한 회귀분석
# - 데이터: NV 전체 레스토랑 (3,891개 브랜드, 323개 단어)
# - 전처리: 로그변환 + StandardScaler
# - WLS (review_count 가중치) + HC3 robust std error
#=================================

# Step 1. 데이터 불러오기 - NV 전체 레스토랑
input_data_filtering_conditions_nv = dict(
    input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm',
    remove_brand_w_word_in_name=False,
    brand_categories_slted=[],
    words_to_delete=[],
    words_to_include_exclusively=[],
)
data_w_meta_cols_nv = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions_nv)
print(f"NV 전체 레스토랑 브랜드 수: {len(data_w_meta_cols_nv)}개")

# Step 2. 독립변수(X) 준비
df1 = data_w_meta_cols_nv.copy().set_index('name')
meta_cols = [col for col in df1.columns if col in meta_cols_pool]
X1 = df1.drop(columns=meta_cols)  # 전체 단어 독립변수 (323개)

# Step 3. 전처리
# - 로그변환: 단어 빈도의 심한 우편향 분포 완화, 고빈도 단어 과도한 영향 억제
# - StandardScaler: 단어 간 스케일 차이 제거, 표준화 계수로 변수 간 영향력 비교 가능
X1_scaled = np.log(X1 + 1)
X1_scaled = StandardScaler().fit_transform(X1_scaled)
X1_scaled = pd.DataFrame(X1_scaled, index=X1.index, columns=X1.columns)
X1_scaled = sm.add_constant(X1_scaled)

# Step 4. 다중공선성 확인
# - 단어 수(323개) < 관측치(3,891개) → 과적합 문제 없음
# - 단어 간 공동출현으로 다중공선성 발생 가능, VIF > 10이면 해석 주의
print("\n===== VIF 상위 20개 =====")
vif_df1 = calc_vif(X=X1_scaled)
print(vif_df1.head(20))

# Step 5. 종속변수 및 가중치 설정
# - WLS: 리뷰 수 많을수록 평균 평점 신뢰도 높음 → 이분산 보정
y1 = df1['avg_stars']
w1 = df1['review_count'].clip(lower=1).astype(float)

# Step 6. 회귀분석 (WLS + HC3 robust std error)
print("\n===== Part 1 회귀분석 결과 =====")
reg_result_1, reg_result_df_1 = reg_analysis(y=y1, X_scaled=X1_scaled, w=w1)
print(reg_result_1.summary())

# 계수 절대값 기준 상위 단어 확인
df_sorted_1 = reg_result_df_1.reindex(
    reg_result_df_1['coef'].abs().sort_values(ascending=False).index)
print("\n===== 계수 절대값 상위 20개 =====")
print(df_sorted_1.head(20))

# p-value 기준 유의미한 단어 확인 (p < 0.05)
print("\n===== 유의미한 변수 (p < 0.05) =====")
sig_vars_1 = reg_result_df_1[reg_result_df_1['p_value'] < 0.05].sort_values('p_value')
print(sig_vars_1)


#=================================
# Part 2. EFA 요인점수를 독립변수로 한 회귀분석
# - 데이터: NV Steakhouses (204개 브랜드)
# - HW6 EFA 동일 조건: TF + 리뷰수 나누기 + L2 미적용 + StdScaler
# - 단어 필터링: sparsity 0.95 + 분산 하위 80% → 64개 단어
# - n_factors=5, method=uls, rotation=varimax
#=================================

# Step 7. 데이터 불러오기 - NV Steakhouses
input_data_filtering_conditions_steak = dict(
    input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm',
    remove_brand_w_word_in_name=False,
    brand_categories_slted=['Steakhouses'],
    words_to_delete=[],
    words_to_include_exclusively=[],
)
data_w_meta_cols_steak = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions_steak)
print(f"\nSteakhouses 브랜드 수: {len(data_w_meta_cols_steak)}개")

# Step 8. EFA 전처리 함수 정의
def filtering_data_via_sparse_variability(data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=50):
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)
    sparsity = (X == 0).mean(axis=0)
    X_trim = X.loc[:, sparsity < sparsity_cutoff_val]
    print(f"희소 단어 제거 후: {X_trim.shape[1]}개 단어")
    sel = VarianceThreshold(threshold=np.percentile(X_trim.var(), vari_cutoff_percentile))
    X_var_array = sel.fit_transform(X_trim)
    selected_cols = X_trim.columns[sel.get_support()]
    X_var = pd.DataFrame(X_var_array, index=X.index, columns=selected_cols)
    print(f"저분산 단어 제거 후: {X_var.shape[1]}개 단어")
    return pd.concat([df[meta_cols], X_var], axis=1).reset_index()

def traing_factor_model(data_w_meta_cols, n_factors, rotation_method='varimax',
                        apply_div_by_review_count=True, apply_l2=False,
                        apply_stdscaler=True, method='uls'):
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)
    X_scaled = X.copy()
    if apply_div_by_review_count:
        X_scaled = X_scaled.div(df['review_count'], axis=0)
    if apply_l2:
        X_scaled = normalize(X_scaled, norm="l2", axis=1)
    if apply_stdscaler:
        X_scaled = StandardScaler().fit_transform(X_scaled)
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation_method, method=method)
    fa.fit(X_scaled)
    factor_loadings = pd.DataFrame(
        fa.loadings_, index=X.columns,
        columns=[f"F{i+1}" for i in range(n_factors)]
    ).round(5)
    factor_scores_arr = fa.transform(X_scaled)
    score_cols = [f"F{i+1}_score" for i in range(n_factors)]
    factor_scores = pd.DataFrame(
        factor_scores_arr, index=X.index, columns=score_cols
    ).round(5)
    brand_factor_scores = pd.concat([df[meta_cols], factor_scores], axis=1)
    return factor_loadings, brand_factor_scores

# Step 9. EFA 수행 (HW6 동일 조건)
data_w_meta_cols_efa = filtering_data_via_sparse_variability(
    data_w_meta_cols_steak, sparsity_cutoff_val=0.95, vari_cutoff_percentile=80)

factor_loadings, brand_factor_scores = traing_factor_model(
    data_w_meta_cols_efa,
    n_factors=5,
    rotation_method='varimax',
    apply_div_by_review_count=True,
    apply_l2=False,
    apply_stdscaler=True,
    method='uls'
)

# Step 10. 독립변수(X) 준비 - EFA 요인점수
# - Varimax 직교회전 → 요인 간 독립 → 다중공선성 낮을 것으로 예상
# - 요인점수는 이미 표준화된 값 → 추가 스케일링 불필요
df2 = brand_factor_scores.copy()
X2 = df2[[f'F{i+1}_score' for i in range(5)]]
X2_scaled = sm.add_constant(X2)

# Step 11. 다중공선성 확인
print("\n===== VIF (EFA 요인점수) =====")
vif_df2 = calc_vif(X=X2_scaled)
print(vif_df2)

# Step 12. 종속변수 및 가중치 설정
y2 = df2['avg_stars']
w2 = df2['review_count'].clip(lower=1).astype(float)

# Step 13. 회귀분석 (WLS + HC3 robust std error)
print("\n===== Part 2 회귀분석 결과 =====")
reg_result_2, reg_result_df_2 = reg_analysis(y=y2, X_scaled=X2_scaled, w=w2)
print(reg_result_2.summary())

# 요인별 회귀계수 정렬
df_sorted_2 = reg_result_df_2.reindex(
    reg_result_df_2['coef'].abs().sort_values(ascending=False).index)
print("\n===== 요인별 회귀계수 =====")
print(df_sorted_2)

# 유의미한 요인 확인 (p < 0.05)
print("\n===== 유의미한 요인 (p < 0.05) =====")
sig_vars_2 = reg_result_df_2[reg_result_df_2['p_value'] < 0.05].sort_values('p_value')
print(sig_vars_2)