import plotly.io as pio
pio.renderers.default = "vscode"  # VS Code·Jupyter 환경에 맞게 설정

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import VarianceThreshold # 분산이 특정 기준(threshold)보다 작은 feature 제거
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import sys
from importlib import reload
sys.path.append(r'C:\Users\seonu\Documents\ewha-marketing_research')
from lib.lib_dtm import lib_filtering_dtm as lfd
reload(lfd)

#=================================
# 공통 설정
#=================================
meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment6\results"

#=================================
# 함수 1. 요인분석 적합성 검정 (KMO & Bartlett)
# - KMO: 변수 간 상관관계가 공통요인으로 설명될 수 있는 정도
#   → 0.6 이상이면 요인분석 적합
# - Bartlett: 상관행렬이 단위행렬(변수 간 상관 없음)과 유의하게 다른지 검정
#   → p < 0.05이면 변수 간 상관이 존재하므로 요인분석 가능
#=================================
def kmo_bartlett_test(data_w_meta_cols, apply_div_by_review_count=True, apply_l2=False, apply_stdscaler=True):

    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    X_scaled = X.copy()
    if apply_div_by_review_count:
        # 리뷰수가 많은 브랜드일수록 단어 빈도가 높아지는 크기 편향 제거
        X_scaled = X_scaled.div(df['review_count'], axis=0)
    if apply_l2:
        # 행별 L2 정규화: 브랜드간 벡터 크기 차이 제거
        X_scaled = normalize(X_scaled, norm="l2", axis=1)
    if apply_stdscaler:
        # 열별 표준화: 고빈도 단어가 분산을 지배하는 현상 방지
        X_scaled = StandardScaler().fit_transform(X_scaled)

    # KMO: kmo_all은 변수별 개별 KMO, kmo_model은 전체 종합 KMO
    kmo_all, kmo_model = calculate_kmo(X_scaled)
    print(f"KMO: {kmo_model:.3f}  (0.6 이상 권장)")

    # Bartlett: H0 = 상관행렬이 단위행렬, H1 = 단위행렬 아님
    chi2, p = calculate_bartlett_sphericity(X_scaled)
    print(f"Bartlett p값: {p:.5f}  (0.05 미만이면 요인분석 가능)")


#=================================
# 함수 2. 희소·저분산 단어 제거
# - 희소 단어: 대부분 브랜드에서 등장하지 않아 상관구조 파악에 노이즈만 추가
# - 저분산 단어: 브랜드 간 값 차이가 거의 없어 요인 형성에 기여하지 못함
# - KMO가 낮거나 모델 수렴 실패 시 적용
#=================================
def filtering_data_via_sparse_variability(data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=80):

    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    # 1) 희소 단어 제거
    # 전체 브랜드 중 sparsity_cutoff_val 이상 비율이 0인 단어 제거
    sparsity = (X == 0).mean(axis=0)
    X_trim = X.loc[:, sparsity < sparsity_cutoff_val]
    print(f"희소 단어 제거: {X.shape[1]}개 → {X_trim.shape[1]}개")

    # 2) 저분산 단어 제거
    # 분산이 vari_cutoff_percentile 백분위수 이하인 단어 제거 → 상위 단어만 유지
    threshold = np.percentile(X_trim.var(), vari_cutoff_percentile)
    sel = VarianceThreshold(threshold=threshold)
    X_var_array = sel.fit_transform(X_trim)
    selected_cols = X_trim.columns[sel.get_support()]
    print(f"저분산 단어 제거: {X_trim.shape[1]}개 → {len(selected_cols)}개")

    X_var = pd.DataFrame(X_var_array, index=X.index, columns=selected_cols)

    return pd.concat([df[meta_cols], X_var], axis=1).reset_index()


#=================================
# 함수 3. 요인 수 결정
# - 상관행렬의 고유값(eigenvalue): 각 요인이 설명하는 분산의 크기
# - Kaiser 기준: 고유값 > 1 인 요인만 채택 (변수 1개 이상의 분산 설명)
# - Scree Plot: 고유값을 내림차순으로 그려 '팔꿈치' 지점에서 요인 수 결정
#=================================
def determine_n_factors(data_w_meta_cols, apply_div_by_review_count=True, apply_l2=False, apply_stdscaler=True):

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

    # rotation=None: 회전 전 고유값 기준으로 요인 수 판단
    # 회전 후에는 요인별 설명 분산이 재분배되어 고유값이 달라질 수 있음
    fa_test = FactorAnalyzer(rotation=None, method='principal')
    fa_test.fit(X_scaled)
    eigenvals_raw, _ = fa_test.get_eigenvalues()
    eigenvals = np.sort(eigenvals_raw)[::-1]  # 내림차순 정렬

    k_kaiser = int((eigenvals > 1).sum())
    print(f"Kaiser 기준 요인 수 (고유값 > 1): {k_kaiser}개")

    # Scree Plot
    x_seq = np.arange(1, len(eigenvals) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_seq, y=eigenvals,
        mode='markers+lines',
        name='고유값',
        marker=dict(symbol='circle', size=7)
    ))
    # Kaiser 기준선: y=1 이상인 요인만 의미있다고 판단
    fig.add_hline(
        y=1,
        line=dict(color='gray', dash='dash'),
        annotation_text='Eigenvalue = 1 (Kaiser 기준)',
        annotation_position='bottom right',
        annotation_font_color='gray'
    )
    fig.update_layout(
        title='Scree Plot - Japanese Restaurants NV',
        xaxis_title='Factor Number',
        yaxis_title='Eigenvalue',
        xaxis=dict(dtick=1),
        template='plotly_white',
        width=750, height=480
    )
    return k_kaiser, fig


#=================================
# 함수 4. 요인모델 학습
# - FactorAnalyzer로 요인적재행렬(Λ)과 요인점수행렬(F) 계산
# - factor_loadings: 각 단어가 각 요인과 얼마나 강하게 연관되는지 (단어×요인)
# - brand_factor_scores: 각 브랜드가 각 요인에서 얼마나 높은 점수를 갖는지 (브랜드×요인)
#=================================
def traing_factor_model(data_w_meta_cols, n_factors, rotation_method='varimax',
                        apply_div_by_review_count=True, apply_l2=False, apply_stdscaler=True, method='uls'):

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

    # 요인적재행렬: 단어와 요인 간 상관계수 (일종의 가중치)
    # 절대값이 클수록 해당 단어가 그 요인을 강하게 대표함
    factor_loadings = pd.DataFrame(
        fa.loadings_,
        index=X.columns,
        columns=[f"F{i+1}" for i in range(n_factors)]
    ).round(5)

    # 요인점수: 학습된 적재행렬을 바탕으로 각 브랜드의 요인별 점수 추정
    factor_scores_arr = fa.transform(X_scaled)
    score_cols = [f"F{i+1}_score" for i in range(n_factors)]
    factor_scores = pd.DataFrame(
        factor_scores_arr,
        index=X.index,
        columns=score_cols
    ).round(5)

    brand_factor_scores = pd.concat([df[meta_cols], factor_scores], axis=1)

    # 결과 저장
    factor_loadings.reset_index().to_csv(
        f"{PATH_to_save}/efa_japanese_nv_factor_loadings_{rotation_method}.csv",
        encoding='utf-8-sig', index=False)
    brand_factor_scores.reset_index().to_csv(
        f"{PATH_to_save}/efa_japanese_nv_brand_factor_scores_{rotation_method}.csv",
        encoding='utf-8-sig', index=False)

    return factor_loadings, brand_factor_scores


#=================================
# 함수 5. 유의미한 factor loading 추출
# - loading 절대값 >= loading_cutoff 인 단어만 남김
# - 각 요인의 양(+)/음(-) 방향 상위 단어를 정리해 요인 의미 해석에 활용
#=================================
def extracting_sig_loadings(factor_loadings, loading_cutoff_value=0.3):

    # cutoff 미만 loading은 NaN 처리 → 모든 요인에서 NaN인 단어 제거
    mask = factor_loadings.abs() >= loading_cutoff_value
    loadings_sig = factor_loadings.where(mask).dropna(how='all', axis=0)

    def summary_top_words(loadings_sig, top_n=10):
        out = {}
        for f in loadings_sig.columns:
            s = loadings_sig[f].dropna()
            if s.empty:
                continue
            # 양(+) 방향: 해당 요인과 정적으로 연관된 단어 (요인 강도 높음)
            pos = s[s > 0].sort_values(ascending=False).head(top_n).index.tolist()
            # 음(-) 방향: 해당 요인과 부적으로 연관된 단어 (요인 강도 낮음)
            neg = s[s < 0].sort_values().head(top_n).index.tolist()
            out[f] = {'pos': pos, 'neg': neg}
        return out

    factor_words = summary_top_words(loadings_sig)

    records = [
        {
            'factor': f,
            'pos_loading_words': ';'.join(d['pos']),
            'neg_loading_words': ';'.join(d['neg'])
        }
        for f, d in factor_words.items()
    ]
    df_factor_words = pd.DataFrame(records).sort_values('factor').reset_index(drop=True)

    return loadings_sig, df_factor_words


#=================================
# 함수 6. 브랜드 포지션 맵 시각화
# - 선택한 두 요인을 x, y축으로 브랜드 포지션 산점도
# - 버블 크기: review_count (브랜드 규모), 색상: avg_stars (평점)
#=================================
def drow_factor_map(brand_factor_scores, x_factor, y_factor, size_col='review_count', color_col='avg_stars'):

    x_col = f"{x_factor}_score"
    y_col = f"{y_factor}_score"

    fig = px.scatter(
        brand_factor_scores.reset_index(),
        x=x_col, y=y_col,
        size=size_col,
        color=color_col,
        color_continuous_scale='Reds',
        hover_data=['name'],
    )
    fig.update_layout(
        title=f"Brand Factor Map - {x_factor} vs {y_factor} (Japanese Restaurants NV)",
        xaxis_title=x_factor,
        yaxis_title=y_factor,
        template='plotly_white'
    )
    return fig

#=================================
# Step 0. 데이터 불러오기 - Japanese 카테고리 필터링
# NV 특화도 1위(+3.74%p) 카테고리로 HW3 근거 기반 선택
#=================================
input_data_filtering_conditions = dict(
    input_file_name = "reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm",
    remove_brand_w_word_in_name = False,
    brand_categories_slted = ["Japanese"],
    words_to_delete = [],
    words_to_include_exclusively = [],
)
data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions=input_data_filtering_conditions)
print("shape:", data_w_meta_cols.shape)

#=================================
# Step 1. 요인분석 적합성 검정
# 전처리 조건: 리뷰수 나누기 적용 + l2 미적용 + StandardScaler 적용
#=================================
kmo_bartlett_test(
    data_w_meta_cols=data_w_meta_cols,
    apply_div_by_review_count=True,
    apply_l2=False,
    apply_stdscaler=True
)

#=================================
# Step 2. 희소·저분산 단어 제거 후 재검정
# KMO nan 원인: 희소 단어가 많아 상관행렬 계산 불안정
# → 희소·저분산 단어 제거 후 재검정
#=================================
data_w_meta_cols_filtered = filtering_data_via_sparse_variability(
    data_w_meta_cols,
    sparsity_cutoff_val=0.95,  # 95% 이상 브랜드에서 0인 단어 제거
    vari_cutoff_percentile=50  # 분산 하위 50% 단어 제거
)

kmo_bartlett_test(
    data_w_meta_cols=data_w_meta_cols_filtered,
    apply_div_by_review_count=True,
    apply_l2=False,
    apply_stdscaler=True
)

#=================================
# Step 3. 요인 수 결정
# - Kaiser 기준(고유값 > 1)과 Scree Plot elbow 지점을 함께 고려
# - 필터링된 데이터 기준으로 고유값 계산
# 팔꿈치(elbow) 지점 = 고유값 감소폭이 급격히 줄어들면서 곡선이 완만해지기 시작하는 지점. 즉 그래프가 "꺾이는" 부분
#=================================
k_kaiser, scree_fig = determine_n_factors(
    data_w_meta_cols=data_w_meta_cols_filtered,
    apply_div_by_review_count=True,
    apply_l2=False,
    apply_stdscaler=True
)
scree_fig.show()

#=================================
# Step 3. 요인 수 결정 - Scree Plot 재시각화
# x축 범위를 30으로 제한하고 틱 간격 조정
#=================================
k_kaiser, scree_fig = determine_n_factors(
    data_w_meta_cols=data_w_meta_cols_filtered,
    apply_div_by_review_count=True,
    apply_l2=False,
    apply_stdscaler=True
)

# x축 범위 제한 (고유값 1 이상 구간만 집중해서 보기)
scree_fig.update_layout(
    xaxis=dict(range=[1, 30], dtick=2)  # 30까지만 표시, 2 간격
)
scree_fig.show()

#=================================
# Step 4. 요인모델 학습 - 요인 수 후보 비교 (n=3, 5, 7)
# 각 요인 수별 factor_words 확인 후 해석 가능성 기준으로 최적 요인 수 결정
#=================================
for n in [3, 5, 7]:
    print(f"\n{'='*50}")
    print(f"n_factors = {n}")
    print('='*50)

    factor_loadings_tmp, _ = traing_factor_model(
        data_w_meta_cols=data_w_meta_cols_filtered,
        n_factors=n,
        rotation_method='varimax',  # 직교회전: 요인 간 독립 가정, 해석 단순화
        method='uls',               # 공통분산 기반 추출
        apply_div_by_review_count=True,
        apply_l2=False,
        apply_stdscaler=True
    )

    _, factor_words_tmp = extracting_sig_loadings(
        factor_loadings=factor_loadings_tmp,
        loading_cutoff_value=0.3
    )
    print(factor_words_tmp.to_string())