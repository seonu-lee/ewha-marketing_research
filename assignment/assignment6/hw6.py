import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
pio.renderers.default = "vscode"

# EFA 관련
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, normalize

# LDA 관련
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import sys
sys.path.append(r"C:\Users\seonu\Documents\ewha-marketing_research")
from lib.lib_dtm import lib_filtering_dtm as lfd

#=================================
# 공통 설정
#=================================
meta_cols_pool = ['name', 'review_count', 'avg_stars', 'useful_count',
                  'funny_count', 'cool_count', 'categories']
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment6\results"


#=================================
# 함수 정의 - EFA
#=================================

# ----------------------------------------
# kmo_bartlett_test: 요인분석 적합성 검정
# - KMO: 0.6 이상이면 요인분석 적합
# - Bartlett: p < 0.05이면 요인분석 적합
# ----------------------------------------
def kmo_bartlett_test(data_w_meta_cols, apply_div_by_review_count=True, apply_l2=False, apply_stdscaler=True):
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

    kmo_all, kmo_model = calculate_kmo(X_scaled)
    chi2, p = calculate_bartlett_sphericity(X_scaled)
    print(f"KMO: {kmo_model:.3f}  (0.6 이상 권장)")
    print(f"Bartlett p값: {p:.5f}  (0.05 미만 권장)")
    return kmo_model, p


# ----------------------------------------
# filtering_data_via_sparse_variability: 희소·저분산 단어 제거
# - KMO가 낮을 경우 적용
# - sparsity_cutoff_val: 0이 차지하는 비율 기준 (예, 0.95 = 95% 이상 0인 단어 제거)
# - vari_cutoff_percentile: 분산 기준 하위 단어 제거 (예, 50 = 하위 50% 제거)
# ----------------------------------------
def filtering_data_via_sparse_variability(data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=50):
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    # 1) 희소 단어 제거
    sparsity = (X == 0).mean(axis=0)
    X_trim = X.loc[:, sparsity < sparsity_cutoff_val]
    print(f"희소 단어 제거 후: {X_trim.shape[1]}개 단어")

    # 2) 저분산 단어 제거
    sel = VarianceThreshold(threshold=np.percentile(X_trim.var(), vari_cutoff_percentile))
    X_var_array = sel.fit_transform(X_trim)
    selected_cols = X_trim.columns[sel.get_support()]
    X_var = pd.DataFrame(X_var_array, index=X.index, columns=selected_cols)
    print(f"저분산 단어 제거 후: {X_var.shape[1]}개 단어")

    return pd.concat([df[meta_cols], X_var], axis=1).reset_index()


# ----------------------------------------
# determine_n_factors: 최적 요인 수 결정
# - Scree plot으로 고유값 시각화
# - Kaiser 기준: 고유값 > 1인 요인 수
# ----------------------------------------
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

    fa_test = FactorAnalyzer(rotation=None, method='principal')
    fa_test.fit(X_scaled)
    eigenvals_raw, _ = fa_test.get_eigenvalues()
    eigenvals = np.sort(eigenvals_raw)[::-1]

    k_kaiser = int((eigenvals > 1).sum())
    print(f"Kaiser 기준 요인 수 (고유값 > 1): {k_kaiser}개")

    x_seq = np.arange(1, len(eigenvals) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_seq, y=eigenvals,
        mode="markers+lines", name="Eigenvalues",
        marker=dict(symbol="circle", size=8)
    ))
    fig.add_hline(y=1, line=dict(color="gray", dash="dash"),
                  annotation_text="Eigenvalue = 1 (Kaiser)",
                  annotation_position="bottom right")
    fig.update_layout(
        title="Scree Plot", xaxis_title="Component number",
        yaxis_title="Eigenvalue", xaxis=dict(dtick=1),
        template="plotly_white", width=750, height=480
    )
    return k_kaiser, fig


# ----------------------------------------
# traing_factor_model: 요인모델 학습 및 결과 계산
# - factor_loadings: 단어별 요인 적재값 (단어×요인)
# - brand_factor_scores: 브랜드별 요인 점수 (브랜드×요인)
# ----------------------------------------
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

    # 요인 적재값
    factor_loadings = pd.DataFrame(
        fa.loadings_, index=X.columns,
        columns=[f"F{i+1}" for i in range(n_factors)]
    ).round(5)

    # 요인 점수
    factor_scores_arr = fa.transform(X_scaled)
    score_cols = [f"F{i+1}_score" for i in range(n_factors)]
    factor_scores = pd.DataFrame(
        factor_scores_arr, index=X.index, columns=score_cols
    ).round(5)

    brand_factor_scores = pd.concat([df[meta_cols], factor_scores], axis=1)

    # 저장
    factor_loadings.reset_index().to_csv(
        f"{PATH_to_save}/factor_loadings_{rotation_method}.csv",
        encoding='utf-8-sig', index=False)
    brand_factor_scores.reset_index().to_csv(
        f"{PATH_to_save}/brand_factor_scores_{rotation_method}.csv",
        encoding='utf-8-sig', index=False)

    return factor_loadings, brand_factor_scores


# ----------------------------------------
# extracting_sig_loadings: 유의미한 factor loading 추출
# - loading_cutoff_value 이상인 단어만 추출
# - 각 요인별 양(+)/음(-) 방향 단어 정리
# ----------------------------------------
def extracting_sig_loadings(loadings, loading_cutoff_value=0.3):
    mask_loading = loadings.abs() >= loading_cutoff_value
    loadings_sig = loadings.where(mask_loading)
    loadings_sig = loadings_sig.dropna(how='all', axis=0)

    def summary_top_words(loadings_sig, top_n=10):
        out = dict()
        for f in loadings_sig.columns:
            s = loadings_sig[f].dropna()
            if s.empty:
                continue
            pos = s[s > 0].sort_values(ascending=False).head(top_n).index.tolist()
            neg = s[s < 0].sort_values().head(top_n).index.tolist()
            out[f] = {'pos': pos, 'neg': neg}
        return out

    factor_words = summary_top_words(loadings_sig, top_n=10)
    records = [
        {"factor": f, "pos_loading_words": ";".join(d["pos"]),
         "neg_loading_words": ";".join(d["neg"])}
        for f, d in factor_words.items()
    ]
    df_factor_words = pd.DataFrame(records).sort_values("factor").reset_index(drop=True)
    return loadings_sig, df_factor_words


# ----------------------------------------
# drow_factor_map: 브랜드 포지셔닝 맵 시각화
# - x_factor, y_factor: 시각화할 요인 선택
# ----------------------------------------
def drow_factor_map(brand_factor_scores, x_factor, y_factor,
                    size_col='review_count', color_col='avg_stars'):
    x_col = f"{x_factor}_score"
    y_col = f"{y_factor}_score"
    fig = px.scatter(
        brand_factor_scores.reset_index(),
        x=x_col, y=y_col,
        size=size_col, color=color_col,
        color_continuous_scale="Reds",
        hover_data=["name"],
    )
    fig.update_layout(
        title=f"Brand Factor Map - {x_factor} vs. {y_factor}",
        xaxis_title=x_factor, yaxis_title=y_factor,
        template="plotly_white"
    )
    return fig


#=================================
# 함수 정의 - LDA
#=================================

# ----------------------------------------
# lda_parma_grid_search: Grid Search로 최적 LDA 파라미터 선정
# - perplexity (낮을수록 좋음): 모델이 데이터를 얼마나 잘 예측하는지
# - coherence (높을수록 좋음): 토픽 내 단어들의 의미적 일관성
# - diversity (높을수록 좋음): 토픽 간 단어 중복 비율
# - exclusivity (높을수록 좋음): 토픽별 전용 단어 비율
# ----------------------------------------
def lda_parma_grid_search(data_w_meta_cols, param_grid, top_n=10):
    meta_cols = [col for col in data_w_meta_cols.columns if col in meta_cols_pool]
    data = data_w_meta_cols.drop(columns=meta_cols)

    count_df = data.copy()
    feature_names = count_df.columns.to_numpy()
    count_matrix = count_df.values.astype(int)

    # Coherence 계산을 위한 토큰 리스트 복원
    token_lists = []
    for row in count_matrix:
        tokens = []
        for idx, cnt in enumerate(row):
            if cnt:
                tokens.extend([feature_names[idx]] * cnt)
        token_lists.append(tokens)
    dictionary = Dictionary(token_lists)

    param_search_results = list()

    for params in ParameterGrid(param_grid):
        print("params: ", params)

        # LDA 학습
        lda = LatentDirichletAllocation(
            **params, learning_method="batch",
            random_state=42, n_jobs=-1, max_iter=5
        )
        lda.fit(count_matrix)

        # 토픽별 상위 단어 추출
        topics = list()
        for comp in lda.components_:
            top_indices = comp.argsort()[::-1][:top_n]
            topics.append([feature_names[i] for i in top_indices])

        # perplexity
        perp = lda.perplexity(count_matrix)

        # coherence
        cm = CoherenceModel(
            topics=topics, texts=token_lists,
            dictionary=dictionary, coherence='c_v'
        )
        c_v = cm.get_coherence()

        # diversity
        flat_terms = [w for topic in topics for w in topic]
        diversity = len(set(flat_terms)) / (len(topics) * top_n)

        # exclusivity
        phi = lda.components_.astype(float)
        phi = phi / phi.sum(axis=1, keepdims=True)
        excl_per_topic = list()
        for k, row in enumerate(phi):
            idx_top = np.argsort(row)[::-1][:top_n]
            excl = row[idx_top] / phi[:, idx_top].sum(axis=0)
            excl_per_topic.append(excl.mean())
        exclusivity = float(np.mean(excl_per_topic))

        param_search_results.append({
            **params,
            "perplexity": perp, "c_v": c_v,
            "diversity": diversity, "exclusivity": exclusivity
        })

    param_search_results_df = pd.DataFrame(param_search_results)
    param_search_results_df = param_search_results_df[[
        'n_components', 'doc_topic_prior', 'topic_word_prior',
        'perplexity', 'c_v', 'diversity', 'exclusivity'
    ]]
    param_search_results_df = param_search_results_df.sort_values(
        ['n_components', 'doc_topic_prior', 'topic_word_prior']
    ).reset_index(drop=True)

    return param_search_results_df


# ----------------------------------------
# lda_parma_grid_search_graph: Grid Search 결과 시각화
# ----------------------------------------
def lda_parma_grid_search_graph(param_search_results_df):
    df = param_search_results_df.copy()
    df["param_id"] = (
        df["n_components"].astype(str) + "_" +
        df["doc_topic_prior"].fillna("None").astype(str) + "_" +
        df["topic_word_prior"].fillna("None").astype(str)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["param_id"], y=df["perplexity"],
        name="Perplexity (↓)", marker_color="#1f77b4",
        yaxis="y", offsetgroup=0
    ))

    metric_color = {
        "c_v": "#ff7f0e", "diversity": "#2ca02c", "exclusivity": "#d62728"
    }
    for i, metric in enumerate(metric_color, start=1):
        fig.add_trace(go.Bar(
            x=df["param_id"], y=df[metric],
            name=f"{metric}", marker_color=metric_color[metric],
            yaxis="y2", offsetgroup=i
        ))

    fig.update_layout(
        title="Grid-Search 결과: Perplexity vs Quality Metrics",
        xaxis=dict(title="Hyper-parameters (K_α_β)", tickangle=-45),
        yaxis=dict(title="Perplexity", rangemode="tozero"),
        yaxis2=dict(title="Quality Metrics", overlaying="y",
                    side="right", rangemode="tozero"),
        barmode="group", bargap=0.15, width=1100, height=550,
        legend=dict(title="지표", orientation="h", x=0, y=-0.2)
    )
    return fig


# ----------------------------------------
# traing_lda_best: 최적 파라미터로 LDA 학습
# - topic_word_prob_df: 토픽-단어 확률 분포
# - document_topic_prob_w_meta_cols: 브랜드-토픽 확률 분포
# ----------------------------------------
def traing_lda_best(data_w_meta_cols, params_slted):
    data_w_meta_cols = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in data_w_meta_cols.columns if col in meta_cols_pool]
    data = data_w_meta_cols.drop(columns=meta_cols)

    count_df = data.copy()
    feature_names = count_df.columns.to_list()
    count_matrix = count_df.values.astype(int)

    lda_best = LatentDirichletAllocation(
        n_components=params_slted['n_components_slted'],
        doc_topic_prior=params_slted['doc_topic_prior_slted'],
        topic_word_prior=params_slted['topic_word_prior_slted'],
        learning_method="batch", max_iter=10,
        n_jobs=-1, random_state=42,
    )
    lda_best.fit(count_matrix)

    # 토픽-단어 확률 분포 (φ)
    phi_raw = lda_best.components_.astype('float')
    phi_norm = phi_raw / (phi_raw.sum(axis=1, keepdims=True) + 1e-12)
    topic_word_prob_df = pd.DataFrame(
        phi_norm, columns=feature_names
    ).rename(index=lambda i: f"Topic{i+1}")

    # 브랜드-토픽 확률 분포 (θ)
    theta = lda_best.transform(count_matrix)
    document_topic_prob_df = pd.DataFrame(
        theta, index=count_df.index
    ).rename(columns=lambda i: f"Topic{i+1}")
    document_topic_prob_df["main_topic"] = document_topic_prob_df.idxmax(axis=1)

    document_topic_prob_w_meta_cols = pd.concat(
        [data_w_meta_cols[meta_cols], document_topic_prob_df], axis=1
    )

    return lda_best, topic_word_prob_df, document_topic_prob_w_meta_cols


# ----------------------------------------
# extract_topic_keywords: 토픽별 Top-N 키워드 추출
# - method: "phi"(절대비중), "excl"(전용성), "phi_excl"(혼합)
# ----------------------------------------
def extract_topic_keywords(topic_word_prob_df, method="phi_excl", top_n=10):
    phi = topic_word_prob_df.values.astype(float)
    vocab = topic_word_prob_df.columns.to_numpy()
    K = phi.shape[0]

    col_sum = phi.sum(axis=0, keepdims=True)
    excl = phi / col_sum

    if method == "phi":
        score = phi
    elif method == "excl":
        score = excl
    elif method == "phi_excl":
        score = phi * excl
    else:
        raise ValueError("method는 'phi', 'excl', 'phi_excl' 중에서 선택")

    keywords = []
    for k in range(K):
        idx = score[k].argsort()[::-1][:top_n]
        keywords.append([vocab[i] for i in idx])

    kw_df = pd.DataFrame(
        keywords, columns=[f"kw{i+1}" for i in range(top_n)]
    ).rename(index=lambda i: f"Topic{i+1}")
    return kw_df


# ----------------------------------------
# doucment_topic_heatmap: 브랜드-토픽 히트맵 시각화
# ----------------------------------------
def doucment_topic_heatmap(document_topic_prob_w_meta_cols, n_document_to_graph):
    meta_cols = [col for col in document_topic_prob_w_meta_cols.columns if col in meta_cols_pool]
    data = document_topic_prob_w_meta_cols.drop(columns=meta_cols)
    heat_df = data.drop(columns=["main_topic"]).head(n_document_to_graph)

    fig = go.Figure(go.Heatmap(
        z=heat_df.values, x=heat_df.columns, y=heat_df.index,
        colorscale="YlGnBu", colorbar=dict(title="Topic proportion"),
        zmin=0, zmax=float(heat_df.values.max()),
    ))
    fig.update_layout(
        title="Brand-wise Topic Distribution (LDA)",
        xaxis_title="Topics", yaxis_title="Restaurant Brands",
        width=900, height=800,
    )
    return fig


#=================================
# CASE: Steakhouses 브랜드 선별
# - HW3~HW5에서 지속적으로 분석해온 NV 특화 카테고리
# - NV vs AZ 비율 +2.41%p, 브랜드 수 204개
#=================================

# Step 1. 데이터 불러오기
# - TF DTM 사용 (LDA는 정수 count 필요, EFA도 동일 데이터 사용)
# - l2 미적용 (EFA에서 l2 적용 시 공동 변동성 사라져 요인 구조 파악 불가)
input_data_filtering_conditions = dict(
    input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm',
    remove_brand_w_word_in_name=False,
    brand_categories_slted=['Steakhouses'],
    words_to_delete=[],
    words_to_include_exclusively=[],
)
data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
print(f"Steakhouses 브랜드 수: {len(data_w_meta_cols)}개")

# 전처리 조건 설정
# - tf: TF DTM 사용
# - 리뷰수 나누기: 리뷰 수에 의한 왜곡된 상관관계 제거
# - l2 미적용: 공동 변동성 유지 (요인분석에 필요)
# - stdscaler: 단어 간 스케일 차이 제거
apply_div_by_review_count = True
apply_l2 = False
apply_stdscaler = True


#=================================
# Part 1. EFA 분석
#=================================

# Step 2. 데이터 적합성 점검
print("\n===== KMO & Bartlett 검정 (원본 데이터) =====")
kmo_bartlett_test(data_w_meta_cols, apply_div_by_review_count, apply_l2, apply_stdscaler)

# KMO가 낮을 경우 희소·저분산 단어 제거 후 재검정
# - sparsity_cutoff_val=0.95: 95% 이상 브랜드에서 0인 단어 제거
# - vari_cutoff_percentile=50: 분산 하위 50% 단어 제거
data_w_meta_cols_efa = filtering_data_via_sparse_variability(
    data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=50)
print("\n===== KMO & Bartlett 검정 (단어 제거 후) =====")
kmo_bartlett_test(data_w_meta_cols_efa, apply_div_by_review_count, apply_l2, apply_stdscaler)

# Step 3. 최적 요인 수 결정
# - Scree plot에서 팔꿈치(elbow) 지점 확인
# - Kaiser 기준(고유값 > 1)은 보조적으로 참고
k_kaiser, scree_fig = determine_n_factors(
    data_w_meta_cols_efa, apply_div_by_review_count, apply_l2, apply_stdscaler)
scree_fig.show()
print(f"Kaiser 기준 요인 수: {k_kaiser}개 → Scree plot 확인 후 최종 결정")

# vari_cutoff_percentile을 높여서 더 많은 단어 제거
data_w_meta_cols_efa = filtering_data_via_sparse_variability(
    data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=80)
print("\n===== KMO & Bartlett 검정 (단어 더 제거 후) =====")
kmo_bartlett_test(data_w_meta_cols_efa, apply_div_by_review_count, apply_l2, apply_stdscaler)

# vari_cutoff_percentile=80으로 업데이트
data_w_meta_cols_efa = filtering_data_via_sparse_variability(
    data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=80)

# Step 3. 최적 요인 수 결정
k_kaiser, scree_fig = determine_n_factors(
    data_w_meta_cols_efa, apply_div_by_review_count, apply_l2, apply_stdscaler)
scree_fig.show()
print(f"Kaiser 기준 요인 수: {k_kaiser}개 → Scree plot 확인 후 최종 결정")

# Step 4. 요인모델 학습 - n_factors 비교
# Scree plot elbow 기준 5개, 7개 비교
for n_factors_test in [5, 7]:
    print(f"\n{'='*50}")
    print(f"n_factors = {n_factors_test}")
    print(f"{'='*50}")
    factor_loadings_test, _ = traing_factor_model(
        data_w_meta_cols_efa,
        n_factors=n_factors_test,
        rotation_method='varimax',
        apply_div_by_review_count=apply_div_by_review_count,
        apply_l2=apply_l2,
        apply_stdscaler=apply_stdscaler,
        method='uls'
    )
    _, factor_words_test = extracting_sig_loadings(
        factor_loadings_test, loading_cutoff_value=0.3)
    print(factor_words_test[['factor', 'pos_loading_words', 'neg_loading_words']])


# Step 4. 요인모델 학습 - n_factors=5 최종 확정
# - Scree plot elbow 기준 + 해석 가능성 고려
# - n_factors=7 대비 F3/F7 중복 없이 5개 요인이 더 명확하게 구분됨
# - method: uls (공통분산 기반, 잠재 요인 탐색에 적합)
# - rotation: varimax (직교회전, 요인 간 독립성 유지, 해석 용이)

n_factors = 5
method = 'uls'
rotation_method = 'varimax'

factor_loadings, brand_factor_scores = traing_factor_model(
    data_w_meta_cols_efa,
    n_factors=n_factors,
    rotation_method=rotation_method,
    apply_div_by_review_count=apply_div_by_review_count,
    apply_l2=apply_l2,
    apply_stdscaler=apply_stdscaler,
    method=method
)

# Step 5. 유의미한 factor loading 추출
# - loading_cutoff_value=0.3: |loading| >= 0.3인 단어만 추출
factor_loadings_sig, factor_words = extracting_sig_loadings(
    factor_loadings, loading_cutoff_value=0.3)
print("===== 요인별 주요 단어 =====")
print(factor_words[['factor', 'pos_loading_words', 'neg_loading_words']])

# Step 6. 브랜드 포지셔닝 맵 시각화
# - F1(파인다이닝 경험) vs F2(정통 스테이크 메뉴) 축으로 포지셔닝 비교
fig_f1_f2 = drow_factor_map(brand_factor_scores, 'F1', 'F2')
fig_f1_f2.update_layout(width=1200, height=600)
fig_f1_f2.show()

# - F4(카지노 호텔 분위기) vs F5(해피아워·바) 축으로 포지셔닝 비교
fig_f4_f5 = drow_factor_map(brand_factor_scores, 'F4', 'F5')
fig_f4_f5.update_layout(width=1200, height=600)
fig_f4_f5.show()

# Step 7. 특정 브랜드 포지션 설명
# - monamigabi: HW3~HW5에서 NV Steakhouses 대표 파인다이닝으로 확인된 브랜드
target_brand = 'monamigabi'
brand_scores = brand_factor_scores.loc[target_brand]
print(f"\n===== {target_brand} 요인점수 =====")
print(brand_scores[[f'F{i+1}_score' for i in range(n_factors)]])

#=================================
# Part 2. LDA 분석
#=================================
# - EFA와 달리 LDA는 정수 count 데이터 그대로 사용
# - 리뷰수 나누기 미적용, l2 미적용, stdscaler 미적용
# - data_w_meta_cols: 원본 데이터 (EFA용 필터링 미적용)

# Step 8. Grid Search - 최적 LDA 파라미터 탐색
# - 1단계: K만 탐색 (α, β 기본값 고정) → 빠른 K 범위 확인
# - max_iter=5: 속도 우선, 분석 마무리 후 max_iter=10으로 재탐색 예정
# - K 후보: 3, 5, 7, 10 (단어 64개 기준, K 크면 토픽간 단어 중복 발생)
param_grid = {
    "n_components": [3, 5, 7, 10],
    "doc_topic_prior": [None],
    "topic_word_prior": [None],
}
param_search_results_df = lda_parma_grid_search(
    data_w_meta_cols=data_w_meta_cols, param_grid=param_grid
)

# Grid Search 결과 시각화
fig_grid = lda_parma_grid_search_graph(param_search_results_df)
fig_grid.show()

print(param_search_results_df.to_string())