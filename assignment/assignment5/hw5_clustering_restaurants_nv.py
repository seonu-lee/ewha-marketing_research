import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default = "vscode"

import sys
sys.path.append(r"C:\Users\seonu\Documents\ewha-marketing_research")
from lib.lib_dtm import lib_filtering_dtm as lfd

#=================================
# 공통 설정
#=================================
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment5\results"
meta_cols_pool = ['name', 'review_count', 'avg_stars', 'useful_count',
                  'funny_count', 'cool_count', 'categories']


#=================================
# 함수 정의
#=================================

# ----------------------------------------
# determine_k: 최적 군집 수 k 결정
# - Elbow(Inertia): k를 늘릴수록 SSE 감소폭이 완만해지는 지점을 최적 k로 선택
# - Silhouette: 내부 응집도와 외부 분리도를 동시에 고려하여 최적 k 선택
# [실무 기준 3가지]
# 1. Elbow 완만 지점: k를 2배 늘렸을 때 Inertia 감소폭이 20% 이하인 구간
# 2. Silhouette 지역 피크: 최대값보다 첫 번째 지역 피크 또는 급격히 상승 후 안정되는 지점
# 3. 해석 가능성: 군집별 대표 단어가 의미 있게 구분되는지 (가장 중요)
# ----------------------------------------
def determine_k(data_w_meta_cols, apply_stdscaler=False, use_cosine=True, k_max=20, batch=True, sample_sil=6000, random_state=42):
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    X = X.values  # numpy array로 변환 (인덱싱 오류 방지)
    if apply_stdscaler:
        X = StandardScaler().fit_transform(X)
    if use_cosine:
        X = normalize(X, norm="l2", axis=1)  # 코사인 거리 사용 시 행별 L2 정규화

    Model = MiniBatchKMeans if batch else KMeans

    # Elbow: k별 SSE 계산
    inertias = []
    ks = range(1, k_max + 1)
    for k in ks:
        km = Model(n_clusters=k, n_init="auto", random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)

    # Silhouette: k별 실루엣 점수 계산 (데이터가 클 경우 샘플링)
    sil_scores = []
    ks_sil = range(2, k_max + 1)
    X_sil = X[np.random.choice(X.shape[0], min(sample_sil, X.shape[0]), replace=False)]
    for k in ks_sil:
        km = Model(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X_sil)
        metric_name = "cosine" if use_cosine else "euclidean"
        sil = silhouette_score(X_sil, labels, metric=metric_name)
        sil_scores.append(sil)

    # Elbow 그래프
    fig_elbow = go.Figure(go.Scatter(x=list(ks), y=inertias, mode="lines+markers"))
    fig_elbow.update_layout(title="Elbow Method", xaxis_title="k", yaxis_title="Inertia",
                            template="simple_white", width=800, height=400)
    fig_elbow.show()

    # Silhouette 그래프
    fig_sil = go.Figure(go.Scatter(x=list(ks_sil), y=sil_scores, mode="lines+markers"))
    fig_sil.update_layout(title="Silhouette Score", xaxis_title="k", yaxis_title="Score",
                          template="simple_white", width=800, height=400)
    fig_sil.show()

    df_elbow = pd.DataFrame({'k': ks, 'inertia': inertias})
    df_sil = pd.DataFrame({'k': ks_sil, 'sil_score': sil_scores})
    return pd.merge(df_elbow, df_sil, on='k', how='outer')


# ----------------------------------------
# labeling_cluster_and_cal_center: 군집 라벨링 및 중심값 계산
# - 선택한 k로 KMeans 군집 수행
# - 각 브랜드에 군집 번호(cluster) 부여
# - 군집별 중심값(centroid) 계산
# ----------------------------------------
def labeling_cluster_and_cal_center(data_w_meta_cols, k, apply_stdscaler=False, use_cosine=True, batch=True):
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    X_scaled = X.copy()
    if apply_stdscaler:
        X_scaled = StandardScaler().fit_transform(X)
    if use_cosine:
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    Model = MiniBatchKMeans if batch else KMeans
    model = Model(n_clusters=k, n_init="auto", random_state=42, batch_size=1024)
    labels = model.fit_predict(X_scaled)
    centroids = model.cluster_centers_

    data_labeled = data_w_meta_cols.copy()
    data_labeled['cluster'] = labels

    centroids_df = pd.DataFrame(centroids, index=range(k), columns=X.columns)
    centroids_df.index.name = 'cluster'

    return data_labeled, centroids_df


# ----------------------------------------
# top_representative_words_for_clusters: 군집별 대표 단어 추출
# - 절대값 기준(top_abs): 군집 중심값이 가장 높은 단어 → 군집에서 전반적으로 많이 쓰이는 단어
# - 상대차이 기준(top_rel): 다른 군집 대비 해당 군집에서 유독 높은 단어 (z-score 기반)
#                          → 군집 고유의 특성 단어 파악에 적합
# ----------------------------------------
def top_representative_words_for_clusters(centroids_df, top_n):
    top_abs = centroids_df.apply(
        lambda row: row.nlargest(top_n).index.tolist(), axis=1).rename('top_abs')
    mu = centroids_df.mean(axis=0)
    sigma = centroids_df.std(axis=0) + 1e-9
    salience = (centroids_df - mu) / sigma
    top_rel = salience.apply(
        lambda row: row.nlargest(top_n).index.tolist(), axis=1).rename('top_rel')
    return pd.concat([top_abs, top_rel], axis=1)


# ----------------------------------------
# pca_biplot_w_centroids: 군집 중심벡터 PCA 시각화
# - 군집 중심을 2차원으로 축소하여 군집 간 거리 및 방향 시각화
# - 대표 단어 벡터를 biplot으로 함께 표현
# ----------------------------------------
def pca_biplot_w_centroids(centroids_df, cluster_top_words_df, n_top_loading_words_to_display, apply_stdscaler, apply_l2):
    X_scaled = centroids_df.values
    if apply_stdscaler:
        X_scaled = StandardScaler().fit_transform(X_scaled)
    if apply_l2:
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T

    if len(cluster_top_words_df) == 0:
        rep_words = list()
    else:
        rep_words = list(
            set(cluster_top_words_df['top_abs'].explode()) |
            set(cluster_top_words_df['top_rel'].explode()))

    if n_top_loading_words_to_display == 0:
        loading_words = list()
    else:
        abs_loading_sum = np.abs(loadings).sum(axis=1)
        top_idx = np.argsort(abs_loading_sum)[-n_top_loading_words_to_display:]
        loading_words = list(centroids_df.columns[top_idx])

    plot_words = list(set(loading_words + rep_words))
    plot_words_vectors = loadings[[centroids_df.columns.get_loc(w) for w in plot_words]]

    # pca score와 coeff 간 scale 차이 보정
    pca_score_scope = np.percentile(np.abs(scores), 85)
    coeff_scope = np.percentile(np.abs(plot_words_vectors), 90)
    scale_factor = pca_score_scope / coeff_scope

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scores[:, 0], y=scores[:, 1],
        mode='markers+text',
        text=[f"Cluster {i}" for i in centroids_df.index],
        textposition='top center',
        marker=dict(size=12, color='midnightblue'),
        name='Cluster centroid'
    ))
    for word, vec in zip(plot_words, plot_words_vectors):
        fig.add_trace(go.Scatter(
            x=[0, vec[0] * scale_factor], y=[0, vec[1] * scale_factor],
            mode='lines+text', line=dict(color='tomato', width=1),
            text=[None, word], textposition='top center', showlegend=False
        ))
    fig.update_layout(
        title=f"PCA Biplot of {len(centroids_df)} Cluster Centroids "
              f"(explained {pca.explained_variance_ratio_[:2].sum()*100:.1f}%)",
        xaxis_title='PC1', yaxis_title='PC2',
        template='simple_white', width=1000, height=700
    )
    return fig


# ----------------------------------------
# hierarchical_clustering_words: 단어 기준 계층적 군집분석
# - DTM을 전치(Transpose)하여 단어를 행으로 변환 후 군집분석 수행
# - 단어 간 유사도 기반 덴드로그램 시각화
# ----------------------------------------
def hierarchical_clustering_words(data_w_meta_cols, k, apply_stdscaler=False, use_cosine=True, method="average", metric="cosine"):
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    X = X.T  # 단어별 군집분석을 위해 행열 전치 (단어 = 행, 브랜드 = 열)

    X_scaled = X.copy()
    if apply_stdscaler:
        X_scaled = StandardScaler().fit_transform(X)
    if use_cosine:
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    linked = linkage(X_scaled, method=method, metric=metric)
    cluster_labels = fcluster(linked, t=k, criterion="maxclust")

    data_labeled = X.copy()
    data_labeled['cluster'] = cluster_labels

    fig_dendro = ff.create_dendrogram(
        X_scaled,
        labels=X.index.tolist(),
        linkagefun=lambda _: linked
    )
    fig_dendro.update_layout(
        autosize=True,
        width=1400,
        height=600,
        title="Hierarchical Clustering Dendrogram (Keywords)",
        xaxis_title="Keyword",
        yaxis_title="Distance"
    )
    return data_labeled, fig_dendro


#=================================
# CASE 1: 전체 브랜드 + 전체 단어
#=================================
# 분석 목적: 4가지 전처리 조합별 KMeans 결과 비교 → 최적 조합 선정
#
# [4가지 조합만 분석하는 이유]
# 1. l2 미적용 std 적용 / l2 적용 std 미적용 조합은 분석 의미가 약함
#    - std 먼저 적용 시 리뷰 수 많은 브랜드의 편차가 커져 편향 발생 (HW4 참고)
#    - l2만 적용 시 고빈도 단어 독식 문제 잔존
# 2. 코사인 거리 = l2 정규화 후 유클리드 거리 (수학적으로 동일, 중복 제거)
#
# | 조합 | l2  | std | 거리     |
# |------|-----|-----|----------|
# | 조합 0 | 미적용 | 미적용  | 유클리드 |
# | 조합 1 | 미적용 | 미적용  | 코사인   |
# | 조합 2 | 적용 | 적용  | 유클리드 |
# | 조합 3 | 적용 | 적용  | 코사인   | 

cases = [
    (False, False, "조합 0: l2 미적용 std 미적용 유클리드"),
    (False, True,  "조합 1: l2 미적용 std 미적용 코사인"),
    (True,  False, "조합 2: l2 적용 std 적용 유클리드"),
    (True,  True,  "조합 3: l2 적용 std 적용 코사인"),
]

# Step 1. 데이터 불러오기
input_data_filtering_conditions = dict(
    input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2',
    remove_brand_w_word_in_name=False,
    brand_categories_slted=[],
    words_to_delete=[],
    words_to_include_exclusively=[],
)
data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
print(f"전체 브랜드 수: {len(data_w_meta_cols)}개")

# Step 2. 조합별 최적 k 결정 (Elbow + Silhouette)
for apply_stdscaler, use_cosine, desc in cases:
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}")
    value_for_selectingk = determine_k(
        data_w_meta_cols,
        apply_stdscaler=apply_stdscaler,
        use_cosine=use_cosine
    )
    print(value_for_selectingk)

# [k 결정 결과 요약]
# 조합 0 (l2 미적용 std 미적용 유클리드): Elbow/Silhouette 모두 불명확 → k=8 선택 (Elbow 완만해지는 구간)
# 조합 1 (l2 미적용 std 미적용 코사인):   Silhouette 첫 번째 지역 피크 → k=7 선택
# 조합 2 (l2 적용 std 적용 유클리드): Silhouette 음수값 등장, 군집화 품질 최악 → k=2 선택
# 조합 3 (l2 적용 std 적용 코사인):   Silhouette k=7 첫 번째 지역 피크 → k=7 초기 후보
#                              Silhouette k=12~13 두 번째 지역 피크 → k=12와 비교
#                              k=12에서 세부 군집 추가 분리 확인 → k=12 최종 확정

# Step 3-5. 조합 0, 1, 2: 군집 라벨링, 대표 단어, PCA 시각화
cases_012 = [
    (False, False, False, 8, "조합 0: l2 미적용 std 미적용 유클리드"),
    (False, True,  False, 7, "조합 1: l2 미적용 std 미적용 코사인"),
    (True,  False, True,  2, "조합 2: l2 적용 std 적용 유클리드"),
]
# (apply_stdscaler, use_cosine, apply_l2_biplot, k, desc)

for apply_stdscaler, use_cosine, apply_l2_biplot, k, desc in cases_012:
    print(f"\n{'='*60}")
    print(f"{desc} | k={k}")
    print(f"{'='*60}")

    # Step 3. 군집 라벨링 및 중심값 계산
    data_labeled_tmp, centroids_tmp = labeling_cluster_and_cal_center(
        data_w_meta_cols=data_w_meta_cols,
        k=k,
        apply_stdscaler=apply_stdscaler,
        use_cosine=use_cosine,
        batch=True
    )
    print(f"\n[군집별 브랜드 수]")
    print(data_labeled_tmp['cluster'].value_counts().sort_index())

    # Step 4. 군집별 대표 단어
    cluster_top_words_tmp = top_representative_words_for_clusters(centroids_tmp, top_n=5)
    print(f"\n[군집별 대표 단어 - 절대값 기준]")
    print(cluster_top_words_tmp['top_abs'])
    print(f"\n[군집별 대표 단어 - 상대차이 기준]")
    print(cluster_top_words_tmp['top_rel'])

    # Step 5. 군집 중심벡터 PCA 시각화
    fig = pca_biplot_w_centroids(
        centroids_df=centroids_tmp,
        cluster_top_words_df=cluster_top_words_tmp,
        n_top_loading_words_to_display=0,
        apply_stdscaler=True,
        apply_l2=apply_l2_biplot
    )
    fig.show()


# Step 3-5. 조합 3: k=7(초기 후보) vs k=12(두 번째 지역 피크) 비교 후 최종 k 결정
# - Silhouette k=7: 첫 번째 지역 피크 → 초기 후보 선정
# - Silhouette k=12~13: 두 번째 지역 피크 → 추가 비교 수행
# - k=7: 아시아 음식, 서양 다이닝, 멕시칸 등 큰 카테고리로만 분리
# - k=12: 한국·BBQ, 다이너, 그릭·헬시, 카페·델리, 뷔페 등 세부 군집 추가 분리
# - 해석 가능성 기준 세 번째 실무 기준에 의해 k=12로 최종 확정

print(f"\n{'='*60}")
print(f"조합 3: l2 적용 std 적용 코사인 | k=7 (초기 후보 - Silhouette 첫 번째 지역 피크)")
print(f"{'='*60}")

data_labeled_k7, centroids_k7 = labeling_cluster_and_cal_center(
    data_w_meta_cols, k=7,
    apply_stdscaler=True, use_cosine=True)
cluster_top_words_k7 = top_representative_words_for_clusters(centroids_k7, top_n=5)

print(f"\n[군집별 브랜드 수]")
print(data_labeled_k7['cluster'].value_counts().sort_index())
print(f"\n[군집별 대표 단어 - 절대값 기준]")
print(cluster_top_words_k7['top_abs'])
print(f"\n[군집별 대표 단어 - 상대차이 기준]")
print(cluster_top_words_k7['top_rel'])

fig_k7 = pca_biplot_w_centroids(
    centroids_df=centroids_k7,
    cluster_top_words_df=cluster_top_words_k7,
    n_top_loading_words_to_display=0,
    apply_stdscaler=True, apply_l2=True
)
fig_k7.show()

print(f"\n{'='*60}")
print(f"조합 3: l2 적용 std 적용 코사인 | k=12 (최종 확정 - Silhouette 두 번째 지역 피크 + 해석 가능성)")
print(f"{'='*60}")

data_labeled_k12, centroids_k12 = labeling_cluster_and_cal_center(
    data_w_meta_cols, k=12,
    apply_stdscaler=True, use_cosine=True)
cluster_top_words_k12 = top_representative_words_for_clusters(centroids_k12, top_n=5)

print(f"\n[군집별 브랜드 수]")
print(data_labeled_k12['cluster'].value_counts().sort_index())
print(f"\n[군집별 대표 단어 - 절대값 기준]")
print(cluster_top_words_k12['top_abs'])
print(f"\n[군집별 대표 단어 - 상대차이 기준]")
print(cluster_top_words_k12['top_rel'])

fig_k12 = pca_biplot_w_centroids(
    centroids_df=centroids_k12,
    cluster_top_words_df=cluster_top_words_k12,
    n_top_loading_words_to_display=0,
    apply_stdscaler=True, apply_l2=True
)
fig_k12.show()

# [최적 조합 선정 결론]
# 조합 3 (l2 적용 std 적용 코사인), k=12
# - 조합 0: 거대 군집(1,343개) 형성, 고빈도 단어 지배, 변별력 낮음
# - 조합 1: 조합 0보다 균형잡힌 분포, 고빈도 단어 영향 여전히 존재
# - 조합 2: Cluster 0에 3,824개(98%) 집중, Silhouette 음수값 등장, 군집화 실패
# - 조합 3: 가장 균형잡힌 분포, cuisine별 군집 명확, 최적 조합 선정


#=================================
# CASE 2: Steakhouses 브랜드 선별 + 전체 단어
#=================================
# 분석 목적: Steakhouses 카테고리 내 브랜드 간 세부 포지셔닝 파악
# 최적 조합(조합 3: l2 적용 std 적용 코사인) 적용
#
# [Steakhouses 선정 이유]
# - NV vs AZ 카테고리 비율 비교(HW3)에서 AZ 대비 +2.41%p로 NV 특화 카테고리 2위
# - 브랜드 수 204개로 군집분석에 충분한 표본 확보
# - HW3, HW4에서 단어 변별력이 높음을 확인

apply_stdscaler, use_cosine = True, True

# Step 1. 데이터 불러오기
input_data_filtering_conditions = dict(
    input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2',
    remove_brand_w_word_in_name=False,
    brand_categories_slted=['Steakhouses'],
    words_to_delete=[],
    words_to_include_exclusively=[],
)
data_w_meta_cols_steak = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
print(f"Steakhouses 브랜드 수: {len(data_w_meta_cols_steak)}개")

# Step 2. 최적 k 결정 (Elbow + Silhouette)
# - 브랜드 수 204개로 적어 k 결정이 쉽지 않음
# - k=3(Silhouette 첫 피크)과 k=9(두 번째 피크) 대표 단어 비교
np.random.seed(42)
value_for_selectingk = determine_k(
    data_w_meta_cols_steak,
    apply_stdscaler=apply_stdscaler,
    use_cosine=use_cosine
)
print(value_for_selectingk)

# k=3, k=9 대표 단어 비교
# - k=3: 군집 균형 고름, 정통 파인다이닝/캐주얼/카지노 복합 3가지 포지셔닝 명확
# - k=9: 군집당 7~41개로 불균형, 스테이크하우스 특성과 거리 먼 단어 등장 → 과도한 세분화
for k_test in [3, 9]:
    print(f"\n===== k={k_test} 대표 단어 (상대차이 기준) =====")
    data_tmp, centroids_tmp = labeling_cluster_and_cal_center(
        data_w_meta_cols_steak, k=k_test,
        apply_stdscaler=True, use_cosine=True)
    print(f"군집별 브랜드 수:")
    print(data_tmp['cluster'].value_counts().sort_index())
    words_tmp = top_representative_words_for_clusters(centroids_tmp, top_n=5)
    print(f"대표 단어:")
    print(words_tmp['top_rel'])

# Step 3-5. k=3 최종 확정: 군집 라벨링, 대표 단어, PCA 시각화
# - k=3이 군집 균형(70/44/90개)과 해석 가능성 모두 우수
k = 3

data_labeled_steak, centroids_steak = labeling_cluster_and_cal_center(
    data_w_meta_cols_steak, k=k,
    apply_stdscaler=True, use_cosine=True)

print("===== 군집별 브랜드 수 =====")
print(data_labeled_steak['cluster'].value_counts().sort_index())

cluster_top_words_steak = top_representative_words_for_clusters(centroids_steak, top_n=5)
print("\n===== 군집별 대표 단어 (절대값 기준) =====")
print(cluster_top_words_steak['top_abs'])
print("\n===== 군집별 대표 단어 (상대차이 기준) =====")
print(cluster_top_words_steak['top_rel'])

# [CASE 2 군집 해석]
# Cluster 0 (70개): broth, poke, wonton → 아시아 퓨전 계열 스테이크하우스
# Cluster 1 (44개): view, indian, rio   → 카지노 호텔·복합 다이닝
# Cluster 2 (90개): steak, filet, lobster → 정통 파인다이닝 스테이크하우스

fig = pca_biplot_w_centroids(
    centroids_df=centroids_steak,
    cluster_top_words_df=cluster_top_words_steak,
    n_top_loading_words_to_display=0,
    apply_stdscaler=True,
    apply_l2=True
)
fig.show()


#=================================
# CASE 3: Steakhouses 브랜드 선별 + 서비스/경험·장소/분위기 키워드 선별
#=================================
# 분석 목적: Steakhouses 내 서비스·경험·분위기 측면의 세부 포지셔닝 파악
# 최적 조합(조합 3: l2 적용 std 적용 코사인) 적용
#
# [키워드 선별 이유]
# - 스테이크하우스는 음식 품질보다 다이닝 경험·분위기가 차별화 핵심 요인
# - HW3 고유단어 분석에서 monamigabi의 핵심 단어가 view, outsid, watch, reserv 등
#   서비스·분위기 관련 단어에 집중됨을 확인
# - 음식/메뉴 단어는 스테이크하우스 간 공통으로 등장하여 변별력이 낮을 것으로 예상
# - HW2에서 정의한 단어 카테고리 기준 서비스/경험(71개) + 장소/분위기(48개) 적용

apply_stdscaler, use_cosine = True, True

# HW2에서 정의한 서비스/경험 + 장소/분위기 키워드
service_experience = [
    'bartend', 'call', 'card', 'chang', 'charg', 'chef', 'clean', 'cours',
    'deliveri', 'dine', 'dinner', 'drive', 'event', 'famili', 'fast',
    'free', 'fun', 'groupon', 'guy', 'happi', 'hard', 'help', 'hour',
    'husband', 'kid', 'ladi', 'late', 'line', 'live', 'long', 'manag',
    'mani', 'music', 'night', 'noth', 'offer', 'ok', 'old', 'option',
    'outsid', 'owner', 'parti', 'pay', 'person', 'play', 'pm',
    'point', 'pub', 'reserv', 'room', 'sake', 'sampl', 'seat', 'select',
    'server', 'show', 'song', 'sport', 'start', 'station', 'stay', 'store',
    'tabl', 'town', 'truck', 'view', 'waiter', 'waitress', 'watch',
    'water', 'week',
]
location_atmosphere = [
    'airport', 'aria', 'ayc', 'band', 'bar', 'bellagio', 'cafe', 'casino',
    'citi', 'club', 'cool', 'court', 'danc', 'diner', 'door', 'downtown',
    'express', 'game', 'grill', 'hotel', 'hous', 'insid', 'island', 'king',
    'kitchen', 'lake', 'loung', 'mall', 'market', 'mgm', 'mr', 'palm',
    'palac', 'park', 'pool', 'prime', 'rio', 'roberto', 'rock', 'shop',
    'spot', 'steakhous', 'street', 'strip', 'venetian', 'walk', 'wine',
    'wynn',
]
words_to_include_exclusively = sorted(list(set(service_experience + location_atmosphere)))
print(f"선별 키워드 수: {len(words_to_include_exclusively)}개")

# Step 1. 데이터 불러오기
input_data_filtering_conditions = dict(
    input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2',
    remove_brand_w_word_in_name=False,
    brand_categories_slted=['Steakhouses'],
    words_to_delete=[],
    words_to_include_exclusively=words_to_include_exclusively,
)
data_w_meta_cols_steak_kw = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
print(f"브랜드 수: {len(data_w_meta_cols_steak_kw)}개")

word_cols_check = [col for col in data_w_meta_cols_steak_kw.columns if col not in meta_cols_pool]
print(f"실제 사용 단어 수: {len(word_cols_check)}개")

# Step 2. 최적 k 결정 (Elbow + Silhouette)
# - k=2(Silhouette 최고값), k=4(Elbow 완만 구간), k=6(두 번째 피크) 비교
np.random.seed(42)
value_for_selectingk = determine_k(
    data_w_meta_cols_steak_kw,
    apply_stdscaler=apply_stdscaler,
    use_cosine=use_cosine
)
print(value_for_selectingk)

# k=2, k=4, k=6 대표 단어 비교
for k_test in [2, 4, 6]:
    print(f"\n===== k={k_test} 대표 단어 (상대차이 기준) =====")
    data_tmp, centroids_tmp = labeling_cluster_and_cal_center(
        data_w_meta_cols_steak_kw, k=k_test,
        apply_stdscaler=True, use_cosine=True)
    print(f"군집별 브랜드 수:")
    print(data_tmp['cluster'].value_counts().sort_index())
    words_tmp = top_representative_words_for_clusters(centroids_tmp, top_n=5)
    print(f"대표 단어:")
    print(words_tmp['top_rel'])

# Step 3-5. k=4 최종 확정: 군집 라벨링, 대표 단어, PCA 시각화
# - k=2: 너무 큰 덩어리로만 분리, 세부 포지셔닝 파악 어려움
# - k=4: 군집 균형(44~61개), 카지노 호텔 계열별 포지셔닝 명확 → 선택
# - k=6: 군집당 25~44개로 불균형, 위치 특성에 치우친 단어 등장
k = 4

data_labeled_steak_kw, centroids_steak_kw = labeling_cluster_and_cal_center(
    data_w_meta_cols_steak_kw, k=k,
    apply_stdscaler=True, use_cosine=True)

print("===== 군집별 브랜드 수 =====")
print(data_labeled_steak_kw['cluster'].value_counts().sort_index())

cluster_top_words_steak_kw = top_representative_words_for_clusters(centroids_steak_kw, top_n=5)
print("\n===== 군집별 대표 단어 (절대값 기준) =====")
print(cluster_top_words_steak_kw['top_abs'])
print("\n===== 군집별 대표 단어 (상대차이 기준) =====")
print(cluster_top_words_steak_kw['top_rel'])

# [CASE 3 군집 해석]
# Cluster 0 (61개): pub, airport, club, danc → 캐주얼·엔터테인먼트 스테이크하우스
# Cluster 1 (47개): palm, ayc, mgm          → MGM·팜 계열 복합 다이닝
# Cluster 2 (52개): wine, venetian, wynn    → 베네치안·윈 호텔 파인다이닝
# Cluster 3 (44개): prime, roberto, rio     → 리오·아일랜드 계열 고급 스테이크

fig = pca_biplot_w_centroids(
    centroids_df=centroids_steak_kw,
    cluster_top_words_df=cluster_top_words_steak_kw,
    n_top_loading_words_to_display=0,
    apply_stdscaler=True,
    apply_l2=True
)
fig.show()


#=================================
# CASE 4: 키워드 Hierarchical Clustering
#=================================
# 분석 목적: CASE 3에서 사용한 서비스·분위기 키워드 간 유사도 관계 파악
# - 브랜드가 아닌 단어를 군집화하여 키워드 간 의미적 연관성 확인
# - 덴드로그램으로 키워드 계층 구조 시각화
#
# [CASE 1 최적 조합 대신 ward+euclidean을 사용한 이유]
# - CASE 4는 키워드 군집화로 CASE 1~3의 브랜드 군집화와 분석 구조가 근본적으로 다름
# - CASE 1 최적 조합 선정 기준(브랜드 리뷰 수 편향, 고빈도 단어 독식 제거)은
#   단어가 행(row)인 CASE 4에 적용되지 않음
# - average+cosine 시도 결과: ayc, truck, airport, lake 등 특정 브랜드에만
#   집중 등장하는 고유명사가 단독 군집을 형성하는 문제 발생
#   (average linkage는 이상치에 매우 민감하기 때문)
# - 이상치 단어 제거(옵션 1) 시도 후에도 aria, venetian 등 추가 이상치 발생
# - ward+euclidean(옵션 2): 군집 내 분산 최소화 방식으로 이상치에 덜 민감
#   → 군집 분포(6~33개) 균형, 해석 가능한 군집 형성 확인 → 최종 선택

# Step 1. CASE 3 데이터 그대로 사용 (Steakhouses + 서비스·분위기 키워드)

# Step 2-1. 초기 시도: average + cosine
# - 결과: 114개가 하나의 군집, 나머지 4개(truck, airport, ayc, lake) 단독 군집 형성
# - 이상치에 민감한 average linkage의 한계 확인

k = 5
np.random.seed(42)
data_labeled_words, fig_dendro = hierarchical_clustering_words(
    data_w_meta_cols_steak_kw,
    k=k, apply_stdscaler=False, use_cosine=True,
    method="average", metric="cosine"
)
print("===== [초기 시도] average+cosine 군집별 키워드 수 =====")
print(data_labeled_words['cluster'].value_counts().sort_index())
fig_dendro.show()

# Step 2-2. 옵션 1: 이상치 단어 제거 후 average+cosine 재시도
# - truck, airport, ayc, lake 제거 후에도 aria, venetian 등 추가 이상치 발생
# - 근본적인 해결책이 아님을 확인

words_to_delete = ['truck', 'airport', 'ayc', 'lake']
input_data_filtering_conditions_v2 = dict(
    input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2',
    remove_brand_w_word_in_name=False,
    brand_categories_slted=['Steakhouses'],
    words_to_delete=words_to_delete,
    words_to_include_exclusively=words_to_include_exclusively,
)
data_w_meta_cols_steak_kw_v2 = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions_v2)
word_cols_v2 = [col for col in data_w_meta_cols_steak_kw_v2.columns if col not in meta_cols_pool]
print(f"\n[옵션 1] 이상치 제거 후 단어 수: {len(word_cols_v2)}개")

for k_test in [3, 4, 5]:
    print(f"\n===== [옵션 1] k={k_test} 군집별 키워드 수 =====")
    df = data_w_meta_cols_steak_kw_v2.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols).T
    X_scaled = normalize(X.values, norm="l2", axis=1)
    linked = linkage(X_scaled, method="average", metric="cosine")
    labels = fcluster(linked, t=k_test, criterion="maxclust")
    result = pd.Series(labels, index=X.index)
    print(result.value_counts().sort_index())

# Step 2-3. 옵션 2: ward + euclidean (최종 선택)
# - 이상치에 덜 민감, 균형잡힌 군집 분포 확인
print("\n[옵션 2] ward + euclidean 방법으로 변경")
for k_test in [3, 4, 5]:
    print(f"\n===== [옵션 2] k={k_test} 군집별 키워드 수 =====")
    df = data_w_meta_cols_steak_kw.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols).T
    X_scaled = normalize(X.values, norm="l2", axis=1)
    linked = linkage(X_scaled, method="ward", metric="euclidean")
    labels = fcluster(linked, t=k_test, criterion="maxclust")
    result = pd.Series(labels, index=X.index)
    print(result.value_counts().sort_index())
    for cid in sorted(result.unique()):
        print(f"Cluster {cid}: {result[result==cid].index.tolist()}")

# Step 3-4. k=5, ward+euclidean 최종 확정: 군집별 키워드 확인 + 덴드로그램
# - k=5에서 군집 분포(6~33개) 균형, 서비스·장소 계열 명확하게 분리
data_labeled_words, fig_dendro = hierarchical_clustering_words(
    data_w_meta_cols_steak_kw,
    k=5, apply_stdscaler=False, use_cosine=True,
    method="ward", metric="euclidean"
)
print("\n===== [최종] ward+euclidean k=5 군집별 키워드 =====")
print(data_labeled_words['cluster'].value_counts().sort_index())
for cluster_id in sorted(data_labeled_words['cluster'].unique()):
    words = data_labeled_words[data_labeled_words['cluster'] == cluster_id].index.tolist()
    print(f"\nCluster {cluster_id}: {words}")

# [CASE 4 군집 해석]
# Cluster 1 (22개): cafe, casino, hotel, prime, steakhous, strip → 장소·시설 유형
# Cluster 2 (33개): dinner, reserv, server, tabl, waiter, wine  → 격식 풀서비스 다이닝
# Cluster 3 (31개): bar, clean, fast, kid, music, waitress      → 캐주얼 서비스·분위기
# Cluster 4 ( 6개): club, danc, loung, rio, song, view          → 엔터테인먼트·나이트라이프
# Cluster 5 (26개): aria, bellagio, mgm, palm, venetian         → 대형 카지노 호텔 계열
#
# [덴드로그램 해석]
# - 전체 구조: 서비스·경험 계열(Cluster 1~3) vs 카지노·엔터 계열(Cluster 4~5) 2대 축으로 분리
# - Cluster 2(격식 다이닝)와 Cluster 3(캐주얼 서비스)이 Distance ~1.9에서 병합
#   → 서비스 경험이라는 공통점, 격식 수준만 다름
# - Cluster 4(엔터)와 Cluster 5(카지노 호텔)이 Distance ~2.0에서 병합
#   → 라스베이거스 카지노 엔터테인먼트 공통 맥락
# - 두 대군집이 Distance ~3.3에서 최종 병합
#   → 서비스·경험 vs 카지노·엔터가 가장 이질적인 두 축

fig_dendro.show()