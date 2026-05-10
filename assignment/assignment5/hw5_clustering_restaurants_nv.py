import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
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
# ----------------------------------------
def determine_k(data_w_meta_cols, apply_stdscaler=False, use_cosine=True, k_max=20, batch=True, sample_sil=6000, random_state=42):
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    X = X.values  # numpy array로 먼저 변환
    if apply_stdscaler:
        X = StandardScaler().fit_transform(X)
    if use_cosine:
        X = normalize(X, norm="l2", axis=1)

    Model = MiniBatchKMeans if batch else KMeans

    # Elbow
    inertias = []
    ks = range(1, k_max + 1)
    for k in ks:
        km = Model(n_clusters=k, n_init="auto", random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)

    # Silhouette
    sil_scores = []
    ks_sil = range(2, k_max + 1)
    X_sil = X[np.random.choice(X.shape[0], min(sample_sil, X.shape[0]), replace=False)]
    for k in ks_sil:
        km = Model(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X_sil)
        metric_name = "cosine" if use_cosine else "euclidean"
        sil = silhouette_score(X_sil, labels, metric=metric_name)
        sil_scores.append(sil)

    fig_elbow = go.Figure(go.Scatter(x=list(ks), y=inertias, mode="lines+markers"))
    fig_elbow.update_layout(title="Elbow Method", xaxis_title="k", yaxis_title="Inertia",
                            template="simple_white", width=800, height=400)
    fig_elbow.show()

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
# - 상대차이 기준(top_rel): 다른 군집 대비 해당 군집에서 유독 높은 단어 → 군집 고유의 특성 단어
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
# - 군집 중심을 2차원으로 축소하여 군집 간 거리 시각화
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


#=================================
# CASE 1: 전체 브랜드 + 전체 단어
#=================================
# 분석 목적: 4가지 전처리 조합별 KMeans 결과 비교 → 최적 조합 선정
#
# [4가지 조합만 분석하는 이유]
# 1. l2❌ std✅ / l2✅ std❌ 조합은 분석 의미가 약함
#    - std 먼저 적용 시 리뷰 수 많은 브랜드의 편차가 커져 편향 발생 (HW4 참고)
#    - l2만 적용 시 고빈도 단어 독식 문제 잔존
# 2. 코사인 거리 = l2 정규화 후 유클리드 거리 (수학적으로 동일, 중복 제거)
#
# | 조합 | l2  | std | 거리     |
# |------|-----|-----|----------|
# | 조합 0 | ❌ | ❌  | 유클리드 |
# | 조합 1 | ❌ | ❌  | 코사인   |
# | 조합 2 | ✅ | ✅  | 유클리드 |
# | 조합 3 | ✅ | ✅  | 코사인   | ← 최적 예상

cases = [
    (False, False, "조합 0: l2❌ std❌ 유클리드"),
    (False, True,  "조합 1: l2❌ std❌ 코사인"),
    (True,  False, "조합 2: l2✅ std✅ 유클리드"),
    (True,  True,  "조합 3: l2✅ std✅ 코사인"),
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
# - 각 조합에 대해 k=1~20 범위에서 그래프 확인 후 최적 k 결정
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
# 조합 0 (l2❌ std❌ 유클리드): Elbow/Silhouette 모두 불명확 → k=8 선택 (Elbow 완만해지는 구간)
# 조합 1 (l2❌ std❌ 코사인):   조합 0과 동일한 패턴 → k=7 선택 (Silhouette 첫 번째 지역 피크)
# 조합 2 (l2✅ std✅ 유클리드): Silhouette 매우 불안정(0.07 이하) → k=2 선택 (Silhouette 음수값 등장으로 군집화 품질 낮음, 양수 안정적인 k=2 선택)
# 조합 3 (l2✅ std✅ 코사인):   k=7에서 Silhouette 지역 피크 → k=7 초기 후보 선정

# Step 3-5. 조합 0, 1, 2: 군집 라벨링, 대표 단어, PCA 시각화
# 조합 0: k=8  (Elbow 완만해지는 구간)
# 조합 1: k=7  (Silhouette 첫 번째 지역 피크)
# 조합 2: k=2  (Silhouette 음수값 등장으로 군집화 품질 낮음, 양수 안정적인 k=2 선택)

cases_012 = [
    (False, False, False, 8, "조합 0: l2❌ std❌ 유클리드"),
    (False, True,  False, 7, "조합 1: l2❌ std❌ 코사인"),
    (True,  False, True,  2, "조합 2: l2✅ std✅ 유클리드"),
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


# Step 3-5. 조합 3: k=7 vs k=12 비교 후 최종 k 결정
# - k=7: Silhouette 지역 피크 기준 초기 후보
# - k=12 비교 시 한국·BBQ, 다이너, 그릭·헬시, 카페·델리, 뷔페 등 세부 군집 추가 분리
# - 해석 가능성이 높아져 k=12로 최종 확정

print(f"\n{'='*60}")
print(f"조합 3: l2✅ std✅ 코사인 | k=7 (초기 후보)")
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

# k=7 → k=12 비교: 세부 군집 추가 분리 확인
print(f"\n{'='*60}")
print(f"조합 3: l2✅ std✅ 코사인 | k=12 (최종 확정)")
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

#=================================
# CASE 2: Steakhouses 브랜드 선별 + 전체 단어
#=================================
# 분석 목적: Steakhouses 카테고리 내 브랜드 간 세부 포지셔닝 파악
# 최적 조합(조합 3: l2✅ std✅ 코사인) 적용
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
np.random.seed(42)
value_for_selectingk = determine_k(
    data_w_meta_cols_steak,
    apply_stdscaler=apply_stdscaler,
    use_cosine=use_cosine
)
print(value_for_selectingk)

# k=3, k=9 대표 단어 비교
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

# Step 3-5. Steakhouses k=3 군집 라벨링, 대표 단어, PCA 시각화
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
# 최적 조합(조합 3: l2✅ std✅ 코사인) 적용
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

# 실제 사용 단어 수 확인
word_cols_check = [col for col in data_w_meta_cols_steak_kw.columns if col not in meta_cols_pool]
print(f"실제 사용 단어 수: {len(word_cols_check)}개")

# Step 2. 최적 k 결정 (Elbow + Silhouette)
np.random.seed(42)
value_for_selectingk = determine_k(
    data_w_meta_cols_steak_kw,
    apply_stdscaler=apply_stdscaler,
    use_cosine=use_cosine
)
print(value_for_selectingk)

# k=2, k=4, k=6 대표 단어 비교
# k=2: Silhouette 최고값
# k=4: Elbow 완만해지는 구간 + 해석 가능성 확인
# k=6: Silhouette 두번째 피크

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

# Step 3-5. CASE 3 k=4 군집 라벨링, 대표 단어, PCA 시각화
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

from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff
from sklearn.preprocessing import normalize, StandardScaler

# ----------------------------------------
# hierarchical_clustering_words: 단어 기준 계층적 군집분석
# - DTM을 전치(Transpose)하여 단어를 행으로 변환 후 군집분석 수행
# - 단어 간 코사인 거리 기반 average linkage 방법 적용
# ----------------------------------------
def hierarchical_clustering_words(data_w_meta_cols, k, apply_stdscaler=False, use_cosine=True, method="average", metric="cosine"):
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    # 단어별 군집분석을 위해 행열 전치 (단어 = 행)
    X = X.T

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


# Step 1. CASE 3 데이터 그대로 사용 (Steakhouses + 서비스·분위기 키워드)
# data_w_meta_cols_steak_kw 이미 불러와져 있음

# Step 2. Hierarchical Clustering 수행
# - method: average + cosine 거리 (코사인 거리 기반 평균 linkage)
# - k=5: 서비스·분위기 키워드를 의미 있는 수준으로 군집화
k = 5
apply_stdscaler = False
use_cosine = True
method = "average"
metric = "cosine"

np.random.seed(42)
data_labeled_words, fig_dendro = hierarchical_clustering_words(
    data_w_meta_cols_steak_kw,
    k=k,
    apply_stdscaler=apply_stdscaler,
    use_cosine=use_cosine,
    method=method,
    metric=metric
)

# Step 3. 군집별 키워드 확인
print("===== 군집별 키워드 =====")
print(data_labeled_words['cluster'].value_counts().sort_index())
for cluster_id in sorted(data_labeled_words['cluster'].unique()):
    words = data_labeled_words[data_labeled_words['cluster'] == cluster_id].index.tolist()
    print(f"\nCluster {cluster_id}: {words}")

# Step 4. 덴드로그램 시각화
fig_dendro.show()

# k 값 조정 - 덴드로그램에서 의미 있는 군집 수 탐색
for k_test in [3, 4, 5]:
    print(f"\n===== k={k_test} 군집별 키워드 수 =====")
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.preprocessing import normalize

    df = data_w_meta_cols_steak_kw.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols).T
    X_scaled = normalize(X.values, norm="l2", axis=1)
    linked = linkage(X_scaled, method="average", metric="cosine")
    labels = fcluster(linked, t=k_test, criterion="maxclust")

    result = pd.Series(labels, index=X.index)
    print(result.value_counts().sort_index())
    for cid in sorted(result.unique()):
        print(f"Cluster {cid}: {result[result==cid].index.tolist()}")

