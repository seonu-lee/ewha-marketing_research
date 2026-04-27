'''
1. 군집분석 
1.1 개념: 
    - 레이블이 없는 데이터를 유사성에 기반하여 그룹(cluster)화하는 기법
    - 같은 cluster 내 데이터는 유사하고, 다른 cluster 간 데이터는 상이함

1.2 목적/용도
1.2.1 시장 세분화 - 유사 브랜드끼리 grouping
1.2.2 고객 세분화 - 주어진 브랜드에 대해 리뷰 패턴별로 고객 grouping
1.2.3 차원축소 시각화 지원 - pca 시각화 + clustering

2. 알고리즘
2.1 KMeans
2.2.2 원리 
    - 각 점과 소속 군집 중심간의 거리제곱합 최소화: a) k개의 중심 초기화 -> b) 각 점을 가장 가까운 중심에 할당 -> c) 군집별 할당된 점들의 평균을 새 중심을 조정 -> 모든 군집에 대해여 거리제곱합이 더 이상 줄어들지 않을때까지 b-c 반복
    - 주어진 a)에 대해 b-c를 최대 max_iter 만큼 반복, 중심 초기화 즉, a-b-c전체과정을 n_init 만큼 반복하여, 거리제곱합이 최소가 되는 결과를 선택함
2.2.3 특징
    - 빠르고 효율적, 대규모 데이터에 적합
    - 클러스터 수 k 사전 정의 필요
2.2.4 거리 계산 방식
    - 유클리드 거리: euclid_distance(a,b), ∥a-b∥ <-- ∥a-b∥² = ∥a∥² + ∥b∥² - 2∥a∥∥b∥cosθ
    - 두 벡터의 길이, 방향 모두 반영
    
    - 코사인 거리: 두 벡터가 방향(패턴)을 얼마나 달리하는지 측정, cosine_distance(a,b)=1-cosine_similarity(a,b)=1-cosθ
    - 두 벡터의 길이 영향 제거
    - 두 단위 벡터간의 유클리드 거리는 코사인 거리와 비례, euclid_distance(a,b) = √(2*(1-cosθ))
    - 즉, 데이터를 l2 정규화후 유클리드 거리를 적용하면 코사인 거리를 적용하는 것이 됨

2.2 Hierarchical
2.2.1 원리
    - a) 최초에는 각 점을 개별 군집으로 처리(n개) -> b) 모든 군집 쌍의 거리 계산(nxn 거리 행렬) -> c) 가장 가까운 두 군집 병합. 
    - b)-c)를 n-1번 반복하면 모든 점이 하나의 군집으로 통합. 계산과정이 덴드로그램
    - 군집간 거리계산 방식 (linkage): 
        - single: 최단거리
        - complete: 최장거리
        - average: 평균거리
        - Ward: 병합 후의 군집 내 오차 제곱합(SSE) 증가량이 최소가 되는 두 군집을 병합
2.2.2 특징
    - 트리 형태로 생성되 덴드로그램 시각화 가능, 클러스터 수 미정 상태에서 시작 가능. 
    - 계산량 많고 대용량에 부적합

3. 군집 수 결정
3.1 Inertia(Elbow)
    - k를 1씩 늘릴때 오차(SSE, 샘플과 소속 군집의 중심간의 거리제곱합, 이 값이 작을수록 군집 내부가 조밀하다는 의미임)가 얼마나 줄어드는지 계산
    - 감소 폭이 완만해지는 지점(elbow), 즉 군집 수를 늘려도 오차가 크게 감소하지 않는 지점을 적정 k로 선택

3.2 Silhouette
    - 같은 군집 안은 조밀하고, 다른 군집과는 멀어진 정도를 계산
    - 내부응집도: 같은 군집에 있는 점들과의 평균 거리, 값이 낮을수록 응집도가 높음
    - 외부분리도: 다른 군집 가운데 가장 가까운 군집까지의 평균 거리, 값이 높을수록 분리도가 높음

    - 데이터 i의 실루엣 계수 = (bi-ai)/max(bi, ai), 
    - 여기서 ai: 데이터 i가 속한 자기 군집 안의 다른 점들과의 평균 거리, bi: 데이터 i가 가장 가까운 다른 군집의 점들과의 평균 거리

    - 각 점에 대해 실루엣 계수를 계산한 다음, 모든 점의 실루엣 계수 평균값 기준으로 판별
    - 내부 응집‧외부 분리 모두 고려. 계산량이 많음.


'''

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import pairwise_distances
from plotly.subplots import make_subplots
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA

import sys
from importlib import reload
sys.path.append(r"C:\Users\seonu\Documents\ewha-marketing_research")
from lib.lib_dtm import lib_filtering_dtm as lfd
reload(lfd)


### 공통 조건 설정
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\session4_dtm\results"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\session7_clustering\results"

meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 

# ────────────────────────────────────────────────────────────
# 1) 조건에 맞는 데이터 추출

# ────────────────────────────────────────────────────────────
# 2) 군집 수(k) 결정 함수
def determine_k(data_w_meta_cols, apply_stdscaler=False, use_cosine=True, k_max=20, batch=True, sample_sil=6000, random_state=42):

    '''
    Parameters
    ----------
    data_w_meta_cols: dtm 데이터
    k_max: 최대 군집수 k
    apply_stdscaler: 열(단어)별 표준화, (value-mean)/stderror
    use_cosine: 코사인 거리(행 L2 정규화) 사용 여부

    batch: MiniBatchKMeans 사용 여부
    sample_sil: 실루엣 계산에 사용할 샘플 수
    random_state: 재현성

    Returns
    -------
    elbow,  Silhouette 값 dataframe
    '''

    #=================================
    # 데이터 전처리
    #=================================
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    ### 데이터 표준화
    if apply_stdscaler == True:
        X = StandardScaler().fit_transform(X) # 각 단어(열) 별로 표준화 (value-mean)/stderror

    # 코사인 거리 사용시 데이터 단위길이로 정규화
    if use_cosine: # 코사인 거리 사용 시 각 행을 길이 1로 L2 정규화
        X = normalize(X, norm="l2", axis=1)

    #=================================
    # 모델 선택
    #=================================
    # MiniBatchKMeans(빠름) vs. KMeans(정확하지만 느림)
    Model = KMeans
    if batch:
        Model = MiniBatchKMeans 

    #=================================
    # Inertia(Elbow) 계산
    #=================================
    inertias = [] # SSE(샘플과 소속 군집의 중심간의 거리제곱합) 담을 리스트
    ks = range(1, k_max + 1)
    for k in ks: # 각 k(군집 수)에 대해, 
        km = Model(n_clusters=k, n_init="auto", random_state=random_state) # 군집 모델 객체 생성
        km.fit(X) # 군집 모델 학습
        inertias.append(km.inertia_) # SSE 계산하여 리스트로 저장

    #=================================
    # Silhouette 계산
    #=================================
    sil_scores = []
    ks_sil = range(2, k_max + 1)

    # Silhouette 계산할 데이터 샘플링 (데이터가 커지면 매우 느려짐)
    if X.shape[0] > sample_sil:
        X_sil = X[np.random.choice(X.shape[0], sample_sil, replace=False)] # X=행(브랜드)*열(워드), 전체 행 갯수(N)일때 0 ~ N-1 에서 무작위 추출후, 이 번호들로 행 추출
    else:
        X_sil = X

    # k 값을 바꿔가며 실루엣 점수 계산
    for k in ks_sil:
        km = Model(n_clusters=k, n_init="auto", random_state=random_state) # 군집 모델 객체 생성
        labels = km.fit_predict(X_sil) # 군집 모델 학습
        metric_name = "cosine" if use_cosine else "euclidean" # silhouette_score에 적용한 metric 설정
        sil = silhouette_score(X_sil, labels, metric=metric_name) # 실루엣 점수 계산
        sil_scores.append(sil)

    #=================================
    # Plot
    #=================================
    def plot_metric(x, y, title, x_title, y_title):
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, template="simple_white")
        fig.show()

    plot_metric(list(ks), inertias, "Elbow Method", "k", "Inertia")
    plot_metric(list(ks_sil), sil_scores, "Silhouette Score", "k", "Score")

    # 반환용 dataframe
    df_elbow = pd.DataFrame({'k': ks, 'inertia': inertias})    
    df_sil = pd.DataFrame({'k': ks_sil, 'sil_score': sil_scores})

    return pd.merge(df_elbow, df_sil, on='k', how='outer')

# ────────────────────────────────────────────────────────────
# 3) 선택된 데이터와 k --> 군집분석 후 군집 labeling, centriods 
def labeling_cluster_and_cal_center(data_w_meta_cols, k, apply_stdscaler=False, use_cosine=True, batch=True):

    """
    Parameters
    ----------
    data_w_meta_cols: dtm 데이터    
    k: 클러스터 수
    apply_stdscaler: 열(단어)별 표준화, (value-mean)/stderror
    use_cosine: 코사인 거리 사용 여부
    batch: MiniBatchKMeans 사용 여부

    Returns
    -------
    data_labeled: 인풋 data에 군집 라벨 추가된 dataframe
    centroids_df: 군집별 중심값 dataframe 
    """

    #=================================
    # 데이터 전처리
    #=================================
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    ### 데이터 표준화
    X_scaled = X.copy() # 표준화 적용하지 않을 경우 
    if apply_stdscaler == True:
        X_scaled = StandardScaler().fit_transform(X) # 각 단어(열) 별로 표준화 (value-mean)/stderror

    # 코사인 거리 사용시 데이터 단위길이로 정규화
    if use_cosine: # 코사인 거리 사용 시 각 행을 길이 1로 L2 정규화
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    #=================================
    # 모델 선택 및 학습
    #=================================
    Model = MiniBatchKMeans if batch else KMeans
    model = Model(
        n_clusters=k, 
        n_init="auto", # 반복할 초기 seed 수
        random_state=42, 
        batch_size=1024
    )

    labels = model.fit_predict(X_scaled) # 각 브랜드 라벨
    centroids = model.cluster_centers_ # 군집별 중심값 계산 (k, n_terms)

    #=================================
    # 라벨 부착 
    #=================================
    data_labeled = data_w_meta_cols.copy()
    data_labeled['cluster'] = labels

    #=================================
    # 중심값을 DataFrame으로 변환 
    #=================================
    centroids_df = pd.DataFrame(
        centroids,
        index = range(k), # 0 … k-1  (cluster id)
        columns = X.columns # 단어 컬럼 그대로
    )
    centroids_df.index.name = 'cluster' # index 이름을 cluster로 변경

    return data_labeled, centroids_df

# ────────────────────────────────────────────────────────────
# 4) 군집 중심벡터로부터 각 군집별 대표 단어 추출 - 절대값 기준, 상대차이 기준
def top_representative_words_for_clusters(centroids_df, top_n):

    """
    Parameters
    ----------
    centroids_df: 군집별 중심값 dataframe 
    top_n: 군집별 추출할 대표 단어 수

    Returns
    -------
    cluster_top_words: 군집별 대표 단어 dataframe
    """

    #=================================
    # (A) 절대값 기준
    #=================================
    # top_n개의 가장 큰값을 가지는 단어 추출
    # 모든 군집에서 공통으로 높은 단어가 등장할 수 있음

    top_abs = centroids_df.apply(lambda row: row.nlargest(top_n).index.tolist(), axis=1).rename('top_abs') # nlargest(n): pandas series에 대해 값이 가장 큰 n개 항목 추출 

    #=================================
    # (B) 상대차이 기준
    #=================================
    # 다른 군집들에 비해 해당 군집에서 높게 나타나는 단어 추출

    mu = centroids_df.mean(axis=0) # 단어별 평균값
    sigma = centroids_df.std(axis=0) + 1e-9  # 단어별 표준편차, 0으로 나누기 방지
    salience = (centroids_df - mu) / sigma # 단어별 z-score, 주어진 단어의 z-score가 높을수록, 평균에 비해 그 군집에서 값이 높음을 의미

    top_rel = salience.apply(lambda row: row.nlargest(top_n).index.tolist(), axis=1).rename('top_rel')

    #=================================
    # 절대값 + 상대값 대표 단어 합치기
    #=================================
    cluster_top_words = pd.concat([top_abs, top_rel], axis=1) # 옆으로 합치기

    return cluster_top_words
    
# ────────────────────────────────────────────────────────────
# 5) 군집 중심벡터 pca 분석 (단어 공간에서 군집간 거리 시각화)
def pca_biplot_w_centroids (centroids_df, cluster_top_words_df, n_top_loading_words_to_display, apply_stdscaler, apply_l2):

    """
    Parameters
    ----------
    centroids_df: pca 분석할 군집별 중심값 dataframe 
    cluster_top_words_df: 그래프에 포함할 군집별 대표 단어 dataframe
    n_top_loading_words_to_display: 
    apply_stdscaler: 열별 표준화 여부
    apply_l2: 행별 정규화

    Returns
    -------
    fig: pca 시각화 그래프
    """

    #=================================
    # 데이터 표준화
    #=================================
    X_scaled = centroids_df.values
    if apply_stdscaler == True:
        X_scaled = StandardScaler().fit_transform(X_scaled) # 단어 스케일 차이 제거, 각 단어(열) 별로 표준화 (value-mean)/stderror
    if apply_l2 == True: # 행별 l2 normalize
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    #=================================
    # PCA (2차원으로 고정)
    #=================================
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(X_scaled) # pc 계산 - 원 데이터를 PC 공간으로 투영, 군집 점 좌표
    loadings = pca.components_.T # (411, 2) -> # loadings - 각 주성분을 정의하는 원래 변수들의 가중치, np.transpose(pca.components_)와 동일

    #=================================
    # 시각화용 단어 선정 - 절대값 기준, 상대차이 기준 top word 각각에 대해서 해보고 비교할 것
    #=================================
    # 1) 대표 단어 집합 
    if len(cluster_top_words_df) == 0:
        rep_words = list()
    else:    
        rep_words = list(set(cluster_top_words_df['top_abs'].explode()) | set(cluster_top_words_df['top_rel'].explode())) # 절대값 기준 + 상대차이 기준 top words
        # rep_words = list(cluster_top_words_df['top_rel'].explode()) # 상대차이 기준 top words
        # rep_words = list(cluster_top_words_df['top_abs'].explode()) # 절대값 기준 top words

    # 2) 기존 loading 상위 단어 추출
    if n_top_loading_words_to_display == 0:
        loading_words = list()
    else:
        abs_loading_sum = np.abs(loadings).sum(axis=1) # 각 loading에 대해 절대값의 합
        # abs_loading_sum = np.linalg.norm(loadings, axis=1) # 각 loading에 대해 제곱합의 제곱근 사용할 경우

        top_idx = np.argsort(abs_loading_sum)[-n_top_loading_words_to_display:] # np.argsort 오름차순 정렬했을 때의 인덱스 반환, 뒤에서 n_top_loading_words_to_add 수만큼 추출
        loading_words = list(centroids_df.columns[top_idx]) # word 리스트에서 top_idx를 적용하여 단어 추출

    # 3) 최종 plotting 단어 (두 집합 union)
    plot_words = list(set(loading_words + rep_words)) 
    plot_words_vectors = loadings[[centroids_df.columns.get_loc(w) for w in plot_words]] # get_loc(w): w 컬럼명이 몇 번째 위치(index)에 있는지 반환

    #=================================
    # Plotly Biplot
    #=================================
    
    ### pca score, coeff 값 scale 차이 보정 계수 계산
    pca_score_scope = np.percentile(np.abs(scores), 85)
    coeff_scope = np.percentile(np.abs(plot_words_vectors), 90)
    scale_factor = pca_score_scope / coeff_scope # scale 자동설정
    
    ### 그래프
    fig = go.Figure()

    # 군집 중심 점 (scatter)
    fig.add_trace(go.Scatter(
        x=scores[:, 0], y=scores[:, 1],
        mode='markers+text',
        text=[f"Cluster {i}" for i in centroids_df.index],
        textposition='top center',
        marker=dict(size=12, color='midnightblue'),
        name='Cluster centroid'
    ))

    # 단어 화살표    
    for word, vec in zip(plot_words, plot_words_vectors):
        fig.add_trace(go.Scatter(
            x=[0, vec[0] * scale_factor],
            y=[0, vec[1] * scale_factor],
            mode='lines+text',
            line=dict(color='tomato', width=1),
            text=[None, word], # 선의 끝부분에 단어 표시. 시작부분(즉 원점)에는 아무것도 표시하지 않음(None)
            textposition='top center',
            showlegend=False
        ))

    # 레이아웃
    fig.update_layout(
        # width=900, height=700,
        title=f"PCA Biplot of {len(centroids_df)} Cluster Centroids "
            f"(explained {pca.explained_variance_ratio_[:2].sum()*100:.1f} %)",
        xaxis_title='PC1',
        yaxis_title='PC2',
        template='simple_white'
    )
    return fig

'''
# ────────────────────────────────────────────────────────────
### 6) 각 브랜드의 해당 군집 중심으로부터의 거리 값 추가
def add_center_distance(data_labeled, use_cosine=True):

    """
    Parameters
    ----------
    data_labeled: dtm 데이터 + cluster 컬럼
    metric : 거리 지표 ('cosine' 또는 'euclidean')

    Returns
    -------
    data_with_dist : 원본 + center_dist 컬럼(해당 군집 평균벡터와의 거리)
    """

    #----------------------------
    # 1) 데이터 전처리
    #----------------------------
    
    ### 메타 컬럼 제거
    meta_cols = [col for col in data_labeled.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    data_wo_meta = data_labeled.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    ### tfidf 컬럼만 선택 
    tfidf_cols = data_wo_meta.drop(columns="cluster").columns # words 리스트

    X = data_labeled[tfidf_cols].to_numpy() # 단어 벡터
    labels = data_labeled["cluster"].to_numpy() # 각 행의 클러스터 id 배열

    #----------------------------
    # 2) 군집별 평균 벡터(군집 중심) 계산 
    #----------------------------

    ### 클러스터별로 단어 벡터의 평균을 계산 (군집 중심 좌표)
    centroids = data_labeled.groupby('cluster')[tfidf_cols].mean().to_numpy()

    ### 코사인 거리 사용 시 추가 정규화 
    if use_cosine == True:  
        # 군집분석에 코사인거리를 사용했다면, 행별 L2 정규화를 했다는 의미. 거리계산을 위해서도 동일하게 적용함
        # 코사인 거리는 벡터의 방향만 보므로, 행 길이를 1로 맞춤 (L2 정규화)
        X = normalize(X, axis=1)
        centroids = normalize(centroids, axis=1)

    #----------------------------
    # 3) 각 행의 중심-거리 계산 
    #----------------------------
    ### 군집생성에 사용한 거리계산 방식에 따라, 각 행각과 군집중심간의 거리 계산에도 적용하기 위한 설정
    if use_cosine == True:
        metric = "cosine"
    else:
         metric = "euclidean"

    ### 각 행과 군심 중심간의 거리 계산
    d_all = np.empty(len(X))
    for cid, center_vec in enumerate(centroids): # 각 군집 중심벡터 반복
        mask = (labels == cid) # 현재 군집에 속하는 행 선택
        X_sub = X[mask] # 해당 군집 소속 데이터만 추출
        
        # 각 행과 군집 중심 벡터 사이의 거리 계산 (cosine 또는 euclidean)
        dists = pairwise_distances(
            X_sub, center_vec.reshape(1, -1), metric=metric
        ).flatten() # 결과를 1차원으로 평탄화
        d_all[mask] = dists # 해당 군집 소속 행의 위치에 거리 값 채워 넣기

    #----------------------------
    # 4) 결과 컬럼 추가 후 반환 
    #----------------------------
    data_labeled_dist = data_labeled.copy()
    data_labeled_dist["center_dist"] = d_all
    
    data_labeled_dist = data_labeled_dist.sort_values(by=['cluster', 'center_dist'], ascending=[True, True]) # 군집별, 거리가까운 순으로 정렬

    return data_labeled_dist

# ────────────────────────────────────────────────────────────
### 7) 군집별 대표 브랜드 추출
def get_cluster_top_rep_brands(data_labeled_dist, n_brands):
    # 1) 거리 기준(작을수록 중심에 가까움)으로 정렬
    df_sorted = data_labeled_dist.sort_values(by=["cluster", "center_dist"], ascending=[True, True])

    # 2) 군집별 상위 5개 브랜드 이름만 리스트로 추출
    rep_series = (
        df_sorted
        .groupby("cluster")["name"]
        .apply(lambda s: s.head(n_brands).tolist())
    )

    # 3) 시리즈 → 데이터프레임(행: cluster, 열: 'top5_brands')
    cluster_rep_brands_df = rep_series.to_frame(name="rep_brands").reset_index()

    return cluster_rep_brands_df
'''


if __name__ == "__main__":

    #=====================================
    # 전체 브랜드 + 전체 키워드
    #=====================================

    '''
    0) 행별 l2 normalize 미적용 -> 열별 stdscaler 미적용 -> 유클리드거리:     
    --> 군집수결정 elbow와 실루엣 그래프 모두 불안정, 결과확인을 위해 7,11; pizza, chicken, sandwich, bar은 일반적인 단어들이 거대한 군집을 이룸 (변별력이 거의 없어 보임)

    1) 행별 l2 normalize 미적용 -> 열별 stdscaler 미적용 -> 코사인거리 
      (행별 l2 normalize 적용 -> 열별 stdscaler 미적용 -> 유클리드거리 와 동일조건임):   
    --> 군집수결정 elbow기준 13; 위의 유클리드 거리 적용한 경우보다 합리적인 군집화 양상, 그러나 chicken 같은 고빈도 단어가 중복등장. 즉 열별(단어별) 빈도차이의 영향을 받는 것으로 보임
    
    2) 행별 l2 normalize 적용 -> 열별 stdscaler 적용 -> 유클리드거리: 
    --> 군집수결정 elbow기준 8, 16; 8는 다소 덜 분화된 결과를 보여주고, 16의 군집별 주요단어 구분이 비교적 명확함

    3) 행별 l2 normalize 적용 -> 열별 stdscaler 적용 -> 코사인거리(l2 normalize후 유클리드거리와 동일): 
    --> elbow, 실루엣그래프 모두 12가 최적으로 보임; 군집별 주요단어 구분이 비교적 명확함 
    
    (상세결과해석은 ai 활용해볼것)
    '''

    ## 0) 분석조건
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = [],
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    apply_stdscaler, use_cosine = False, False
    k = 13

    ## 1) 분석할 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

    ## 2) 군집수(k) 결정
    value_for_selectingk = determine_k(
        data_w_meta_cols=data_w_meta_cols, 
        apply_stdscaler=apply_stdscaler, use_cosine=use_cosine
        )

    ## 3) 군집 라벨링, 군집별 중심값
    data_labeled, centroids_df = labeling_cluster_and_cal_center(
        data_w_meta_cols=data_w_meta_cols, k=k, 
        apply_stdscaler=apply_stdscaler, use_cosine=use_cosine, batch=True
        )
    data_labeled['cluster'].value_counts()

    ## 4) 군집별 주요 단어
    cluster_top_words_df = top_representative_words_for_clusters(centroids_df, top_n=5)
    cluster_top_words_df

    ## 5) 군집 중심벡터 pca 분석 (단어 공간에서 군집간 거리 시각화)
    fig = pca_biplot_w_centroids (
        centroids_df, 
        cluster_top_words_df, # biplot에 표시할 군집별 주요 단어
        n_top_loading_words_to_display=0, # biplot에 표시할 loading 기준 상위 단어 갯수
        apply_stdscaler=True, apply_l2=True
        )
    fig.show()

    
    ## 6) 군집별 브랜드 “중심-거리" 값 추가
    # **NOTE** stdscaler 반영하지 안음. apply_stdscaler=False 인 조건에서만 사용할것
    # data_labeled_dist = add_center_distance(data_labeled, use_cosine)

    ## 7) 군집별 대표 브랜드 추출
    # rep_brands = get_cluster_top_rep_brands(data_labeled_dist, n_brands=5) 
    # rep_brands.loc[0, 'rep_brands']
    
    ## 8) 각 군집 profiling
    # avg star ratings, business categories, avg review count 등
    # data_labeled['cluster'].value_counts().sort_index() # 군집별 브랜드 수


    #=====================================
    # 분석대상 브랜드 선별 (예, 카테고리 기준 필터링) 후 분석
    #=====================================
    ## 0) 분석조건
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = ['Italian'],
        words_to_delete = [],
        words_to_include_exclusively = [],
        )

    apply_stdscaler, use_cosine = True, True
    k = 7

    ## 1) 분석할 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

    ## 2) 군집수(k) 결정
    value_for_selectingk = determine_k(
        data_w_meta_cols=data_w_meta_cols, 
        apply_stdscaler=apply_stdscaler, use_cosine=use_cosine
        )

    ## 3) 군집 라벨링, 군집별 중심값
    data_labeled, centroids_df = labeling_cluster_and_cal_center(
        data_w_meta_cols=data_w_meta_cols, k=k, 
        apply_stdscaler=apply_stdscaler, use_cosine=use_cosine, batch=True
        )
    data_labeled['cluster'].value_counts()

    ## 4) 군집별 주요 단어
    cluster_top_words_df = top_representative_words_for_clusters(centroids_df, top_n=5)
    cluster_top_words_df

    ## 5) 군집 중심벡터 pca 분석 (단어 공간에서 군집간 거리 시각화)
    fig = pca_biplot_w_centroids (
        centroids_df, 
        cluster_top_words_df, # biplot에 표시할 군집별 주요 단어
        n_top_loading_words_to_display=0, # biplot에 표시할 loading 기준 상위 단어 갯수
        apply_stdscaler=True, apply_l2=True
        )
    fig.show()

    ## 6) 군집별 브랜드 “중심-거리" 값 추가
    # **NOTE** stdscaler 반영하지 안음. apply_stdscaler=False 인 조건에서만 사용할것
    # data_labeled_dist = add_center_distance(data_labeled, use_cosine)

    ## 7) 군집별 대표 브랜드 추출
    # rep_brands = get_cluster_top_rep_brands(data_labeled_dist, n_brands=5) 
    # rep_brands.loc[0, 'rep_brands']
    
    ## 8) 각 군집 profiling
    # avg star ratings, business categories, avg review count 등
    # data_labeled_dist['cluster'].value_counts().sort_index() # 군집별 브랜드 수


    #=====================================
    # 전체 브랜드 + 선택 키워드 
    #=====================================
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
    words_to_include_exclusively = sorted(list(set(service_format)))
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = [], # ['Buffets'],
        words_to_delete = [],
        words_to_include_exclusively = words_to_include_exclusively,
        )
    apply_stdscaler, use_cosine = True, True
    k = 5

    ## 1) 분석할 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

    ## 2) 군집수(k) 결정
    value_for_selectingk = determine_k(
        data_w_meta_cols=data_w_meta_cols, 
        apply_stdscaler=apply_stdscaler, use_cosine=use_cosine
        )

    ## 3) 군집 라벨링, 군집별 중심값
    data_labeled, centroids_df = labeling_cluster_and_cal_center(
        data_w_meta_cols=data_w_meta_cols, k=k, 
        apply_stdscaler=apply_stdscaler, use_cosine=use_cosine, batch=True
        )
    data_labeled['cluster'].value_counts()

    ## 4) 군집별 주요 단어
    cluster_top_words_df = top_representative_words_for_clusters(centroids_df, top_n=5)
    cluster_top_words_df

    ## 5) 군집 중심벡터 pca 분석 (단어 공간에서 군집간 거리 시각화)
    fig = pca_biplot_w_centroids (
        centroids_df, 
        cluster_top_words_df, # biplot에 표시할 군집별 주요 단어
        n_top_loading_words_to_display=0, # biplot에 표시할 loading 기준 상위 단어 갯수
        apply_stdscaler=True, apply_l2=True
        )
    fig.show()
    
    ## 6) 군집별 브랜드 “중심-거리" 값 추가
    # **NOTE** stdscaler 반영하지 안음. apply_stdscaler=False 인 조건에서만 사용할것
    # data_labeled_dist = add_center_distance(data_labeled, use_cosine)

    ## 7) 군집별 대표 브랜드 추출
    # rep_brands = get_cluster_top_rep_brands(data_labeled_dist, n_brands=5) 
    # rep_brands.loc[0, 'rep_brands']
    
    ## 8) 각 군집 profiling
    # avg star ratings, business categories, avg review count 등
    # data_labeled_dist['cluster'].value_counts().sort_index() # 군집별 브랜드 수



