import plotly.io as pio
pio.renderers.default = "browser"  # VS Code·Jupyter 환경에 맞게 설정

import pandas as pd
from sklearn.metrics import pairwise_distances 
from sklearn.manifold import MDS 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans, KMeans

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

import sys
from importlib import reload
sys.path.append('/Users/carrot/Dropbox/Learning/inflearn/902_textanalytics_class/class20261')
from lib.lib_dtm import lib_filtering_dtm as lfd
from s11_mds_pmap import a1_mds_perBrand
reload(lfd); reload(a1_mds_perBrand)

meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 

# ────────────────────────────────────────────────────────────
# 1) 조건에 맞는 데이터 추출
# ────────────────────────────────────────────────────────────
# lib.lib_dtm

# ────────────────────────────────────────────────────────────
# 2) MDS 계산
# ────────────────────────────────────────────────────────────
# s11_mds_pmap.a1_mds_perBrand

# ────────────────────────────────────────────────────────────
# 3) 군집분석 w MDS (군집라벨, 중심 산출)
# ────────────────────────────────────────────────────────────
def labeling_cluster_on_mds(mds_w_meta_cols, k, apply_stdscaler=False, use_cosine=False, batch=True):
    """
    Parameters
    ----------
    mds_w_meta_cols : create_mds_perBrand(...)의 결과 (name, dim1, dim2 포함)
    k : 클러스터 수 (반드시 지정)
    apply_stdscaler : MDS 좌표(dim1, dim2)에 표준화 적용 여부 (기본 False)
    batch : MiniBatchKMeans 사용 여부 (True: MiniBatchKMeans, False: KMeans)

    Returns
    -------
    data_labeled : mds_w_meta_cols + 'cluster' 열 추가
    centroids_df : 군집별 중심값 (dim1, dim2) DataFrame
    """

    #=================================
    # 데이터 전처리
    #=================================
    df = mds_w_meta_cols.set_index('name')
    dim_cols = ['dim1', 'dim2']
    X = df[dim_cols].copy() # MDS 좌표만 사용하여 군집

    ### 데이터 표준화
    X_scaled = X.values
    if apply_stdscaler:
        X_scaled = StandardScaler().fit_transform(X)

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
    labels = model.fit_predict(X_scaled)
    centroids = model.cluster_centers_ # 군집별 MDS 좌표계의 중심, shape (k, 2) 

    #=================================
    # 라벨 부착
    #=================================
    data_labeled = mds_w_meta_cols.copy()
    data_labeled['cluster'] = labels

    #=================================
    # 중심값 DataFrame
    #=================================
    centroids_df = pd.DataFrame(
        centroids,
        index=range(k),
        columns=dim_cols
    )
    centroids_df.index.name = 'cluster' # index 이름을 cluster로 변경

    return data_labeled, centroids_df

# ────────────────────────────────────────────────────────────
# 4) MDS+군집 시각화 (Perceptual Map) 
# ────────────────────────────────────────────────────────────
def pmap_w_mds_clustered(
    mds_w_cluster, centroids_df, size_col, 
    stress1, dist_metric, apply_stdscaler, apply_l2, input_data_filtering_conditions # mds 계산조건 표시용 (그래프 작성에는 사용하지 않음)
    ):

    df = mds_w_cluster.copy()

    #=================================
    # 버블 크기 계산 
    #=================================
    min_size, max_size = 5, 30 # 버블 최소, 최대 크기
    rc_norm = (df[size_col] - df[size_col].min()) / (df[size_col].max() - df[size_col].min())
    sizes = rc_norm * (max_size - min_size) + min_size

    #=================================
    # 색상 팔레트 설정 (군집별 다른 색 적용하기 위함)
    #=================================
    palette = px.colors.qualitative.Set2 # Plotly Express에서 미리 정의해 둔 색상 팔레트 (8개 색상으로 구성됨)
    num_colors = len(palette)

    #=================================
    # 군집별 그룹 생성
    #=================================
    cluster_groups = df.groupby('cluster', sort=True) # 같은 cluster 번호를 가진 행끼리 묶어서 (cluster, 그룹 DataFrame) 형태의 iterator 반환, sort=True: groupby 키값 (cluster) 기준 정렬
    # for c, g in cluster_groups: print(c, g) # 그룹별 내용확인

    #------------------------
    # 그래프에 표시할 정보
    #------------------------
    subtitle = '<br>'.join([
        f'input_file_name: {input_data_filtering_conditions["input_file_name"]}',
        f'remove_brand_w_word_in_name: {input_data_filtering_conditions["remove_brand_w_word_in_name"]}',
        f'brand_categories_slted: {input_data_filtering_conditions["brand_categories_slted"]}',
        f'words_to_delete: {input_data_filtering_conditions["words_to_delete"]}',
        f'words_to_include_exclusively: {input_data_filtering_conditions["words_to_include_exclusively"]}',
        f'apply_stdscaler: {apply_stdscaler}',
        f'apply_l2_norm: {apply_l2}',        
        f'dist_metric: {dist_metric}',
        f'stress1: {stress1:.4f}',
    ])

    #------------------------
    # 그래프 생성
    #------------------------
    fig = go.Figure()

    ### 1) 브래드별 위치 표시, 군집별로 trace 분리 (범례/색상 구분)
    for c, cluster_group in cluster_groups:
        fig.add_trace(
            go.Scatter(
                x=cluster_group['dim1'],
                y=cluster_group['dim2'],
                mode="markers+text", # 점+라벨
                text=cluster_group['name'], # 브랜드명을 라벨로 사용
                textposition="top center",
                marker=dict(
                    size=sizes[cluster_group.index], # 버블크기
                    color=palette[int(c) % num_colors], # 색 인덱스가 0~7 범위로 순환 (군집번호를 팔레트 색상 개수로 나눈 나머지)
                    opacity=0.9,
                    line=dict(width=0.4, color="rgba(0,0,0,0.6)") # 마커 외곽선
                ),
                name=f"Cluster {int(c)} (n={len(cluster_group)})",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Dim1: %{x:.3f}<br>"
                    "Dim2: %{y:.3f}<br>"
                ),
            )
        )
    ### 2) 군집별 중심 표시
    centers = centroids_df.reset_index()
    fig.add_trace(
        go.Scatter(
            x=centers['dim1'],
            y=centers['dim2'],
            mode="markers+text",
            text=[f"C{int(c)}" for c in centers['cluster']], # 중심점 라벨
            textposition="middle right",
            marker=dict(
                size=14,
                symbol="x",
                color="black",
                line=dict(width=2, color="black")
            ),
            name="Centers",
            hoverinfo="skip" # 마우스오버 시 정보 표시 안 함
        )
    )
    ### 3) 레이아웃
    fig.update_layout(
        title=dict(
            text=f'Brand Perceptual Map via MDS (Clustered)<br><span style="font-size:11px">{subtitle}</span>',
            x=0.05
        ),
        xaxis_title="MDS Dimension 1",
        yaxis_title="MDS Dimension 2",
        template="plotly_white",
        dragmode="pan",
        legend=dict(title="Clusters"),
        # width=900, height=650,
    )
    # fig.show()
    return fig


if __name__ == '__main__':

    ## 1) 분석할 데이터 추출
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = ["Fast Food"], # Fast Food
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
    data_w_meta_cols = data_w_meta_cols[data_w_meta_cols['categories'].str.contains('Burgers')].reset_index(drop=True)
    
    ## 2) mds 계산
    apply_stdscaler, apply_l2 = False, False
    dist_metric = 'cosine' # cosine, euclidean, correlation, manhattan
    mds_w_meta_cols, stress1 = a1_mds_perBrand.create_mds_perBrand(data_w_meta_cols, dist_metric, apply_stdscaler, apply_l2)

    ## 3) 군집분석 w MDS (군집라벨, 중심 산출)
    mds_w_cluster, centroids_df = labeling_cluster_on_mds(mds_w_meta_cols, k=4)
    mds_w_cluster['cluster'].value_counts()
    
    ## 4) MDS+군집 시각화 (Perceptual Map) 
    size_col = 'review_count'
    fig = pmap_w_mds_clustered(
        mds_w_cluster, centroids_df, size_col, 
        stress1, dist_metric, apply_stdscaler, apply_l2, input_data_filtering_conditions # mds 계산조건 표시용 (그래프 작성에는 사용하지 않음)
        )
    fig.show()


