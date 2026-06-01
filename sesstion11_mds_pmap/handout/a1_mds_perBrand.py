'''
1. MDS
1.1 목적
    - 고차원 공간에서의 원자료들을 거리관계를 최대한 보존하는 저차원 공간으로 맵핑하는 거리 기반 차원축소 기법
    - 고차원 데이터를 거리 행렬 (또는 근접도 행렬)로 변환하고, 이 거리(또는 근접도) 관계를 최대한 보존하는 저차원 공간의 점들을 찾아내는 기법
1.2 특징
    - 좌표 축은 해석 불가, 관측치간 유사성을 직관적으로 보여줌

2. 데이터 전처리
2.1 tf 
    - 원래 정보(빈도) 그대로 반영하여 해석이 직관적
    - 문서 길이 영향이 큼. 빈도가 높은 단어가 거리 계산에 과도한 영향을 미칠 위험이 있음 (자주 등장하는 단어를 중요하게, 드물게 등장하는 단어는 덜 중요하게 반영함)
2.2 tfidf
    - 일반적인 단어 영향 억제하고 브랜드별 고유 특성 강조하여, 브랜드간 차이 표현에 유리
    - 희귀하지만 의미 없는 단어에 가중치가 과도하게 주어질 수 있음
2.3 l2 정규화 (행별)
    - 각 문서를 단위 벡터로 맞추어 문서길이, 리뷰수 차이 제거
    - 브랜드 특성 중심으로 표현
2.4 StandardScaler (열별 표준화)
    - 단어별 중요도 차이를 없애는 부작용이 생길 수 있음

3. 거리 계산 방법
3.1 euclidean
    - 절대적 크기/빈도 차이 (브랜드 리뷰 수 차이, 절대 점수 차이) 반영
    - 값이 큰 변수/단어가 전체 거리를 지배할 수 있음0
3.2 cosine
    - 방향, 패턴 기반 (단어 사용 비율)
    - 문서 길이(리뷰 수) 영향 제거, 크기 차이 정보를 완전히 버림
'''

import plotly.io as pio
pio.renderers.default = "browser"  # VS Code·Jupyter 환경에 맞게 설정

import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from sklearn.manifold import MDS
import plotly.graph_objects as go
import numpy as np

import sys
from importlib import reload
sys.path.append("") 
from lib.lib_dtm import lib_filtering_dtm as lfd
reload(lfd)

meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함

# ────────────────────────────────────────────────────────────
# 1) 조건에 맞는 데이터 추출
# ────────────────────────────────────────────────────────────
# lib.lib_dtm

# ────────────────────────────────────────────────────────────
# 2) MDS 계산
# ────────────────────────────────────────────────────────────
def create_mds_perBrand(data_w_meta_cols, dist_metric, apply_stdscaler, apply_l2):

    #------------------------------
    # 데이터 전처리
    #------------------------------
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    #------------------------------
    # 데이터 표준화
    #------------------------------
    X_scaled = X.copy() # 표준화 적용하지 않을 경우 
    if apply_stdscaler == True:
        X_scaled = StandardScaler().fit_transform(X_scaled) # 각 단어(열) 별로 표준화 (value-mean)/stderror
    if apply_l2 == True: # 행별 l2 normalize - 거리행렬 계산시 코사인거리를 사용하면, 여기서 l2정규화는 불필요(결과 동일). 다른 거리계산법을 사용할 경우를 대비해 그대로 둠.
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    #------------------------------
    # 거리 행렬 계산
    #------------------------------
    dist_mat = pairwise_distances(X_scaled, metric=dist_metric) # tfidf 공간에서의 거리 계산

    #------------------------------
    # Metric MDS (2차원) 학습
    #------------------------------    
    mds = MDS(n_components=2,
            dissimilarity="precomputed", # 이미 계산된 거리행렬 사용
            random_state=42,
            n_init=4, # 초기값 반복(기본=4) — stress 안정화
            max_iter=300 # 반복 횟수
            )
    coords = mds.fit_transform(dist_mat)   # shape = (브랜드수, 2)
    coords_df = pd.DataFrame(coords, index=X.index, columns=["dim1", "dim2"]) # dataframe으로 변환    
    mds_w_meta_cols = pd.concat([df, coords_df], axis=1).reset_index() # 기존 data에 mds결과 추가

    #------------------------------
    # Stress-1 직접 계산: 
    # 거리 재현 오차 지표 (원 데이터에서의 브랜드간 거리 구조를 저차원에서의 유클리드 공간에 근사가 되지 않는 정도를 측정)
    #------------------------------
    D_hat = pairwise_distances(coords) # 저차원 공간에서 각 브랜드간의 유클리드 거리
    stress1 = np.sqrt(((dist_mat - D_hat)**2).sum() / (D_hat**2).sum()) 
    print(f"Stress-1 (manual) : {stress1:.4f}")

    return mds_w_meta_cols, stress1

# ────────────────────────────────────────────────────────────
# 3) MDS 시각화 (Perceptual Map) 
# ────────────────────────────────────────────────────────────
def pmap_w_mds(mds_w_meta_cols, stress1, dist_metric, apply_stdscaler, apply_l2, input_data_filtering_conditions):

    #------------------------------
    # 원 설정
    #------------------------------
    ### 원 크기: review_count 기준 (5‒30 사이로 변환)
    min_size, max_size = 5, 30
    rc_norm = (mds_w_meta_cols['review_count'] - mds_w_meta_cols['review_count'].min()) \
            / (mds_w_meta_cols['review_count'].max() - mds_w_meta_cols['review_count'].min())
    sizes = rc_norm * (max_size - min_size) + min_size   # ndarray

    ### 원 색: avg_stars 기준
    color_vals = mds_w_meta_cols['avg_stars']

    #------------------------------
    # 그래프에 표시할 정보
    #------------------------------
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

    #------------------------------
    # 그래프
    #------------------------------
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mds_w_meta_cols['dim1'],
            y=mds_w_meta_cols['dim2'],
            mode="markers+text", # 점 + 라벨
            text=mds_w_meta_cols['name'] , #X.index, # 브랜드명
            textposition="top center",
            marker=dict(
                size=sizes,
                color=color_vals,
                colorscale="RdBu_r", 
                cmin=color_vals.min(),
                cmax=color_vals.max(),
                showscale=True, # 컬러바 표시
                colorbar=dict(title="Avg Stars"),
                # line=dict(width=0.5, color="black"),
                opacity=0.8,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Dim1: %{x:.3f}<br>"
                "Dim2: %{y:.3f}<br>"
                "Avg Stars: %{marker.color:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=f'Brand Perceptual Map via MDS<br><span style="font-size:11px">{subtitle}</span>',
            x=0.05 # 위치
        ),
        xaxis_title="MDS Dimension 1",
        yaxis_title="MDS Dimension 2",
        # width=800,
        # height=600,
        template="plotly_white"
    )
    # fig.show()
    return fig


if __name__ == "__main__":

    #------------------------------------
    # 특정 카테고리 + 전체 키워드   
    #------------------------------------
    ## 1) 분석할 데이터 추출
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = ["Fast Food"], # Burgers
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
    # data_w_meta_cols = data_w_meta_cols[data_w_meta_cols['categories'].str.contains('Burgers')].reset_index(drop=True)

    ## 2) mds 계산
    apply_stdscaler, apply_l2 = True, True
    dist_metric = 'cosine' # cosine, euclidean, correlation, manhattan
    mds_w_meta_cols, stress1 = create_mds_perBrand(data_w_meta_cols, dist_metric, apply_stdscaler, apply_l2)

    ## 3) mds 그래프
    mds_fig = pmap_w_mds(mds_w_meta_cols, stress1, dist_metric, apply_stdscaler, apply_l2, input_data_filtering_conditions)
    mds_fig.show()

    #------------------------------------
    # 특정 카테고리 + 선택 키워드
    #------------------------------------
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

    ## 1) 분석할 데이터 추출
    words_to_include_exclusively = sorted(list(set(location_atmosphere)))
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = ["Fast Food"],
        words_to_delete = [],
        words_to_include_exclusively = words_to_include_exclusively,
        )
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions=input_data_filtering_conditions)
    data_w_meta_cols = data_w_meta_cols[data_w_meta_cols['categories'].str.contains('Burgers')].reset_index(drop=True)

    ## 2) mds 계산
    apply_stdscaler, apply_l2 = False, False
    dist_metric = 'cosine' # cosine, euclidean, correlation, manhattan
    mds_w_meta_cols, stress1 = create_mds_perBrand(data_w_meta_cols, dist_metric, apply_stdscaler, apply_l2)

    ## 3) mds 그래프
    mds_fig = pmap_w_mds(mds_w_meta_cols, stress1, dist_metric, apply_stdscaler, apply_l2, input_data_filtering_conditions)
    mds_fig.show()




