'''
단어들의 덴드로그램을 통해 유사도 관계 파악

'''

import plotly.io as pio
pio.renderers.default = "browser" # 브라우저에 그래프 출력하는 것으로 디폴트로설정 - 한번만 하면됨 

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.figure_factory as ff
import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

import sys
from importlib import reload
sys.path.append('')
from lib.lib_dtm import lib_filtering_dtm as lfd
reload(lfd)


### 조건 설정
meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 

# ────────────────────────────────────────────────────────────
# 1) 조건에 맞는 데이터 추출

# ────────────────────────────────────────────────────────────
# 2) 계층적 군집분석 수행
def hierarchical_clustering_words(data_w_meta_cols, k, apply_stdscaler=False, use_cosine=True, method="average", metric="cosine"):

    """
    Parameters
    ----------
    data_w_meta_cols: dtm 데이터    
    k: 클러스터 수
    apply_stdscaler: 열(단어)별 표준화, (value-mean)/stderror
    use_cosine: 코사인 거리 사용 여부
    method: 클러스터 간 거리 계산 기준
    metric: 거리 계산법

    Returns
    -------
    data_labeled: 인풋 data에 군집 라벨 추가된 dataframe
    fig_dendro: 군집 덴드로그램
    """

    #=================================
    # 데이터 전처리
    #=================================
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터
    
    X = X.T # 단어별 군집분석을 위해 행열 transpose
    
    ### 데이터 표준화
    X_scaled = X.copy() # 표준화 적용하지 않을 경우 
    if apply_stdscaler == True:
        X_scaled = StandardScaler().fit_transform(X) # 각 단어(열) 별로 표준화 (value-mean)/stderror

    # 코사인 거리 사용시 데이터 단위길이로 정규화
    if use_cosine: # 코사인 거리 사용 시 각 행을 길이 1로 L2 정규화, 군집분석할때 metric="cosine"을하면 L2정규화를 자체적으로 수행하기 때문에 필요없음. 하지만, 미리 먼저 해도 상관없으므로 그대로 둠
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    #=================================
    # linkage 행렬 생성 (계층적 군집분석)
    #=================================
    # 군집병합 방법: 최초 n개의 군집 (즉, 각각의 개별 브랜드가 군집)에서 시작해서, 1회 병합에서 가장 가까운 두 군집이 합쳐서서 n-1개의 군집이 남고, 2회 병합에서 이들 n-1개 군집 중 가장 가까운 두 군집이 합쳐서 n-2개 군집이 남고, .... n-1회 병합에서 마지막 남은 2개 군집이 합쳐져서 단일 군집이 됨.
    # 거리 계산 업데이트: 코사인·유클리드 등 선택한 거리 metric으로 모든 샘플 쌍 간 거리를 구해 1D 거리벡터 생성 (n(n-1)/2 길이) → 병합이 일어날때마다 새 군집과 나머지 군집 사이 거리를 update함 → 이 거리벡터를 기반으로 군집병합 수행

    linked = linkage(X_scaled, method=method, metric=metric) # Ward 는 'euclidean' 만 지원
    # linked.shape # (n-1)*4  컬럼0,1: 병합된 두 군집의 인덱스, 컬럼2: 두 군집사이의 거리, 컬럼3: 합쳐진 새 군집의 표본수

    #=================================
    # flat cut - 트리를 k개로 ‘컷’해서 군집 라벨 얻기
    #=================================
    cluster_labels = fcluster(linked, t=k, criterion="maxclust") # 군집 개수를 최대 k개로 맞춰 자르라는 뜻

    ### 원본 데이터에 클러스터 라벨 추가 (반환용)
    data_labeled = X.copy() 
    data_labeled['cluster'] = cluster_labels
    
    #=================================
    # 덴드로그램
    #=================================
    fig_dendro = ff.create_dendrogram(
        X_scaled,
        labels=X.index.tolist(),
        linkagefun=lambda _: linked   # 이미 계산한 linked 재사용
    )
    fig_dendro.update_layout(
        # width=1200,
        # height=600,
        autosize = True,
        title="Hierarchical Clustering Dendrogram",
        xaxis_title="Word",
        yaxis_title="Distance"
    )

    return data_labeled, fig_dendro


if __name__ == "__main__":

    #=====================================
    # 전체 브랜드 + 전체 키워드
    #=====================================
    ## 1) 분석할 원 데이터 추출
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = [],
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

    ## 2) Hierarchical Clustering + 덴드로그램
    k = 20
    apply_stdscaler=False; use_cosine=True
    method="ward" 
    metric="euclidean"
    np.random.seed(44)
    data_labeled, fig_dendro = hierarchical_clustering_words(data_w_meta_cols.sample(100), k, apply_stdscaler=apply_stdscaler, use_cosine=use_cosine, method=method, metric=metric)
    data_labeled['cluster'].value_counts()
    fig_dendro.show()

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


    # 1) 분석할 원 데이터 추출
    price = ['cheap', 'coupon', 'free', 'worth'] # 'price', 
    quality = ['authent', 'fresh', 'healthi', 'green', 'qualiti'] #, 'fast' 
    eval = ['amaz', 'awesom', 'bad', 'best', 'better', 'enjoy', 'favorit', 'good', 'great', 'love', 'perfect', 'nice', 'happi', 'disappoint']
    taste = ['chili', 'sour', 'sweet'] # delici
    words_to_include_exclusively = sorted(list(set(eval+quality)))

    words_to_include_exclusively = sorted(list(set(service_format)))
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = [], # ['Buffets'],
        words_to_delete = [],
        words_to_include_exclusively = words_to_include_exclusively,
        )
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
    
    # 2) Hierarchical Clustering + 덴드로그램: ward+euclidean    
    k = 5
    apply_stdscaler=False; use_cosine=True
    method="ward" 
    metric="euclidean"
    data_labeled, fig_dendro = hierarchical_clustering_words(data_w_meta_cols, k, apply_stdscaler=apply_stdscaler, use_cosine=use_cosine, method=method, metric=metric)
    data_labeled['cluster'].value_counts()
    fig_dendro.show()

