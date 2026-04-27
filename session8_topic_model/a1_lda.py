'''
1. 토픽모델
1.1 개념 - 문서집합에서 단어가 함께 등장하는 패턴을 바탕으로 토픽(주제)라는 잠재의미구조를 추정
1.2 결과물
    - 문서->토픽 분포: 각 문서가 어떤 토픽을 어느 정도 포함하는지 보여줌
    - 토픽->단어 분포: 각 토픽을 설명하는 대표 단어 분포
1.3 목적
1.3.1 고차원 텍스트를 토픽 공간으로 요약

2. 종류
2.1 LDA - 생성 확률 모델 (확률적 접근)
2.1.1 가정 - 한 문서는 토픽의 혼합, 한 토픽은 단어의 혼합
2.1.2 생성 - 문서별 토픽의 확률분포, 토픽별 단어의 확률분포
2.1.3 단어별 발생횟수를 기반으로 하므로 입력 데이터는 정수 count 이어야함

2.2 NMF (Non-negative Matrix Factorization) - 행렬분해 기반 (선형대수적 접근)
2.2.1 문서-단어행렬 V ≈ WH
    - V: 문서-단어행렬
    - W: (문서수*토픽수), 문서->토픽 가중치 행렬, 각 문서에 대한 각 열(토픽)의 기여도/중요도/가중치를 나타냄. 
    - H: (토픽수*단어수), 토픽->단어 비중 행렬, 각 토픽에 대한 각 단어의 기여도/중요도/가중치. 
2.2.2 생성
    - 문서별 토픽의 가중치 벡터, 토픽별 단어 가중치 벡터

3. LDA 모델 parameters
3.1 n_components (토픽수): 
    - 추출할 토픽 수 
3.2 doc_topic_prior (알파): 
    - 문서에 포함된 토픽이 다양한 정도를 조절 
    - 알파값이 낮으면 대부분의 토픽 확율은 0에 가깝고 일부만 확률이 높아짐. 
    - 알파값이 높으면 문서에 여러 토픽이 골고루 분포함
    - 기본값은 1/토픽수(K)
3.3 topic_word_prior (베타): 
    - 토픽의 단어 분포 정도를 조절
    - 베타값이 낮으면 대부분의 단어 확률이 0에 가깝고 각 토픽이 몇 개 핵심 단어 확률이 높아짐 
    - 베타값이 높으면 많은 단어가 비슷한 확률로 포함됨. 
    - 기본값은 1/어휘수(V)

4. 최적 LDA 모델 parameters 결정을 위한 성능지표
4.1 perplexity
    - 모델이 생성한 확률 분포 (i.e., 문서별 토픽분포, 토픽별 단어분포)가 실제 데이터에 얼마나 잘 맞는지를 나타내는 값
        - 학습된 모델을 바탕으로 각 문서에 나타난 단어들이 발생할 우도(Likelihood) 계산
            - 문서별로 단어들의 등장 확률 계산: 문서별로 단어들의 등장 확률값을 계산
            - 문서별로 관측 단어들의 확률의 곱 (우도) 계산: 각 문서에 대해 이 값이 높을수록 실제 등장하는 단어들을 잘 맞춘다는 의미가 됨
        - 우도에 대해 역수의 기하평균 후 로그변환
    - 값이 낮을수록 모델이 데이터를 더 쉽게 예측
4.2 coherence (토픽 일관성)*
    - 한 토픽의 상위 단어들이 실제 문서에서 얼마나 자주 함께 등장하는지를 정량화한 지표
    - 토픽의 상위 단어들이 실제 문서들에서 함께 자주 등장하면 같은 주제를 말하는 단어일 가능성이 높음 (Coherence 높음) 
    - 반대로 상위 단어들이 실제 문서들에서 서로 따로따로 등장한다면 해당 토픽은 의미가 섞인 잡탕일 가능성이 높음 (예, 피자, 세탁기, 미용실) (Coherence 낮음)
4.3 diversity (고유단어 비율)
    - K(토픽수)*N(토픽별상위단어수)개의 상위단어 리스트에 중복 없이 들어간 단어 비율 (고유단어 비율), (고유 단어 갯수)/(K*N)
    - 값이 높을수록 토픽 간 공유 단어가 적고 구분이 명확함
4.4 Exclusivity (토픽 전용성)*
    - 특정 단어가 한 토픽에만 강하게 속해 있는 정도를 측정
    - (단어 단위) 토픽 k에서 단어 w의 exclusivity, Excl(w, k) = 토픽 k에서 단어 w에 부여한 확률 / 모든 토픽이 단어 w에 부여한 확률의 합 ==> 단어가 여러 토픽에 고르게 분포되어 있으면 분모가 커짐
    - (토픽 단위) 토픽 k의 Top-N 각 단어에 대해 Excl(w, k) 계산후, 평균 --> 토픽 k에 포함된 단어들의 exclusivity, Exclusivity(k) 
    - (모델 단위) 모든 토픽의 Exclusivity(k)를 평균 --> 전체 토픽에서의 exclusivity가 됨, 즉 모델 전체 Exclusivity 가 됨.

    - exclusivity값이 높으면 각 토픽만의 “시그니처” 단어가 많음
    - exclusivity값이 낮으면 단어가 다수 토픽에 공유, 토픽 구분 모호

'''

import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import plotly.graph_objects as go
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import sys
from importlib import reload
sys.path.append('')
from lib.lib_dtm import lib_filtering_dtm as lfd
reload(lfd)

### 공통 조건 설정
meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 
PATH_to_save = ""


#=================================
# 0. 데이터 불러오기 
#=================================

#=================================
# 1. Grid‑search LDA 
#=================================
# lda param 조건별 판단지표 계산
def lda_parma_grid_search(data_w_meta_cols, param_grid, top_n=10):

    #----------------------------
    # 0. 데이터 전처리
    #----------------------------
    ### 메타 컬럼 제거
    meta_cols = [col for col in data_w_meta_cols.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    data = data_w_meta_cols.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    ### 데이터 준비
    count_df = data.copy()
    feature_names = count_df.columns.to_numpy()
    count_matrix = count_df.values.astype(int)  # LDA에 사용하는 데이터는 integer counts

    ### 문서별 토큰 리스트 복원 – Coherence 계산용
    # count_matrix(문서×단어) → [[tok1,tok1,...], ...]
    token_lists = []
    for row in count_matrix: # 각 문서에 대해 실행
        tokens = [] # 각 문서별 토큰(워드) 리스트 담을 리스트
        for idx, cnt in enumerate(row):
            if cnt: # 등장횟수가 0인 단어는 건너뜀
                tokens.extend([feature_names[idx]] * cnt) # 해당 토큰(word)을 등장횟수(cnt) 만큼 리스트에 추가
        token_lists.append(tokens)
    dictionary = Dictionary(token_lists) # gensim Dictionary 생성 - CoherenceModel 내부 연산에서 사용

    #----------------------------
    # 2) grid search
    #----------------------------
    param_search_results = list() # 각 조합 결과 저장할 list

    # params = {'doc_topic_prior': None, 'n_components': 5, 'topic_word_prior': None} # for 문 내부 테스트용
    for params in ParameterGrid(param_grid): # 각 조합의 params에 대해서 실행
        print("params: ", params)

        #----------------------------------
        # 1) 모델 학습
        lda = LatentDirichletAllocation(
            **params, # 딕셔너리 안에 든 키-값 쌍을 키워드 인자로 한꺼번에 풀어서 전달
            learning_method="batch",  # batch (모든 문서 한꺼번에 학습), online (미니배치 단위로 순차처리)
            random_state=42,
            n_jobs=-1, # CPU 코어 모두 사용
            max_iter=10 #100, # 반복 횟수
            )
        lda.fit(count_matrix) # 모델 학습 (fit)

        # 토픽 상위단어 추출
        topics = list()
        for comp in lda.components_:  # components_: 토픽-단어 기대 count, comp: 길이 V(단어 수) 벡터            
            top_indices = comp.argsort()[::-1][:top_n] # argsort()로 작은 -> 큰 순 인덱스 반환 -> [::-1]으로 역순으로 뒤집은 후, [:top_n]상위 N개 선택            
            topics.append([feature_names[i] for i in top_indices]) # feature_names 배열에서 해당 인덱스의 실제 단어 문자열 추출

        #----------------------------------
        ## 2a) perplexity (낮을수록 좋음)
        perp = lda.perplexity(count_matrix) # 학습된 모델 평가 – perplexity 계산
        # print(f"perplexity: {perp}")

        #----------------------------------
        # 2b) Coherence, C_v (높을수록 좋음)
        cm = CoherenceModel(
                topics=topics, # 토픽별 상위단어 리스트 (list[list[str]])
                texts=token_lists, # 원본 문서 토큰 (list[list[str]])
                dictionary=dictionary, # gensim Dictionary(토큰과 숫자id 매핑)
                coherence='c_v' # 계산알고리즘, 'u_mass', 'c_uci', 'c_npmi' 도 동일 방식
            )
        c_v = cm.get_coherence()
        # print(f"C_v Coherence: {c_v:.3f}")    

        #----------------------------------
        # 2c) Diversity (높을수록 좋음)
        flat_terms = [w for topic in topics for w in topic] # 토픽별 단어리스트의 리스트를 평탄화, 예) [['pizza','crust'], ['sushi','roll']] -> ['pizza','crust','sushi','roll']
        diversity  = len(set(flat_terms)) / (len(topics) * top_n) # 중복 제거한 고유단어수/전체 단어수

        #----------------------------------
        # 2d) Exclusivity (높을수록 좋음)
        phi = lda.components_.astype(float) # 토픽-단어 기대 count
        phi = phi/phi.sum(axis=1, keepdims=True) # 행 L1 정규화 -> 토픽 k 안에서 단어 w 의 확률, P(w|k)

        excl_per_topic = list() # 토픽별 독점성 저장할 리스트
        # row = phi[0] # 확인용
        for k, row in enumerate(phi):
            idx_top = np.argsort(row)[::-1][:top_n] # 내림차순 정렬 후 앞에서 top_n개의 인덱스 추출

            # 각 단어의 해당 토픽에서의 확률 / 각 단어의 모든 토픽에서의 확률 합
            excl = row[idx_top] / phi[:, idx_top].sum(axis=0) # 분자: 해당 토픽 k 에서의 확률, 분모: 모든 토픽에서의 확률 총합, 값이 1에 가까우면 토픽k 전용단어
            excl_per_topic.append(excl.mean()) # N개 평균 -> 토픽 k의 Exclusivity 점수
        exclusivity = float(np.mean(excl_per_topic)) # K개 토픽 평균 -> 모델 점수

        #----------------------------------
        # 결과 누적
        param_search_results.append({
            **params, # 딕셔너리 언팩 - 모든 키–값 쌍을 그대로 펼쳐서 새 딕셔너리의 항목으로 추가함
            "perplexity":   perp,
            "c_v":          c_v,
            "diversity":    diversity,
            "exclusivity":  exclusivity
        })

    ### Grid‑search 결과 DataFrame 정리
    param_search_results_df = pd.DataFrame(param_search_results)
    param_search_results_df = param_search_results_df[['n_components', 'doc_topic_prior', 'topic_word_prior', 'perplexity', 'c_v', 'diversity', 'exclusivity']] # 컬럼 순서 조정
    param_search_results_df = param_search_results_df.sort_values(['n_components', 'doc_topic_prior', 'topic_word_prior']).reset_index(drop=True) # 정렬

    return param_search_results_df

# 그래프 - lda_parma_grid_search결과 그래프로 표시
def lda_parma_grid_search_graph(param_search_results_df):

    df = param_search_results_df.copy()

    # --------------------------------------------------
    # 식별자 컬럼 생성
    # --------------------------------------------------
    # x축: "K_α_β" 문자열 식별자 (예: "10_0.1_None")
    df["param_id"] = (
        df["n_components"].astype(str)
        + "_" + df["doc_topic_prior"].fillna("None").astype(str)
        + "_" + df["topic_word_prior"].fillna("None").astype(str)
    )

    #----------------------------------
    # 그래프
    #----------------------------------
    fig = go.Figure()

    # --------------------------
    # 1) 왼쪽 y축: Perplexity
    fig.add_trace(
        go.Bar(
            x=df["param_id"],
            y=df["perplexity"],
            name="Perplexity (↓)",
            marker_color="#1f77b4", # 파랑
            yaxis="y", # 왼쪽 y축 사용
            offsetgroup=0 # 막대 그룹 0 (왼쪽)
        )
    )

    # --------------------------
    # 2) 오른쪽 y축: Coherence / Diversity / Exclusivity 지표를 순차적으로 배열

    ### 그래프에 표시할 지표와 색깔 지정
    metric_color = {
        "c_v": "#ff7f0e",  # 주황
        "diversity": "#2ca02c",  # 초록
        "exclusivity": "#d62728" # 빨강
        }

    for i, metric in enumerate(metric_color, start=1): # dict를 enumerate에 넣으면 키값을 기준으로함
        fig.add_trace(
            go.Bar(
                x=df["param_id"],
                y=df[metric],
                name=f"{metric}",
                marker_color=metric_color[metric],
                yaxis="y2", # 오른쪽 y축
                offsetgroup=i # 그룹 1,2,3 … -> 각 metric의 막대가 겹치지 않고 나란히 배열됨
            )
        )

    # --------------------------
    # 3) 레이아웃: 두 y축 설정 + 그룹 모드
    fig.update_layout(
        title="Grid-Search 결과: Perplexity vs Quality Metrics",
        xaxis=dict(
            title="Hyper-parameters (K_α_β)",
            tickangle=-45
        ),
        yaxis=dict(
            title="Perplexity",
            rangemode="tozero"
        ),
        yaxis2=dict(
            title="Quality Metrics",
            overlaying="y",
            side="right",
            rangemode="tozero"
        ),
        barmode="group",  # offsetgroup과 함께 '나란히' 배치
        bargap=0.15,
        width=1100,
        height=550,
        legend=dict(title="지표", orientation="h", x=0, y=-0.2)
    )
    return fig

#=================================
# 2. 최적 파라미터로 LDA 학습 + 토픽‑단어 확률 분포 + 문서‑토픽 확률 분포 계산
#=================================
def traing_lda_best(data_w_meta_cols, params_slted):

    data_w_meta_cols = data_w_meta_cols.set_index('name')

    ### 메타 컬럼 제거
    meta_cols = [col for col in data_w_meta_cols.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    data = data_w_meta_cols.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    ### 데이터 전처리
    count_df = data.copy()
    feature_names = count_df.columns.to_list() 
    count_matrix = count_df.values.astype(int)

    n_components_slted = params_slted['n_components_slted']
    doc_topic_prior_slted = params_slted['doc_topic_prior_slted']
    topic_word_prior_slted = params_slted['topic_word_prior_slted']

    # ------------------------------------------------
    # 최적 파라미터로 LDA 학습
    # ------------------------------------------------
    # lda 모델 객체 생성 & 학습
    lda_best = LatentDirichletAllocation(
        n_components = n_components_slted,  # K
        doc_topic_prior = doc_topic_prior_slted,  # α
        topic_word_prior = topic_word_prior_slted, # β
        learning_method = "batch", # batch (모든 문서 한꺼번에 학습), online (미니배치 단위로 순차처리)
        max_iter = 10,
        n_jobs = -1,
        random_state = 42,
    )
    lda_best.fit(count_matrix) # 실제 데이터를 넣어 모델 학습: 토픽-단어 분포 phi만 추청

    # --------------------------------------------------
    # φ (토픽‑단어 확률 분포) 계산
    # --------------------------------------------------
    # best_lda.components_  : shape = (K, V), 각 원소 = 단어 ‘가중치’ (pseudocount)
    # 행 L1 정규화(L1‑norm) → 확률 분포로 변환 (합 = 1)

    phi_raw = lda_best.components_.astype('float') # 토픽-단어 기대 count
    phi_norm = phi_raw / (phi_raw.sum(axis=1, keepdims=True) + 1e-12) # 행 L1 정규화하여 확률 분포(phi_norm)로 변환. 분모 0이 되는 경우 발행하는 오류방지 위해 매우 작은 값 더해줌. keepdims=True은 sum 계산후에도 기존 차원을 유지

    topic_word_prob_df = pd.DataFrame(phi_norm, columns=feature_names).rename(index=lambda i: f"Topic{i+1}") # DataFrame으로 변환. rename(index=)는 모든 index명을 순회하면서, 각각의 index명을 인자 i로 넣어 lambda함수를 호출함.

    ## 저장
    # topic_word_prob_df.reset_index().to_csv(f"{PATH_to_save}/lda_k{n_components_slted}_a{doc_topic_prior_slted}_b{topic_word_prior_slted}_topic_word_prob.csv", encoding='utf-8-sig', index=False)

    # --------------------------------------------------
    # θ (문서/브랜드‑토픽 확률 분포) 계산
    # --------------------------------------------------
    # best_lda.transform(X) --> 문서마다 토픽 분포 θ (자동 row‑normalize)

    theta = lda_best.transform(count_matrix) # 문서-토픽 확률분포 (각 문서에서 토픽들의 확률분포)
    # theta.sum(axis=1, keepdims=True) # normalize 여부 확인용

    document_topic_prob_df = pd.DataFrame(theta, index=count_df.index).rename(columns=lambda i: f"Topic{i+1}") # DataFrame으로 변환. rename(columns=)는 모든 컬럼명을 순회하면서, 각각의 컬럼명을 인자 i로 넣어 lambda함수를 호출함.
    document_topic_prob_df["main_topic"] = document_topic_prob_df.idxmax(axis=1) # 브랜드가 가장 많이 포함한 토픽 컬럼 추가. idmax(axis=1)는 각 행에서 최대값을 가지는 컬럼명을 반환
    # document_topic_prob_df['main_topic'].value_counts()

    ## meta cols 추가
    document_topic_prob_w_meta_cols= pd.concat([data_w_meta_cols[meta_cols], document_topic_prob_df], axis=1)

    ## 저장
    # document_topic_prob_w_meta_cols.reset_index().to_csv(f"{PATH_to_save}/lda_k{n_components_slted}_a{doc_topic_prior_slted}_b{topic_word_prior_slted}_document_topic_prob.csv", encoding='utf-8-sig', index=False)

    return lda_best, topic_word_prob_df, document_topic_prob_w_meta_cols

#=================================
# 3. 토픽별 Top-N 키워드 추출 (절대비중, 전용성, 혼합 기준)
#=================================
def extract_topic_keywords(topic_word_prob_df, method = "phi_excl", top_n=10):

    #------------------------------
    # 토픽-단어 확률 데이터
    #------------------------------
    phi = topic_word_prob_df.values.astype(float)  # 토픽-단어 확률 (K,V) phi[k, v]: 토픽 k 에서 단어 v 가 차지하는 절대 비중(확률)
    vocab = topic_word_prob_df.columns.to_numpy() # 단어
    K = phi.shape[0] # 토픽수

    #------------------------------
    # Exclusivity 행렬 계산
    #------------------------------
    col_sum = phi.sum(axis=0, keepdims=True) # 각 단어별 모든 토픽에서의 확률 총합 (1,V)
    excl = phi / col_sum # 각 토픽에서 각 단어의 전용성 (K,V), excl[k, v]: 토픽 k 에서 단어 v의 전용성 (각 단어의 각 토픽에서의 확률 / 각 단어의 모든 토픽에서의 확률 총합)

    #------------------------------
    # 점수 계산 방식 선택에 따른 최종 점수 계산  
    #------------------------------
    if method == "phi": # 단어가 토픽내에서 치지하는 절대 비중 (확률) 기준, 파이
        score = phi 
    elif method == "excl": # 전용성 기준 (절대 확률이 낮을 경우 대표성이 떨어질 수 있음)
        score = excl 
    elif method == "phi_excl": # 절대비중 × 전용성 (해당 단어가 토픽 안에서도 비중이 크고, 다른 토픽엔 잘 안 쓰이는 정도를 동시에 반영)        
        score = phi * excl # 동일 위치 셀끼리 곱하는 ‘원소별 곱’: score[k, v] = phi[k, v] * excl[k, v], 토픽 k 에서 단어 v 가 차지하는 절대 비중(확률)*그 단어가 토픽 k 에서의 전용성
    else:
        raise ValueError("method는 'phi', 'excel', 'phi_excl' 중에서 선택해야함")

    #------------------------------
    # 토픽별 Top‑N 인덱스 -> 단어 변환
    #------------------------------
    keywords = []
    for k in range(K): # 각 토픽별
        idx = score[k].argsort()[::-1][:top_n] # score 상위 top_n개의 인덱스 추출
        keywords.append([vocab[i] for i in idx]) # 해당 인덱스에 해당하는 단어들 추출
    kw_df = pd.DataFrame(keywords, columns=[f"kw{i+1}" for i in range(top_n)]).rename(index=lambda i: f"Topic{i+1}") # df로 변환

    return kw_df

#=================================
# 4. topic_word 분포 데이터에 pca 적용 시각화, topic간 상대적 위치 확인
#=================================
def pca_biplot_w_topic_word_prob (topic_word_prob_df, top_kws_df, n_top_loading_words_to_display, apply_stdscaler, apply_l2=True):

    # ------------------------------------------------------------------
    # 1) 데이터 표준화
    # ------------------------------------------------------------------
    # topic_word_prob_df - 각 토픽별, 단어들의 확률의 합은 1이므로 이미, L2 normaiize 되어 있는 것임.
    X_scaled = topic_word_prob_df.values

    if apply_stdscaler == True:
        X_scaled = StandardScaler().fit_transform(X_scaled) # 단어 스케일 차이 제거, 각 단어(열) 별로 표준화 (value-mean)/stderror
    if apply_l2 == True: # 행별 l2 normalize (이미 되어 있어 필요없지만 그대로 둠)
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    # ------------------------------------------------------------------
    # 2) PCA (PC1, PC2만)
    # ------------------------------------------------------------------
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(X_scaled) # (10, 2) -> 군집 점 좌표
    loadings = pca.components_.T # (411, 2) -> 단어 로딩 (화살표 벡터)

    # ------------------------------------------------------------------
    # 3) 시각화용 단어 선정 - 절대값 기준, 상대차이 기준 top word 각각에 대해서 해보고 비교할것
    # ------------------------------------------------------------------
    # 1) 대표단어 집합 
    if len(top_kws_df) == 0:
        rep_words = list()
    else:
        # top_kws_df에 포함된 모든단어를 리스트로 추출
        rep_words = list(set(top_kws_df.to_numpy().flatten().tolist())) 

    # 2) 기존 loading 상위 단어 추출
    if n_top_loading_words_to_display == 0:
        loading_words = list()
    else:
        abs_loading_sum = np.abs(loadings).sum(axis=1)
        top_idx = np.argsort(abs_loading_sum)[::-1][:n_top_loading_words_to_display] # np.argsort 오름차순정렬를 역순으로 한 후(즉 내림차순정렬 후) 인덱스 반환, 상위 n_top_loading_words_to_add 수만큼 인덱스 추출
        loading_words = list(topic_word_prob_df.columns[top_idx])

    # 3) 두 집합 union → plotting 단어
    plot_words = list(set(loading_words + rep_words)) 
    plot_words_vectors = loadings[[topic_word_prob_df.columns.get_loc(w) for w in plot_words]] # w 컬럼명이 몇 번째 위치(index)에 있는지 반환

    # ------------------------------------------------------------------
    # 4) Plotly Biplot
    # ------------------------------------------------------------------
    ## 1) pca score, coeff 값 scale 차이 보정 
    pca_score_scope = np.percentile(np.abs(scores), 85)
    coeff_scope = np.percentile(np.abs(plot_words_vectors), 90)
    scale_factor = pca_score_scope / coeff_scope # scale 자동설정
    
    ## 그래프
    fig = go.Figure()

    # 4-1) 군집 중심 점 (scatter)
    fig.add_trace(go.Scatter(
        x=scores[:, 0], y=scores[:, 1],
        mode='markers+text',
        text=topic_word_prob_df.index,
        textposition='top center',
        marker=dict(size=12, color='midnightblue'),
        name='Topics'
    ))

    # 4-2) 단어 화살표    
    for word, vec in zip(plot_words, plot_words_vectors):
        fig.add_trace(go.Scatter(
            x=[0, vec[0] * scale_factor],
            y=[0, vec[1] * scale_factor],
            mode='lines+text',
            line=dict(color='tomato', width=1),
            text=[None, word],
            textposition='top center',
            showlegend=False
        ))

    # 4-3) 레이아웃
    fig.update_layout(
        # width=900, height=700,
        title=f"PCA Biplot of {len(topic_word_prob_df)} Topics "
            f"(explained {pca.explained_variance_ratio_[:2].sum()*100:.1f} %)",
        xaxis_title='PC1',
        yaxis_title='PC2',
        template='simple_white'
    )
    # fig.show()
    return fig

#=================================
# 5. 브랜드‑토픽 히트맵 시각화 (Plotly)
#=================================
def doucment_topic_heatmap(document_topic_prob_w_meta_cols, n_document_to_graph):

    #----------------------------
    # 데이터 전처리
    #------------------------------
    # 메타 컬럼 제거
    meta_cols = [col for col in document_topic_prob_w_meta_cols.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    data = document_topic_prob_w_meta_cols.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    #----------------------------
    # 그래프 그리기
    #------------------------------
    heat_df = data.drop(columns=["main_topic"]).head(n_document_to_graph) # 그래프에 나타낼 브랜드 수 지정

    fig = go.Figure(
        go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns,
            y=heat_df.index,
            colorscale="YlGnBu",  # 연속형 팔레트 (Blue‑Green‑Yellow)
            colorbar=dict(title="Topic proportion"),
            zmin=0,
            zmax=float(heat_df.values.max()),
        )
    )
    fig.update_layout(
        title="Brand‑wise Topic Distribution (LDA on Count DTM)",
        xaxis_title="Topics (K)",
        yaxis_title="Restaurant Brands",
        width=900,
        height=1000,
    )
    # fig.show()
    return fig


#-------------------------------------
if __name__ == "__main__":

    #-------------------------------
    ### 전체 브랜드 + 전체 키워드
    ## 0) 분석할 원 데이터 추출
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = [],
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions=input_data_filtering_conditions)

    ## 1) Grid‑search LDA
    param_grid1 = {
        "n_components": [5, 10, 15], # 토픽 수 후보
        "doc_topic_prior": [None, 0.1, 0.5], # α (θ sparsity)
        "topic_word_prior": [None, 0.01], # β (φ sparsity)
        }
    param_grid2 = {
        "n_components": range(2,21), # 토픽 수 후보
        "doc_topic_prior": [None], # α (θ sparsity)
        "topic_word_prior": [None], # β (φ sparsity)
        }
    param_search_results_df = lda_parma_grid_search(data_w_meta_cols=data_w_meta_cols, param_grid=param_grid2)
    
    fig_param_search = lda_parma_grid_search_graph(param_search_results_df) # 파라미터별 성능지표 그래프로 표시
    fig_param_search.show() # 토론: param_search_results_df 해석, 최적 파라미터 찾기

    ## 2) 최적 parameter로 lda 모델 학습
    params_slted = dict(
        n_components_slted = 11, # 5, 7, 11
        doc_topic_prior_slted = None, 
        topic_word_prior_slted = None
        )
    lda_best, topic_word_prob_df, document_topic_prob_w_meta_cols = traing_lda_best(data_w_meta_cols, params_slted)

    ## 3) 토픽별 Top-N 키워드 추출 (절대비중, 전용성, 혼합 기준)
    top_n = 10 # 토픽당 추출할 단어 수
    kw_phi = extract_topic_keywords(topic_word_prob_df, "phi", top_n)
    kw_excl = extract_topic_keywords(topic_word_prob_df, "excl", top_n)
    kw_phi_excl = extract_topic_keywords(topic_word_prob_df, "phi_excl", top_n)

    ## 4) topic_word 분포 데이터에 pca 적용 시각화, topic간 상대적 위치 확인
    top_kws_df = kw_phi # pca biplot에 표시할 키워드 , ""이면 키워드 포함하지 않음을 의미
    n_top_loading_words_to_display = 0 # pca biplot에 표시할 pca loadings 기준 상위 단어 갯수
    fig = pca_biplot_w_topic_word_prob(topic_word_prob_df, top_kws_df, n_top_loading_words_to_display, apply_stdscaler=True)
    fig.show()

    ## 5) 브랜드‑토픽 히트맵 시각화 (Plotly)
    fig = doucment_topic_heatmap(document_topic_prob_w_meta_cols, n_document_to_graph=30)
    fig.show()

    ## 6) 토픽별 브랜드 profiling


    #-------------------------------
    ### 특정카테고리 브랜드 + 전체 키워드
    ## 0) 분석할 원 데이터 추출
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = ["Italian"],
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions=input_data_filtering_conditions)

    ## 1) Grid‑search LDA
    param_grid1 = {
        "n_components": [5, 10, 15], # 토픽 수 후보
        "doc_topic_prior": [None, 0.1, 0.5],  # α (θ sparsity)
        "topic_word_prior": [None, 0.01],  # β (φ sparsity)
        }
    param_grid2 = {
        "n_components": range(2,21), # 토픽 수 후보
        "doc_topic_prior": [None], # α (θ sparsity)
        "topic_word_prior": [None], # β (φ sparsity)
        }
    param_search_results_df = lda_parma_grid_search(data_w_meta_cols=data_w_meta_cols, param_grid=param_grid1)
    
    fig_param_search = lda_parma_grid_search_graph(param_search_results_df) # 파라미터별 성능지표 그래프로 표시
    fig_param_search.show() # 토론: param_search_results_df 해석, 최적 파라미터 찾기

    ## 2) 최적 parameter로 lda 모델 학습
    params_slted = dict(
        n_components_slted = 11, # 6, 11
        doc_topic_prior_slted = None,
        topic_word_prior_slted = None
        )
    lda_best, topic_word_prob_df, document_topic_prob_w_meta_cols = traing_lda_best(data_w_meta_cols, params_slted)

    ## 3) 토픽별 Top-N 키워드 추출 (절대비중, 전용성, 혼합 기준)
    top_n = 10 # 토픽당 추출할 단어 수
    kw_phi = extract_topic_keywords(topic_word_prob_df, "phi", top_n)
    kw_excl = extract_topic_keywords(topic_word_prob_df, "excl", top_n)
    kw_phi_excl = extract_topic_keywords(topic_word_prob_df, "phi_excl", top_n)

    ## 4) topic_word 분포 데이터에 pca 적용 시각화, topic간 상대적 위치 확인
    top_kws_df = kw_phi # pca biplot에 표시할 키워드 , ""이면 키워드 포함하지 않음을 의미
    n_top_loading_words_to_display = 0 # pca biplot에 표시할 pca loadings 기준 상위 단어 갯수
    fig = pca_biplot_w_topic_word_prob(topic_word_prob_df, top_kws_df, n_top_loading_words_to_display, apply_stdscaler=True)
    fig.show()

    ## 5) 브랜드‑토픽 히트맵 시각화 (Plotly)
    fig = doucment_topic_heatmap(document_topic_prob_w_meta_cols, n_document_to_graph=30)
    fig.show()

    ## 6) 토픽별 브랜드 profiling


