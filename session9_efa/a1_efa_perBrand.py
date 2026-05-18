'''
1. EFA 탐색적 요인분석
1.1 개념
    - 관측된 변수 (예, 단어)들 간의 상관을 더 작은 수의 잠재 변수(Factor)로 설명하는 기법
    - 문서×단어 행렬에서 단어들의 공분산(covariance) 구조를 분해해 숨은 잠재 요인을 찾음
1.2 결과물
    - Λ 요인적재행렬 (loading matrix): 각 단어가 각 요인과 얼마나 관련되어 있는지를 나타냄 (단어수x요인수) -> 각 요인의 의미 해석 
    - F 요인점수행렬 (factor score): 각 문서가 각 요인과 얼마나 관련되어 있는지를 나타냄, 문서수x요인수
1.3 용도
    - 잠재 인자 식별
    - 브랜드/제품의 포지셔닝 축을 텍스트에서 도출

2. 가정: 변수들간의 상관이 충분해야 함 (상관관계가 없으면 요인 형성이 불가)
2.1 KMO ((Kaiser-Meyer-Olkin)): 
    - 변수 간 상관관계가 얼마나 요인(factor)에 의해 설명될 수 있는지를 나타냄
    - 0과 1 사이의 값을 가지며: 1에 가까울수록 요인분석에 적합. 0에 가까울수록 부적합
2.2 Bartlett test: 
    - 상관행렬이 단위행렬(identity matrix)과 유의하게 다른지를 테스트
    - 상관행렬이 단위행렬에 가까울수룩 변수들 사이에 상관관계가 없다는 뜻이며, 이는 요인분석에 부적합 (H0: 상관행열이 단위행별, H1: 단위행렬아님)

3. 요인수 결정
    - 상관행렬의 고유값 λ: 각 요인이 설명하는 분산의 크기
    - 스크리플롯: 상관행렬의 고유값(λ₁≥λ₂≥…)을 순서대로 그려 ‘팔꿈치’(기울기 급변) 지점 
    - 고유값 > 1 (보조적) 데이터 표준화시 각 변수의 분산=1, 즉 각 요인이 1개 보다 많은 변수의 분산을 설명할 경우 포함한다는 의미임

4. 상관행렬로부터 요인 추출 방법
4.1 전체분산 기반
    - Principal Factor Method: "principal"
        - 변수의 전체 분산(공통 분산 + 고유 분산)을 모두 요인이 설명하도록 함, 주성분 분석과 동일한 방식임 
        - 정보 보존 & 차원 축소에 초점
4.2 공통 분산 기반
    - 변수 간의 관계(공분산)에서 기인한 공통 분산만을 사용하여 숨겨진 잠재 요인 탐색
    - Unweighted Least Squares/Minimum Residual: "uls", "minres"
        - 원래의 상관행렬과 모델로 재구성된 상관행렬 사이의 잔차(Residual)를 최소화하는 방식
        - 변수들 사이의 공통 분산만을 추출하며, 정규성 가정이 필요 없음
        - 잠재 요인 탐색에 초점
    - Maximum Likelihood: "ml", "mle"
        - 모집단의 상관행렬에서 표본이 관찰될 확률을 최대화하는 최대우도법 기반
        - 엄역한 정규성 가정

5. 회전
5.1 목적
    - 변수 값과 공분산 구조(상관관계)는 그대로 유지한 채, 요인축의 방향만 회전하여 해석을 용이하게 함
    - 즉, 축을 돌려 변수들이 보다 명확히 하나의 요인에 속하도록 하여 단순 구조(simple structure)를 만듦
5.2 방법
5.2.1 직교 회전
    - 인자들의 독립성(직교성)을 유지하는 회전 (varimax 등)
    - 요인 간 관계가 없다고 가정하고, 각 요인을 분리된 축으로 만들어 해석을 단순화함
    - 그러나 현실에서는 요인들이 서로 상관되어 있을 수 있으며, 직교회전은 이를 반영하지 못함
5.2.2 사교 회전
    - 요인 간 상관을 허용하는 회전 (promax, oblimin 등)
    - 실제 인자들이 서로 상관되어 있을 수 있는 현실적 구조를 반영함

6. 데이터 전처리 (참고)
6.1 행(브랜드)별 정규화, L2 normalize

    - dtm 데이터 (행별 정규화 전)
        문서	맛	서비스
        A	  10	10
        B	  20	20
        C	  30	30   

    - 행별 정규화 후
        문서	맛	서비스
        A	0.707	0.707
        B	0.707	0.707
        C	0.707	0.707

    - 문제점: 
        - 맛과 서비스는 같은 방향으로 움직임 (양의 상관관계). 즉 공통된 잠재요인이 존재할 가능성 높음. 
        - 행별 정규화 적용하면 변수들이 함께 커지지는 정보 (공동 변동성)이 사라짐. 즉, 요인분석에 필요한 공통요인구조를 파악할 수 없음

6.2 브랜드별 리뷰수로 나누기

    - case 1: 맛과 서비스 간에 상관관계가 실제로 존재하는 경우
        - dtm 데이터 & 리뷰수
            문서 맛	  서비스  review_count
            A	10	 10	   10
            B	20	 20	   10
            C	30	 30	   10

        - 리뷰수로 나누기
            문서	맛	서비스
            A	   1	1
            B	   2	2
            C	   3	3    

    - case 2: 맛과 서비스 간에 상관관계가 실제로는 없으나 리뷰수 때문에 상관관계가 있는 것처럼 보이는 경우
        - dtm 데이터 & 리뷰수
            문서  맛  서비스  review_count
            A	10	 10	   10
            B	20	 20	   20
            C	30	 30	   30
        
        - 리뷰수로 나누기
            문서	맛	서비스
            A	   1	1
            B	   1	1
            C	   1	1    

    - 리뷰수로 나누는 경우에는, 크기(리뷰 수)에 의해 발생한 왜곡된 상관관계를 제거하고 실제 데이터의 상관구조를 복원하는 역할을 함
    
6.3 열(단어)별 표준화, StandardScaler
    - dtm 데이터
        문서	맛	서비스
        A   	1	100
        B   	2	200
        C   	3	300

    - 열별 표준화 
        문서	맛	서비스
        A	-1.22	-1.22
        B	   0	0
        C	1.22	1.22        

    - dtm에서는 서비스의 값이 훨씬 크므로 분석에서 서비스가 지배할 가능성 있음
    - standardScaler는 상관구조를 유지하면서 변수 간 영향력을 균형을 맞춤 
    - (FactorAnalyzer를 포함한 대부분의 패키지에서 기본으로 적용됨)

'''

import plotly.io as pio
pio.renderers.default = "browser"  # VS Code·Jupyter 환경에 맞게 설정

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
sys.path.append('/Users/carrot/Dropbox/Learning/inflearn/902_textanalytics_class/class20261')
from lib.lib_dtm import lib_filtering_dtm as lfd
reload(lfd)

meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 
PATH_to_save = "/Users/carrot/Dropbox/Learning/inflearn/902_textanalytics_class/class20261/s09_efa/data"


# ────────────────────────────────────────────────────────────
# 1) 조건에 맞는 데이터 추출
# lib/lib_dtm/filtering_dtm_at_brand_level.py

# ────────────────────────────────────────────────────────────
# 2) 데이터 적합성 확인
def kmo_bartlett_test(data_w_meta_cols, apply_div_by_review_count=True, apply_l2=False, apply_stdscaler=True):

    #=================================
    # 데이터 전처리
    #=================================
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    #=================================
    # 데이터 정규화, 표준화
    #=================================
    # KMO & Bartlett test는 상관형렬 기반 테스트여서 내부적으로 열별 표준화 후 테스트가 이루어지기 때문에, 별도의 표준화는 필요없지만 확인용으로 추가함
    
    # 정규화(normalize)로 문서(행) 길이의 차이 제거하여 크기 효과를 제거하고, 
    # 그다음 각 변수를 표준화하여 변수간 스케일의 차이 제거(특정 변수가 분산을 지배하는 현상 방지)

    X_scaled = X.copy() # 표준화 적용하지 않을 경우 

    if apply_div_by_review_count == True: # dtm 데이터를 리뷰수로 보정
        X_scaled = X_scaled.div(df['review_count'], axis=0)

    if apply_l2 == True: # 행별 l2 normalize
        X_scaled = normalize(X_scaled, norm="l2", axis=1)        

    if apply_stdscaler == True: # 단어별 표준화
        X_scaled = StandardScaler().fit_transform(X_scaled) # 각 단어(열) 별로 표준화 (value-mean)/stderror        

    #=================================
    # 요인분석 적합성 검정 (KMO & Bartlett)
    #=================================    
    # KMO (0.6 이상 권장)
    kmo_all, kmo_model = calculate_kmo(X_scaled) # kmo_all: 각 변수별 KMO 값 (각 변수의 다른 모든 변수들과의 상관관계가 공통요인에 의해 설명될 수 있는지에 대한 지표), kmo_model: 전체 데이터의 종합 KMO
    print(f"KMO: {kmo_model:.3f}")
    
    # Bartlett 검정 (p < .05, 요인분석 가능)
    chi2, p = calculate_bartlett_sphericity(X_scaled)
    print(f"Bartlett p값: {p:.5f}")

# ────────────────────────────────────────────────────────────
# 2-1) dtm에서 희소·저분산 단어 제거 (요인분석 적합성이 부족하거나, 수렴하지 않을 경우 적용)
def filtering_data_via_sparse_variability(data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=80):

    #=================================
    # 데이터 전처리 
    #=================================
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    #=================================
    # 1) 희소 단어 제거: 대부분의 문서(브랜드)에서 값이 0인, 즉 등장하지 않는 단어 제거
    #=================================
    # 대부분의 브랜드에서 등장하지 않는 ‘희귀 단어’는 상관·분산이 극히 작아 요인분석, PCA 등에 노이즈만 더할 수 있음
    # 각 열(단어)의 0 비율(= sparsity) 계산 후, 기준값 (예, 95 %) 이상이 0인 열을 삭제
    sparsity = (X == 0).mean(axis=0) # 해당 단어가 0인 행 비율
    X_trim = X.loc[:, sparsity < sparsity_cutoff_val] # 0 비율이 95 % 미만(즉, ≥ 5 % 행에서 등장)인 단어만 남김

    #=================================
    # 2) 분산(variance) 기준: 대부분의 문서(브랜드)에서 값이 거의 동일한 단어, 즉 저분산 단어 제거
    #=================================
    # 값의 변화가 크지 않은 “무난한 단어”는 정보량이 적으므로, 분산 상위 subset으로 차원 축소
    # 열 분산의 기준 백분위수(percentile, 예 80) 값보다 큰 단어만 유지 -> 분산 기준 상위 단어만 선택 (예, 20 %)
    sel = VarianceThreshold(threshold=np.percentile(X_trim.var(), vari_cutoff_percentile)) # cut-off 값 (예, 80->상위 20% 분산)
    X_var_array = sel.fit_transform(X_trim) # 열별 분산 계산 후, 분산 > threshold 인 열만 선택(support)
    # X_var_array.shape
    selected_cols = X_trim.columns[sel.get_support()] # 선택된 단어 리스트
    X_var = pd.DataFrame(
        X_var_array,
        index   = X.index, # 브랜드/행 식별자
        columns = selected_cols # 고분산 단어만
    )
    return pd.concat([df[meta_cols], X_var], axis=1).reset_index() # 기존 메타컬럼 결합하여 반환

# ────────────────────────────────────────────────────────────
# 3) 요인 수(k) 결정
def determine_n_factors(data_w_meta_cols, apply_div_by_review_count=True, apply_l2=False, apply_stdscaler=True):

    #=================================
    # 데이터 전처리 
    #=================================
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    #=================================
    # 데이터 정규화, 표준화
    #=================================
    # 정규화(normalize)로 문서(행) 길이의 차이 제거하여 크기 효과를 제거하고, 
    # 그다음 각 변수를 표준화하여 변수간 스케일의 차이 제거(특정 변수가 분산을 지배하는 현상 방지)

    X_scaled = X.copy() # 표준화 적용하지 않을 경우 

    if apply_div_by_review_count == True: # dtm 데이터를 리뷰수로 보정
        X_scaled = X_scaled.div(df['review_count'], axis=0)

    if apply_l2 == True: # 행별 l2 normalize
        X_scaled = normalize(X_scaled, norm="l2", axis=1)        
        # np.linalg.norm(X, axis=1) # 확인 
        # np.linalg.norm(X_scaled, axis=1) 

    if apply_stdscaler == True: # 단어별 표준화
        X_scaled = StandardScaler().fit_transform(X_scaled) # 각 단어(열) 별로 표준화 (value-mean)/stderror        
        # X.mean(axis=0) # 확인용
        # X_scaled.mean(axis=0)
        # X.var(axis=0)
        # X_scaled.var(axis=0)

    #=================================
    # 고유값 계산
    #=================================
    # rotation=None -> 회전전의 고유값 기준으로 테스트, 회전을 하게되면 각 요인이 설명하는 분산의 분포가 바뀔 수 있음
    # 고유값은 공통분산과 고유분산 구분전
    fa_test = FactorAnalyzer(rotation=None, method='principal')  
    fa_test.fit(X_scaled) 
    eigenvals_raw, _ = fa_test.get_eigenvalues() # 1-D array (len = n_vars), eigenvals_raw는 요인 추출 전 상관행렬로부터 직접 추출. 공통분산과 고유분산 구분전이므로 methed의 영향을 받지 않음
    eigenvals = np.sort(eigenvals_raw)[::-1] # Scree 내림차순 정렬

    # Kaiser 기준(k) : 고유값 > 1 의 갯수
    k_kaiser = int((eigenvals > 1).sum())

    #=================================
    # Scree Plot
    #=================================
    x_seq = np.arange(1, len(eigenvals) + 1) # 1, 2, …, p
    fig = go.Figure()

    # 고유값
    fig.add_trace(
        go.Scatter(
            x = x_seq, y = eigenvals,
            mode = "markers+lines",
            name = "Actual eigenvalues",
            marker=dict(symbol="circle", size=8)
        )
    )
    # Kaiser 기준선 y = 1
    fig.add_hline(
        y = 1,
        line=dict(color="gray", dash="dash"),
        annotation_text="Eigenvalue = 1 (Kaiser)",
        annotation_position="bottom right",
        annotation_font_color="gray"
    )
    # 레이아웃‧축 라벨 등
    fig.update_layout(
        title = dict(
            text = (
                "Scree Plot"
            ),
            xref = "paper", # 그래프 전체를 기준으로 좌표 계산
            x = 0.0,
            y = 0.92,
            font = dict(size=18)
        ),
        xaxis_title = "Component number",
        yaxis_title = "Eigenvalue",
        xaxis = dict(dtick=1),
        template = "plotly_white",
        width  = 750,
        height = 480,
        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # fig.show()
    return k_kaiser, fig

# ────────────────────────────────────────────────────────────
# 4) 요인모델 적합 & 회전
def traing_factor_model(data_w_meta_cols, n_factors, rotation_method='varimax', apply_div_by_review_count=True, apply_l2=False, apply_stdscaler=True, method='uls'):

    #=================================
    # 데이터 전처리 
    #=================================
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    #=================================
    # 데이터 정규화, 표준화
    #=================================
    # 정규화(normalize)로 문서(행) 길이의 차이 제거하여 크기 효과를 제거하고, 
    # 그 다음 각 변수를 표준화하여 변수간 스케일의 차이 제거(특정 변수가 분산을 지배하는 현상 방지)

    X_scaled = X.copy() # 표준화 적용하지 않을 경우 
    
    if apply_div_by_review_count == True: # dtm 데이터를 리뷰수로 보정
        X_scaled = X_scaled.div(df['review_count'], axis=0)

    if apply_l2 == True: # 행별 l2 normalize
        X_scaled = normalize(X_scaled, norm="l2", axis=1)        
        # np.linalg.norm(X, axis=1) # 확인 
        # np.linalg.norm(X_scaled, axis=1) 

    if apply_stdscaler == True: # 단어별 표준화
        X_scaled = StandardScaler().fit_transform(X_scaled) # 각 단어(열) 별로 표준화 (value-mean)/stderror        
        # X.mean(axis=0) # 확인용
        # X_scaled.mean(axis=0)
        # X.var(axis=0)
        # X_scaled.var(axis=0)

    #=================================
    # 모델 학습
    #=================================
    fa = FactorAnalyzer(
            n_factors=n_factors,
            rotation=rotation_method,
            method=method, # 'principal', 'uls', 'minres', 'ml'
        ) 
    fa.fit(X_scaled)

    #=================================
    # 요인적재(factor loadings) 계산 
    #=================================
    #  각 변수(단어)가 숨은 공통요인(latent factor)과 얼마나 강하게 연결돼 있는지를 나타내는 지표(일종의 상관계수)
    factor_loadings = pd.DataFrame(
        fa.loadings_,
        index=X.columns,
        columns=[f"F{i+1}" for i in range(n_factors)]
        ).round(5)

    #=================================
    # 요인점수(Factor Scores) 계산
    #=================================
    # 1) 점수 계산
    factor_scores_arr = fa.transform(X_scaled)  # ndarray (n_brands × n_factors)
    score_cols = [f"F{i+1}_score" for i in range(n_factors)]

    factor_scores = pd.DataFrame(
        factor_scores_arr,
        index=X.index, # 브랜드명
        columns=score_cols
    ).round(5)
    

    # 2) 메타컬럼과 병합
    meta_df = df[meta_cols] # 브랜드 메타 정보
    brand_factor_scores = pd.concat([meta_df, factor_scores], axis=1)
    
    #=================================
    # 결과 파일로 저장
    #=================================
    factor_loadings.reset_index().to_csv(f"{PATH_to_save}/factor_loadings_{rotation_method}.csv", encoding='utf-8-sig', index=False)
    brand_factor_scores.reset_index().to_csv(f"{PATH_to_save}/brand_factor_scores_{rotation_method}.csv", encoding='utf-8-sig', index=False)

    return factor_loadings, brand_factor_scores

# ────────────────────────────────────────────────────────────
# 5) 유의미한 factor loading 추출
def extracting_sig_loadings(loadings, loading_cutoff_value=0.3):

    #=================================
    # loading 절대값이 기준값 이상인 항목만 남기기
    #=================================
    mask_loading = loadings.abs() >= loading_cutoff_value # bool DataFrame (단어×요인)
    loadings_sig = loadings.where(mask_loading) # loading_cutoff_value 미만 이면 NaN
    loadings_sig = loadings_sig.dropna(how='all', axis=0) # 모든 요인에서 NaN인 단어 제거

    #=================================
    # factor별 loading 워드 정리
    #=================================
    # 요인별 상위단어 (factor words) 추출
    def summary_top_words(loadings_sig, top_n=10):
        out = dict()
        # f = loadings_sig.columns[0]
        for f in loadings_sig.columns:  # F1, F2, ...
            s = loadings_sig[f].dropna() # 주어진 factor에 대해 facotr score가 na 인 word 제거
            if s.empty: # 추출된 word가 없으면 통과
                continue
            pos = s[s > 0].sort_values(ascending=False).head(top_n).index.tolist() # factor loading값이 양인 경우 상위 단어 추출
            neg = s[s < 0].sort_values().head(top_n).index.tolist() # factor loading값이 음인 경우 크기 기준 상위 단어 추출
            out[f] = {'pos': pos, 'neg': neg} # 사전형태로 저장
        return out
    factor_words = summary_top_words(loadings_sig, top_n=10)

    #=================================
    # factor_words를 DataFrame으로 변환하기
    #=================================
    records = [
        {
            "factor": f, 
            "pos_loading_words": ";".join(d["pos"]), 
            "neg_loading_words": ";".join(d["neg"]) 
        }
        for f, d in factor_words.items()
    ]
    df_factor_words = pd.DataFrame(records).sort_values("factor").reset_index(drop=True)

    return loadings_sig, df_factor_words

# ────────────────────────────────────────────────────────────
# 6) 선택한 요인들 기준 시각화
def drow_factor_map(brand_factor_scores, x_factor, y_factor, size_col='review_count', color_col='avg_stars'):

    # x, y 축 이름 지정
    x_col = f"{x_factor}_score"
    y_col = f"{y_factor}_score"

    # fig 그리기
    fig = px.scatter(
        brand_factor_scores.reset_index(),
        x=x_col, y=y_col,
        size=size_col, 
        color=color_col,
        color_continuous_scale="Reds", # 연속형 컬러스케일 지정
        hover_data=["name"],
        # text="name"
    )
    fig.update_layout(
        title=f"Brand Factor Map - {x_factor} vs. {y_factor}",
        xaxis_title=x_factor,
        yaxis_title=y_factor,
        template="plotly_white",
        # width=850,
        # height=600
    )
    # fig.show()
    return fig


if __name__ == '__main__':

    '''
    ## 공통: l2 미적용, stdscaler 적용

    ## 리뷰수 나누기 적용 효과 비교
    1-1) tf + 리뷰수로 나누기 미적용 + 요인추출: principal + rotation_method: varimax
    1-1) tf + 리뷰수로 나누기 적용 + 요인추출: principal + rotation_method: varimax
        
    ## 요인추출 방법 영향 비교
    2-1) tf + 리뷰수로 나누기 적용 + 요인추출: principal + rotation_method: varimax
    2-2) tf + 리뷰수로 나누기 적용 + 요인추출: uls + rotation_method: varimax
    2-3) tf + 리뷰수로 나누기 적용 + 요인추출: mle + rotation_method: varimax

    
    ## 회전방법 비교
    3-1) tf + 리뷰수로 나누기 적용 + 요인추출: uls + rotation_method: varimax
    3-2) tf + 리뷰수로 나누기 적용 + 요인추출: uls + rotation_method: oblimin

    3. 브랜드 포지션 해석    
    - oreganospizzabistro

    '''
    #-------------------------------
    ### 전체 브랜드

    ### 0) 분석 조건
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = [], # Italian, Mexican
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    apply_div_by_review_count=True; apply_l2=False; apply_stdscaler=True

    ## 1) 분석할 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

    ## 2) 데이터 적합성 점검
    kmo_bartlett_test(data_w_meta_cols=data_w_meta_cols, apply_div_by_review_count=apply_div_by_review_count, apply_l2=apply_l2, apply_stdscaler=apply_stdscaler)

    # 데이터 적합성이 문제가 있을 경우, dtm 에서 희소·저분산 단어 제거  
    data_w_meta_cols = filtering_data_via_sparse_variability(data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=50)
    kmo_bartlett_test(data_w_meta_cols=data_w_meta_cols)

    ## 3) 요인 수(k) 결정
    k_kaiser, scree_fig = determine_n_factors(data_w_meta_cols=data_w_meta_cols, apply_div_by_review_count=apply_div_by_review_count, apply_l2=apply_l2, apply_stdscaler=apply_stdscaler)
    scree_fig.show()

    ## 4) 요인모델 적합 & 회전
    n_factors = 10   # 위 단계에서 결정한 k 값 # 5, 10
    method = "uls" # 'principal', 'uls'/'minres', 'ml'/'mle'
    rotation_method = 'oblimin' # varimax, promax, oblimin
    factor_loadings, brand_factor_scores = traing_factor_model(
        data_w_meta_cols=data_w_meta_cols, n_factors=n_factors, rotation_method=rotation_method, 
        apply_div_by_review_count=apply_div_by_review_count, apply_l2=apply_l2, apply_stdscaler=apply_stdscaler, method=method
        )

    ## 5) 유의미한 factor loading 추출
    factor_loadings_sig, factor_words = extracting_sig_loadings(loadings=factor_loadings, loading_cutoff_value=0.3)
    factor_words

    ## 6) 선택한 요인들 기준 시각화
    x_factor='F1'
    y_factor='F2'
    fig = drow_factor_map(brand_factor_scores, x_factor, y_factor, size_col='review_count', color_col='avg_stars')
    fig.show()


    #-------------------------------
    ### 특정카테고리 브랜드

    ### 0) 분석 조건
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = ["Italian"], # Italian, Mexican
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    apply_div_by_review_count=True; apply_l2=False; apply_stdscaler=True


    ## 1) 분석할 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

    ## 2) 데이터 적합성 점검
    kmo_bartlett_test(data_w_meta_cols=data_w_meta_cols, apply_div_by_review_count=apply_div_by_review_count, apply_l2=apply_l2, apply_stdscaler=apply_stdscaler)

    # 데이터 적합성이 문제가 있을 경우, dtm 에서 희소·저분산 단어 제거  
    data_w_meta_cols = filtering_data_via_sparse_variability(data_w_meta_cols, sparsity_cutoff_val=0.95, vari_cutoff_percentile=50)
    kmo_bartlett_test(data_w_meta_cols=data_w_meta_cols)

    ## 3) 요인 수(k) 결정
    k_kaiser, scree_fig = determine_n_factors(data_w_meta_cols=data_w_meta_cols, apply_div_by_review_count=apply_div_by_review_count, apply_l2=apply_l2, apply_stdscaler=apply_stdscaler)
    scree_fig.show()

    ## 4) 요인모델 적합 & 회전
    n_factors = 10   # 위 단계에서 결정한 k 값 # 5, 10
    method = "ml" # 'principal', 'uls'/'minres', 'ml'/'mle'
    rotation_method = 'varimax' # varimax, promax, oblimin
    factor_loadings, brand_factor_scores = traing_factor_model(
        data_w_meta_cols=data_w_meta_cols, n_factors=n_factors, rotation_method=rotation_method, 
        apply_div_by_review_count=apply_div_by_review_count, apply_l2=apply_l2, apply_stdscaler=apply_stdscaler, method=method
        )

    ## 5) 유의미한 factor loading 추출
    factor_loadings_sig, factor_words = extracting_sig_loadings(loadings=factor_loadings, loading_cutoff_value=0.3)
    factor_words

    ## 6) 선택한 요인들 기준 시각화
    x_factor='F1'
    y_factor='F2'
    fig = drow_factor_map(brand_factor_scores, x_factor, y_factor, size_col='review_count', color_col='avg_stars')
    fig.show()

