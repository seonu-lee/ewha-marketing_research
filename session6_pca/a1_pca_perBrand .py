'''
1. 주성분 분석
1.1 개념
    - 서로 연관된 다차원 변수를 더 적은 축(주성분)으로 변환해 차원을 축소하는 통계 기법
    - 원 변수들의 선형 결합으로 새로운 축(주성분)을 만들고, 이 축들은 서로 상관계수가 0(uncorrelated)임.

1.2 원리
    - 각 주성분 축에 문서(관측치)를 투영하면 문서마다 하나의 점수(투영값)를 얻고,
    - PCA는 이 점수들의 분산이 가장 크게 나타나도록 첫 번째 주성분을 선택하고, 
    - 이어서 직교하면서 남은 분산을 최대화하는 축을 차례로 찾음
    - 결과적으로 첫 몇 개 축만으로 전체 변동성의 큰 비율을 보존할 수 있음

1.3 용도
    - 고차원 자료를 2~3축으로 축소하여 시각화
    - 회귀·클러스터링 전에 다중공선성 제거
    - 적은 축으로 분산의 많은 부분을 설명하지만, 차원 축소시 정보의 손실이 발생함

1.4 데이터 구조
    - data * coeff = score
    - coeff: 주성분 계수 (loading), 원본 단어들이 각 주성분에 기여하는 가중치, 즉 어떤 단어가 특정 주성분을 구성하는지 보여줌 
    - score: 주성분 점수, 차원이 축소된 결과물, 즉 각 문서의 새로운 축에서의 좌표값

    - 예)
        data.shape # (5151, 337)
        coeff.shape # (337, 2)
        score.shape # (5151, 2)

1.5 데이터 전처리
1.5.1 열(단어)별 표준화, StandardScaler
    - 단어간 출현빈도, 분산 차이에 따라 고빈도 단어가 주성분을 독식하는 현상을 억제함
    - PCA는 단어들의 선형결합으로 이루어진 가상의 축에 문서 벡터를 투영했을 때, 그 투영값(score)의 분산이 가장 크게 보존되도록 하는 축을 찾기 때문에, 고빈도 단어(예: food, good)는 값과 분산이 커서 표준화를 하지 않으면 이들 소수 단어가 분산을 거의 독식해 주성분 축을 사실상 결정해 버리는 문제가 발생함.
    - 모든 단어(열)을 평균 0, 표준편차 1로 표준화: (value-mean)/stderror
    - 표준화로 모든 단어의 분산을 1로 맞추면, “단어 규모”가 아닌 문서간 “단어 사용 패턴”이 주성분에 반영됨
1.5.2 행(브랜드)별 정규화, L2 normalize 
    - 행 벡터 길이 1 로 조정: v / ‖v‖₂
    - 브랜드간 리뷰 수와 텍스트 길이가 다를 때, 리뷰 수와 텍스트 길이의 영향 없이 “패턴”만 비교함


2. 분석 1: 전체 브랜드 + 전체 단어
2.1 분석 모델
    1) default: 
        - tf + stdscaler 미적용 + l2 미적용
    2) tfdif 효과: 
        - tfidf + stdscaler 미적용 + l2 미적용
    3) stdscaler 효과: 
        - tfidf + stdscaler 적용 + l2 미적용
    4) l2 효과: 
        - tfidf + stdscaler 미적용 + l2 적용
    5) stdscaler + l2 효과
        - tfidf + stdscaler 적용 + l2 적용
    6) l2 + stdscaler 효과
        - tfidf w l2 + stdscaler 적용 + l2 미적용
    7) l2 + stdscaler + l2 효과
        - tfidf w l2 + stdscaler 적용 + l2 적용
2.2 결과 해석
    - 단어벡터 해석, 브랜드 위치 해석
    - 각 모델 결과의 특징? 결과 차이 이유?
    - 사용모델 선택?

3. 분석 2: 선별 브랜드 + 전체 단어
3.1 결과해석
3.2 분석대상 브랜드 카테고리 선별의 장단점

4. 분석 3: 전체 브랜드 + 선별 단어
4.1 결과해석
4.2 키워드 카테고리 선별의 장단점

5. pca 분석 토론
5.1 pca 분석을 통해 얻을 수 있는 insight
    -  소비자들이 브랜드를 평가/리뷰하는 기준
    -  소비자 관점에서 경쟁 브랜드 - 비슷한 포지션을 가진 가계들, 각 포지션의 키워드들
5.2 장점
5.3 한계점


'''

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import sys
from importlib import reload
sys.path.append(r"C:\Users\seonu\Documents\ewha-marketing_research")
from lib.lib_dtm import lib_filtering_dtm as lfd
reload(lfd)

import plotly.io as pio
pio.renderers.default = "vscode"  # 항상 브라우저로 열기 설정

### 조건 설정
PATH_to_data = "C:\\Users\\seonu\\Documents\\ewha-marketing_research\\session4_dtm\\results" 
PATH_to_save = "C:\\Users\\seonu\\Documents\\ewha-marketing_research\\session6_pca\\results"
meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 

# ────────────────────────────────────────────────────────────
# 1) 조건에 맞는 데이터 추출
# lib/lib_dtm/filtering_dtm_at_brand_level.py

# ────────────────────────────────────────────────────────────
# 2) PCA - 각 주성분의 전체 분산 설명력 확인
def pca_scree_plot_perBrand(data_w_meta_cols, apply_stdscaler, apply_l2, input_data_filtering_conditions):

    '''
    Parameters
    ----------
    data_w_meta_cols: 분석할 데이터
    apply_stdscaler: 열(단어)별 표준화 여부
    apply_l2: 행(브랜드)별 정규화
    input_data_filtering_conditions: 분석할 데이터 추출 조건 그래프에 표시용 (분석에는 미사용)    

    Returns
    -------
    fig_scree_plot
    '''

    #=================================
    # 데이터 전처리
    #=================================
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    #=================================
    # 데이터 표준화
    #=================================
    X_scaled = X.copy() # 표준화 적용하지 않을 경우 
    if apply_stdscaler == True:
        X_scaled = StandardScaler().fit_transform(X) # 각 단어(열) 별로 표준화 (value-mean)/stderror
        
        # X.mean(axis=0) # 확인용
        # X_scaled.mean(axis=0)
        # X.var(axis=0)
        # X_scaled.var(axis=0)

    if apply_l2 == True: # 행별 l2 normalize
        X_scaled = normalize(X_scaled, norm="l2", axis=1)
        
        # np.linalg.norm(X, axis=1) # 확인 
        # np.linalg.norm(X_scaled, axis=1) 
    
    #=================================
    # PCA 적용 
    #=================================
    # fit() - PCA 모델 학습만 수행 (주성분 벡터 계산), 반환값 없음
    # fit_transform() - 학습 후 원 데이터를 주성분 공간으로 투영, 투영된 데이터 (PCA score) 반환함
    pca = PCA()
    pca.fit(X_scaled)

    explained_var = pca.explained_variance_ratio_ # 각 주성분의 분산이 전체 분산에서 차지하는 비율
    cumulative_var = np.cumsum(explained_var) # 누적 합계

    #=================================
    # explained_var 시각화
    #=================================    
    # 레이아웃 설정
    subtitle = '<br>'.join([
        f'input_file_name: {input_data_filtering_conditions.get("input_file_name", None)}',
        f'apply_stdscaler: {apply_stdscaler}',
        f'apply_l2_norm: {apply_l2}',
        f'remove_brand_w_word_in_name: {input_data_filtering_conditions.get("remove_brand_w_word_in_name", None)}',
        f'brand_categories_slted: {input_data_filtering_conditions.get("brand_categories_slted", None)}',
        f'words_to_delete: {input_data_filtering_conditions.get("words_to_delete", None)}',
        f'words_to_include_exclusively: {input_data_filtering_conditions.get("words_to_include_exclusively", None)}',
    ])

    fig_scree_plot = go.Figure()
    # 누적 설명력 (라인그래프)
    fig_scree_plot.add_trace(go.Scatter(
        x=list(range(1, len(cumulative_var)+1)), # x값
        y=cumulative_var, # y값
        mode='lines+markers',
        name='누적 설명 분산',
        line=dict(color='blue')
    ))
    # Scree plot (막대그래프)
    fig_scree_plot.add_trace(go.Bar(
        x=list(range(1, len(explained_var)+1)),
        y=explained_var,
        name='개별 설명 분산',
        marker=dict(color='lightgray'),
        opacity=0.7
    ))
    # 기준선 (90%)
    fig_scree_plot.add_shape(
        type='line',
        x0=1, x1=len(cumulative_var),
        y0=0.9, y1=0.9,
        line=dict(color='red', dash='dot'),
    )
    fig_scree_plot.update_layout(
        # title='Scree Plot + 누적 설명 분산 비율 (PCA)',
        title=dict(
            text=f'Scree Plot + 누적 설명 분산 비율 (PCA)<br><span style="font-size:11px">{subtitle}</span>',
            x=0.5,            # 가운데 정렬
            xanchor='center'
        ),
        xaxis_title='주성분 개수',
        yaxis_title='설명력',
        # width=900,
        # height=600,
        xaxis=dict(dtick=1, range=[0, len(explained_var)+1]),
        yaxis=dict(range=[0, 1.05]),
        plot_bgcolor='white',
        margin=dict(t=60)   # 위쪽 여백(제목·자막 공간)
    )
    # fig_scree_plot.show()
    return fig_scree_plot

# ────────────────────────────────────────────────────────────
# 3) PCA - coeff, score 계산
def calculate_pca_coeff_score_perBrand(data_w_meta_cols, apply_stdscaler, apply_l2, num_comp_to_extract):

    '''
    Parameters
    ----------
    data_w_meta_cols: 분석할 데이터
    apply_stdscaler: 열(단어)별 표준화 여부
    apply_l2: 행(브랜드)별 정규화
    num_comp_to_extract: 추출할 주성분 수

    Returns
    -------
    pca_score_w_meta_cols, pca_coeff_df
    '''

    #=================================
    # 데이터 전처리
    #=================================
    df = data_w_meta_cols.copy()  
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.
    X = df.drop(columns=meta_cols) # 메타컬럼을 제거한 데이터

    #=================================
    # 데이터 표준화
    #=================================
    X_scaled = X.copy() # 표준화 적용하지 않을 경우 
    if apply_stdscaler == True:
        X_scaled = StandardScaler().fit_transform(X) # 각 단어(열) 별로 표준화 (value-mean)/stderror
    if apply_l2 == True: # 행별 l2 normalize
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    #=================================
    # pca 실행 - coeff, score 계산
    #=================================
    pca = PCA(n_components=num_comp_to_extract) # ex. 2
    score = pca.fit_transform(X_scaled) # pc 계산 - 원 데이터를 PC 공간으로 투영한 결과, ex. (5151, 2)
    coeff = np.transpose(pca.components_) # loadings - 각 주성분을 정의하는 원래 변수들의 가중치 - 원래 단어들이 각 주성분에 얼마나 기여하는지 (주성분 축 방향), ex. (337, 2)
    pca.explained_variance_ratio_.round(4) # 각 주성분의 분산이 전체 분산에서 차지하는 비율

    print('각 주성분에 의해 설명 가능한 variance 비율(%): ', pca.explained_variance_ratio_.round(4)*100)
    print(f'pca 입력 데이터 구조: {X_scaled.shape} --> pca 결과 score 데이터 구조: {score.shape}, coeff 데이터 구조: {coeff.shape}')

    ### 확인용    
    # X_scaled.shape
    # score.shape
    # coeff.shape
    # pca.components_.shape

    #=================================
    # 결과 DataFrame으로 변환
    #=================================
    pca_score_columns = [f'x{i}' for i in range(num_comp_to_extract)] # pca score 컬럼명칭 정의 (x1, x2, ...)
    pca_df = pd.DataFrame(score, columns=pca_score_columns) # dataframe으로 변환
    pca_score_w_meta_cols = pd.concat([data_w_meta_cols[meta_cols], pca_df], axis=1) # meta 데이터와 pca score 데이터 합치기

    pca_coeff_df = pd.DataFrame(coeff, columns=pca_score_columns) # dataframe으로 변환
    pca_coeff_df['word'] = X.columns # 단어 컬럼 추가
    pca_coeff_df = pca_coeff_df.set_index('word').reset_index() # 단어컬럼 앞으로 이동

    return pca_score_w_meta_cols, pca_coeff_df

# ────────────────────────────────────────────────────────────
# 4) PCA 시각화 - 그래프 그리기
def graph_pca_biplot_perBrand(pca_score_w_meta_cols, pca_coeff_df, num_words_to_display, scale_factor, apply_stdscaler, apply_l2, input_data_filtering_conditions):

    #=================================
    # 데이터 전처리
    #=================================
    pca_df = pca_score_w_meta_cols.copy()
    coeff_df = pca_coeff_df.copy()

    #=================================
    # 설정
    #=================================
    # 1) pca score, coeff 값 scale 차이 보정 
    if scale_factor == "auto":
        # scale_factor = 1
        # scale_factor = np.max(np.array(pca_df[['x0', 'x1']])) / np.max(np.abs(np.array(coeff_df[['x0', 'x1']]))) # max값 기준 scale 설정
        pca_score_scope = np.percentile(np.abs(np.array(pca_df[['x0', 'x1']])), 85) # outliear의 영향을 최소화하기 위해 90%에 해당하는 값을 이용
        coeff_scope = np.percentile(np.abs(np.array(coeff_df[['x0', 'x1']])), 90) # outliear의 영향을 최소화하기 위해 90%에 해당하는 값을 이용
        scale_factor = pca_score_scope / coeff_scope # scale 자동설정

    # 2) 원(브랜드) 사이즈
    # brand_circle_size = 7
    brand_circle_size = 5+pca_df['review_count']*0.01 # 원 크기 - 최초에는 7 로 설정해서 하고, review count로 연동해서 보여줄것

    # 3) 벡터 길이 계산하여 상위 단어 선택 - 2차원을 기준으로함
    coeff_df['loading_strength'] = np.linalg.norm(coeff_df[['x0', 'x1']].values, axis=1) 
    coeff_df = coeff_df.sort_values(by='loading_strength', ascending=False)
    top_coeff_df = coeff_df.head(num_words_to_display) # 벡터길이 기준 상위단어 선택 

    # 4) 그래프에 표시할 데이터 정보
    subtitle = '<br>'.join([
        f'input_file_name: {input_data_filtering_conditions.get("input_file_name", None)}',
        f'apply_stdscaler: {apply_stdscaler}',
        f'apply_l2_norm: {apply_l2}',
        f'remove_brand_w_word_in_name: {input_data_filtering_conditions.get("remove_brand_w_word_in_name", None)}',
        f'brand_categories_slted: {input_data_filtering_conditions.get("brand_categories_slted", None)}',
        f'words_to_delete: {input_data_filtering_conditions.get("words_to_delete", None)}',
        f'words_to_include_exclusively: {input_data_filtering_conditions.get("words_to_include_exclusively", None)}',
    ])

    #=================================
    # 그래프 그리기
    #=================================
    fig_pca_biplot = go.Figure()

    ### 브랜드 위치
    fig_pca_biplot.add_trace(go.Scatter(
        x=pca_df['x0'], y=pca_df['x1'],
        mode='markers',
        marker=dict(
            size=brand_circle_size, # 원 크기 - 최초에는 7 로 설정해서 하고, review count로 연동해서 보여줄것
            color=pca_df['avg_stars'], colorscale='RdBu_r', colorbar=dict(title='Avg Stars')
            ),
        text=pca_df['name'],
        hovertemplate='Brand: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    ### 단어 벡터
    for _, row in top_coeff_df.iterrows():
        x_end = row['x0'] * scale_factor
        y_end = row['x1'] * scale_factor
        fig_pca_biplot.add_trace(go.Scatter(
            x=[0, x_end],
            y=[0, y_end],
            mode='lines+text',
            line=dict(color='black', width=1),
            text=[None, row['word']],
            textposition='top center',
            textfont=dict(size=12),
            hoverinfo='text',
            showlegend=False
        ))
    ### 레이아웃 설정
    fig_pca_biplot.update_layout(
        title=dict(
            text=f'PCA Biplot<br><span style="font-size:11px">{subtitle}</span>',
            x=0.5,            # 가운데 정렬
            xanchor='center'
        ),
        xaxis_title='PC1',
        yaxis_title='PC2',
        # width=950,
        # height=700,
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(t=60)   # 위쪽 여백(제목·자막 공간)
    )
    # fig_pca_biplot.show()
    return fig_pca_biplot

# ────────────────────────────────────────────────────────────
# 5) 주성분별 상위 top_n 단어 추출
def get_top_words_per_pc(pca_coeff_df, top_n=10):

    rows_list = []
    pc_cols = [col for col in pca_coeff_df.columns if col != 'word'] # 'word' 컬럼은 제거

    # pc = pc_cols[1]
    for pc in pc_cols:

        ### pca loading(coeff) 절댓값 기준 상위 top_n 단어 선택 
        top_rows = pca_coeff_df.reindex( # b) 추출된 인덱스 기준 재 배열
            pca_coeff_df[pc].abs().sort_values(ascending=False).index # a) pc 절대값 기준 내람차순 정렬후, 인덱스 추출
        ).head(top_n) # c) 상위 단어 추출

        ### '단어(+/-)' 형식으로 변환
        words_signed = [f"{w}(+)" if loading > 0 else f"{w}(-)" for w, loading in zip(top_rows['word'], top_rows[pc])]
        
        ### 행 딕셔너리 생성: {'pc': pc, 1: word1, 2: word2, ...}
        row_dict = {'pc': pc}
        row_dict.update({i: word for i, word in enumerate(words_signed)}) # dictionary comprehension 이용하여 순위: 단어 사전 만들어, row_dict에 추가
        
        rows_list.append(row_dict)

    top_words_df = pd.DataFrame(rows_list)
    
    return top_words_df


if __name__ == '__main__':

    #=====================================
    # CASE 1 전체 브랜드 + 전체 키워드
    #=====================================

    # 1) tf + stdscaler 미적용 + l2 미적용
    # --> pizza, food, order, time, great 등 일반적인 빈도가 높은 단어들의 loading 값이 높음, 리뷰수가 많은 브랜드의 pca score가 높음
    # 2) tfidf + stdscaler 미적용 + l2 미적용
    # --> pizza, burger 등 tfidf 값이 높은 단어들의 loading 값이 높음. 여전히 리뷰수가 많은 브랜드의 pca score가 높음

    # 3) tfidf + stdscaler 적용 + l2 미적용
    # --> 일부 단어가 높은 loading값을 가지는 현상은 사라짐. 여전히 리뷰가 많은 브랜드의 pca score가 높음
    # 4) tfidf + stdscaler 미적용 + l2 적용
    # --> pizza, taco 등 tfidf 값이 높은 단어들의 loading값이 높음. 리뷰가 많은 브랜드의 pca score가 높게 나오는 현상은 많이 완화됨

    # 5) tfidf + stdscaler 적용 + l2 적용
    # --> 단어 벡터들은 다양한 방향으로 퍼짐. 여전히 리뷰수가 많은 브랜드가 pca score가 높음 (즉, l2의 효과가 약한 것으로 보임)
    # --> NOTE 중요 **stdscaler 를 먼저하면, 리뷰가 많은 브랜드의 단어들의 편차가 크게 되어 일정한 방향을 띠게 됨. 여기에 l2를 적용하여 단어벡터 길이를 1로 고정해도 방향성은 그대로 남게됨.**

    # 6) tfidf w l2 + stdscaler 적용 + l2 미적용
    # --> 특정 단어가 압도하는 현상 사라짐. 리뷰수의 영항도 사라림
    # --> l2에 의해 리뷰수의 영향 완전히 제거하고, stdscaler가 고빈도 단어의 영향을 제거함.

    # 7) tfidf w l2 + stdscaler 적용 + l2 적용
    # --> 키워드 벡터로 더 다양한 방향으로 퍼짐. 리뷰수의 영향도 사라짐


    ### 0) 분석조건
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = [],
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    apply_stdscaler, apply_l2 = True, False

    # 1) 분석할 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

    # 2) scree plot 그래프 - 각 주성분의 전체 분산 설명력 확인
    fig_scree_plot = pca_scree_plot_perBrand(
        data_w_meta_cols=data_w_meta_cols, 
        apply_stdscaler=apply_stdscaler, 
        apply_l2=apply_l2,
        input_data_filtering_conditions=input_data_filtering_conditions # 원 데이터 추출 조건 출력용(데이터분석에는 미사용)
        )
    fig_scree_plot.show()

    # 3) PCA - coeff, score 계산
    pca_score_w_meta_cols, pca_coeff_df = calculate_pca_coeff_score_perBrand(
        data_w_meta_cols=data_w_meta_cols, 
        apply_stdscaler=apply_stdscaler, 
        apply_l2=apply_l2, 
        num_comp_to_extract=2, # 추출할 pca component 수
        )

    # 4) pca_biplot 그래프 
    fig_pca_biplot = graph_pca_biplot_perBrand(
        pca_score_w_meta_cols=pca_score_w_meta_cols,
        pca_coeff_df=pca_coeff_df,
        num_words_to_display=100, # 그래프에 표시할 단어벡터(pca coeff) 수
        scale_factor='auto', # 단어벡터에 적용하는 scale, - pca score와 coeff간의 scale차이 보정
        apply_stdscaler=apply_stdscaler,
        apply_l2=apply_l2,
        input_data_filtering_conditions=input_data_filtering_conditions # 원 데이터 추출 조건 출력용(데이터분석에는 미사용)
        ) 
    fig_pca_biplot.show()
    # fig_pca_biplot.write_image(f'{PATH_to_save}/pca_biplot.svg', width=1000, height=1000) # 파일로 저장할 경우

    # 5) 주성분별 상위 top_n 단어 추출
    top_words_df = get_top_words_per_pc(pca_coeff_df, top_n=10)


    #=====================================
    # CASE 2 분석대상 브랜드 선별 (예, 카테고리 기준 필터링) + 전체 키워드
    #=====================================
    # 1) tf + stdscaler 미적용 + l2 미적용
    # 2) tfidf + stdscaler 미적용 + l2 미적용
    # 3) tfidf + stdscaler 적용 + l2 미적용
    # 4) tfidf + stdscaler 미적용 + l2 적용
    # 5) tfidf + stdscaler 적용 + l2 적용
    # 6) tfidf w l2 + stdscaler 적용 + l2 미적용
    # 7) tfidf w l2 + stdscaler 적용 + l2 적용

    ### 0) 분석조건
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = ['Buffets'], # ['Buffets'],
        words_to_delete = [],
        words_to_include_exclusively = [],
        )
    apply_stdscaler, apply_l2 = True, False


    # 1) 분석할 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

    # 2) scree plot 그래프 - 각 주성분의 전체 분산 설명력 확인
    fig_scree_plot = pca_scree_plot_perBrand(
        data_w_meta_cols=data_w_meta_cols, 
        apply_stdscaler=apply_stdscaler, 
        apply_l2=apply_l2,
        input_data_filtering_conditions=input_data_filtering_conditions # 원 데이터 추출 조건 출력용(데이터분석에는 미사용)
        )
    fig_scree_plot.show()

    # 3) PCA - coeff, score 계산
    pca_score_w_meta_cols, pca_coeff_df = calculate_pca_coeff_score_perBrand(
        data_w_meta_cols=data_w_meta_cols, 
        apply_stdscaler=apply_stdscaler, 
        apply_l2=apply_l2, 
        num_comp_to_extract=2, # 추출할 pca component 수
        )
    # 4) pca_biplot 그래프 
    fig_pca_biplot = graph_pca_biplot_perBrand(
        pca_score_w_meta_cols=pca_score_w_meta_cols,
        pca_coeff_df=pca_coeff_df,
        num_words_to_display=100, # 그래프에 표시할 단어벡터(pca coeff) 수
        scale_factor='auto', # 단어벡터에 적용하는 scale, - pca score와 coeff간의 scale차이 보정
        apply_stdscaler=apply_stdscaler,
        apply_l2=apply_l2,
        input_data_filtering_conditions=input_data_filtering_conditions # 원 데이터 추출 조건 출력용(데이터분석에는 미사용)
        ) 
    fig_pca_biplot.show()

    # 5) 주성분별 상위 top_n 단어 추출
    top_words_df = get_top_words_per_pc(pca_coeff_df, top_n=10)


    #=====================================
    # CASE 3 전체 브랜드 + 선별 키워드 
    #=====================================

    # 1) tf + stdscaler 미적용 + l2 미적용
    # 2) tfidf + stdscaler 미적용 + l2 미적용
    # 3) tfidf + stdscaler 적용 + l2 미적용
    # 4) tfidf + stdscaler 미적용 + l2 적용
    # 5) tfidf + stdscaler 적용 + l2 적용
    # 6) tfidf w l2 + stdscaler 적용 + l2 미적용
    # 7) tfidf w l2 + stdscaler 적용 + l2 적용

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

    ### 0) 분석조건
    words_to_include_exclusively = sorted(list(set(service_format)))
    input_data_filtering_conditions = dict(
        input_file_name = "reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2",
        remove_brand_w_word_in_name = False,
        brand_categories_slted = ['Buffets'],
        words_to_delete = [],
        words_to_include_exclusively = words_to_include_exclusively,
        )
    apply_stdscaler, apply_l2 = True, False

    # 1) 분석할 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

    # 2) scree plot 그래프 - 각 주성분의 전체 분산 설명력 확인
    fig_scree_plot = pca_scree_plot_perBrand(
        data_w_meta_cols=data_w_meta_cols, 
        apply_stdscaler=apply_stdscaler, 
        apply_l2=apply_l2,
        input_data_filtering_conditions=input_data_filtering_conditions # 원 데이터 추출 조건 출력용(데이터분석에는 미사용)
        )
    fig_scree_plot.show()

    # 3) PCA - coeff, score 계산
    pca_score_w_meta_cols, pca_coeff_df = calculate_pca_coeff_score_perBrand(
        data_w_meta_cols=data_w_meta_cols, 
        apply_stdscaler=apply_stdscaler, 
        apply_l2=apply_l2, 
        num_comp_to_extract=2, # 추출할 pca component 수
        )
    # 4) pca_biplot 그래프 
    fig_pca_biplot = graph_pca_biplot_perBrand(
        pca_score_w_meta_cols=pca_score_w_meta_cols,
        pca_coeff_df=pca_coeff_df,
        num_words_to_display=100, # 그래프에 표시할 단어벡터(pca coeff) 수
        scale_factor='auto', # 단어벡터에 적용하는 scale, - pca score와 coeff간의 scale차이 보정
        apply_stdscaler=apply_stdscaler,
        apply_l2=apply_l2,
        input_data_filtering_conditions=input_data_filtering_conditions # 원 데이터 추출 조건 출력용(데이터분석에는 미사용)
        ) 
    fig_pca_biplot.show()
    # fig_pca_biplot.write_image(f'{PATH_to_save}/pca_biplot.svg', width=1000, height=1000) # 파일로 저장할 경우

    # 5) 주성분별 상위 top_n 단어 추출
    top_words_df = get_top_words_per_pc(pca_coeff_df, top_n=10)
