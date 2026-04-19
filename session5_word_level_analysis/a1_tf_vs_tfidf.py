'''
1. 단어 출현 빈도 분포
    - 빈도별 단어 특성?

2. tf vs tfidf 비교
2.1 출현빈도 상위 단어 분포 비교
    - 순위가 올라간 단어들의 특징?
    - 순위가 내려간 단어들의 특징?

2.2 코사인 유사도 비교
2.2.1 코사인 유사도 개념
    - dtm은 한 문서(브랜드)를 단어차원으로 표현한 벡터 
    - 두 벡터 사이의 각도가 작을수록 방향 즉, 단어 사용 패턴이 비슷함의 나타냄
2.2.2 계산방법
    - a⋅b=∥a∥∥b∥cosθ --> cosθ = a⋅b / ∥a∥∥b∥
2.2.3 값의 범위
    - dtm의 값은 모두 양수이므로 내적도 모두 양수. 
    - 따라서 코사인 유사도 (cosθ) 도 0-1 사이 값을 가짐
2.2.4 tf vs tfidf 코사인 유사도 차이
    - 브랜드들간의 유사도 평균? 
    - 브랜드들간 유사도 표준편차?

2.3. 단어별 브랜드간 변동성 비교
2.3.1 변동성 계산: 분산 이용 (값의 평균과의 차이 제곱의 평균)
2.3.2 브랜드간 변동성이 높은 단어
    - tf 기준 변동성이 높은 단어들? 특징
    - tfidf 기준 변동성이 높은 단어들? 특징
    - tf vs tfidf 기준, 단어들의 변동성 순위가 달라지는 이유?
    - tf 기반 분산: 얼마나 자주 말했나에 민감. 문서량이 큰 브랜드가 부각
    - tfidf 기반 분산: 얼마나 고유하게 말했나에 초점. 브랜드 특화 키워드가 부각

'''

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity

#=================================
# 조건 설정
#=================================
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\session4_dtm\results"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\session5_word_level_analysis\results"

#=================================
# 0. 데이터 불러오기 
#=================================
df_dtm = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm.csv") # dtm 데이터 불러오기
df_dtm_tfidf = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf.csv") # tfidf적용한 dtm 데이터 불러오기
df_dtm_tfidf_l2 = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2.csv") # tfidf적용한 dtm 데이터 불러오기

# df_dtm = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_1.0_0.3_10_dtm.csv") # dtm 데이터 불러오기
# df_dtm_tfidf = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_1.0_0.3_10_dtm_tfidf.csv") # tfidf적용한 dtm 데이터 불러오기
# df_dtm_tfidf_l2 = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_az_perBrand_0.1_1.0_0.3_10_dtm_tfidf_l2.csv") # tfidf적용한 dtm 데이터 불러오기


### 사용할 데이터 선택
df_tf = df_dtm.copy()
df_tfidf = df_dtm_tfidf.copy()
df_tfidf_l2 = df_dtm_tfidf_l2.copy()

### 컬럼 구분 - meta, word 컬럼
meta_cols = ['name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories']
word_cols = [col for col in df_tf.columns if col not in meta_cols]

data_tf = df_tf[word_cols] # 단어 컬럼만 추출
data_tfidf = df_tfidf[word_cols] # 단어 컬럼만 추출
data_tfidf_l2 = df_tfidf_l2[word_cols] # 단어 컬럼만 추출


#=================================
# 1. 단어 출현 빈도
#=================================

### 전체 단어 빈도(단어 총 출현 수) 계산
term_freq = data_tf.sum(axis=0) # 시리즈: index=단어, value=총빈도

term_freq = term_freq.sort_values(ascending=False) # 내림차순 정렬
cumulative = term_freq.cumsum() / term_freq.sum() # 누적 빈도 - 아래 그래프에서 사용함

### 단어 출현 빈도 그래프 ----------------------
## subplot 레이아웃 (가로 2칸)
ranks = np.arange(1, len(term_freq) + 1) # Rank(1부터 시작) 벡터
fig = make_subplots(
    rows=1, cols=2, shared_yaxes=False,
    subplot_titles=("Linear (raw scale)", "log-log Plot")
)

## 선형 스케일
fig.add_trace(
    go.Scatter(
        x=ranks, y=term_freq.values,
        mode="markers", marker=dict(size=4),
        text=term_freq.index,
        customdata=np.column_stack([cumulative]),
        hovertemplate=(
            "Rank %{x}<br>"
            "Freq %{y}<br>"
            "Word %{text}<br>"
            "CumShare %{customdata[0]:.2%}<extra></extra>"
        )
    ),
    row=1, col=1
)
fig.update_xaxes(title_text="Rank", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=1) 

## log–log 스케일
fig.add_trace(
    go.Scatter(
        x=ranks, y=term_freq.values,
        mode="markers", marker=dict(size=4),
        text=term_freq.index,
        customdata=np.column_stack([cumulative]),
        hovertemplate=(
            "Rank %{x}<br>"
            "Freq %{y}<br>"
            "Word %{text}<br>"
            "CumShare %{customdata[0]:.2%}<extra></extra>"
        )
    ),
    row=1, col=2
)
fig.update_xaxes(type="log", title="log Rank", row=1, col=2)
fig.update_yaxes(type="log", title="log Frequency", row=1, col=2)

fig.update_layout(
    width=1100, height=500, template="simple_white",
)
fig.show(renderer="browser")
fig.write_image(f"{PATH_to_save}/word_frequency.svg") # 저장하기


### 빈도 기준 단어 구분 ----------------------
core_threshold = term_freq.quantile(0.9)  # 상위 10%의 단어
content_threshold = term_freq.quantile(0.2)  # 상위 80%의 단어

# 상위 빈도 단어
upper_words = term_freq[term_freq >= core_threshold]

# 중간 빈도 단어 (콘텐츠 단어)
middle_words = term_freq[(term_freq < core_threshold) & (term_freq >= content_threshold)]

# 하위 빈도 단어 (롱테일 단어)
bottom_words = term_freq[term_freq < content_threshold]

# 결과 확인
print(f"상위 빈도 단어: {upper_words.index.to_list()}")
print(f"중간 빈도 단어 : {middle_words.index.to_list()}")
print(f"하위 빈도 단어: {bottom_words.index.to_list()}")


#=================================
# 1.1 tf vs tfidf vs tfidf+l2: 출현 빈도 상위 단어
#=================================
### “상위 k 단어” 구성 비교
k = 50
top_tf    = data_tf.sum().sort_values(ascending=False).head(k)
top_tfidf = data_tfidf.sum().sort_values(ascending=False).head(k)
top_tfidf_l2 = data_tfidf_l2.sum().sort_values(ascending=False).head(k)

comparison = pd.DataFrame({
    'rank_tf'   : top_tf.index,
    'weight_tf' : top_tf.values,
    'rank_tfidf': top_tfidf.index,
    'weight_tfidf': top_tfidf.values,
    'rank_tfidf_l2': top_tfidf_l2.index,
    'weight_tfidf_l2': top_tfidf_l2.values

})
print(comparison)

### 순위 변화 계산
# 1) 각 단어의 순위를 사전형태로 생성 
tf_ranks = {word: rank for rank, word in enumerate(comparison['rank_tf'], start=1)} 
tfidf_ranks = {word: rank for rank, word in enumerate(comparison['rank_tfidf'], start=1)} 
tfidf_l2_ranks = {word: rank for rank, word in enumerate(comparison['rank_tfidf_l2'], start=1)} 

all_words = set(tf_ranks) | set(tfidf_ranks) | set(tfidf_l2_ranks) # 단어 합집합

# 2) tf, tfidf, tfidf_l2 간의 순위 변화계산
changes = []
for w in all_words:
    tf_rank = tf_ranks.get(w)
    tfidf_rank = tfidf_ranks.get(w)
    tfidf_l2_rank = tfidf_l2_ranks.get(w)
    changes.append({
        'word': w,
        'change_tf_to_tfidf': (tf_rank - tfidf_rank) if (tf_rank and tfidf_rank) else None,
        'change_tfidf_to_l2': (tfidf_rank - tfidf_l2_rank) if (tfidf_rank and tfidf_l2_rank) else None
    })
changes_df = pd.DataFrame(changes)

# 3) 순위 변화가 큰 단어들 추출
top_up_tf_to_tfidf = changes_df.sort_values('change_tf_to_tfidf', ascending=False).head(10)
top_down_tf_to_tfidf = changes_df.sort_values('change_tf_to_tfidf').head(10)
top_up_tfidf_to_l2 = changes_df.sort_values('change_tfidf_to_l2', ascending=False).head(10)
top_down_tfidf_to_l2 = changes_df.sort_values('change_tfidf_to_l2').head(10)

#=================================
# 누적 빈도/가중치 곡선 비교
#=================================
# 상위 80%에 포함된 단어 수: tf 173 vs tf-idf 214
# 상위 50개 단어가 전체 토큰에서 차지하는 비중: tf 43.75% vs tf-idf 37.39% 
# 해석 - tf에서의 소수의 단어가 전체 토큰 지배하는 현상이 tf-idf를 통해 완화됨, 공통어 영향이 완화됨

### ── 누적 비율 계산

# 1) 누적 합계 계산
cum_tf = data_tf.sum(axis=0).sort_values(ascending=False).cumsum()
cum_tfidf = data_tfidf.sum(axis=0).sort_values(ascending=False).cumsum()
cum_tfidf_l2 = data_tfidf_l2.sum(axis=0).sort_values(ascending=False).cumsum()

# 2) 누적 비율 계산
cum_tf = cum_tf/cum_tf.iloc[-1] # tf기준 누적비중, .iloc[-1] 맨마지막값 의미
cum_tfidf = cum_tfidf/cum_tfidf.iloc[-1] # tfidf기준 누적비중
cum_tfidf_l2 = cum_tfidf_l2/cum_tfidf_l2.iloc[-1] # tfidf기준 누적비중

ranks = np.arange(1, len(cum_tf)+1) # 그래프 x 축으로 사용

### ── 80 %에 도달하는 랭크
reference_prop_point = 0.8

cut_tf = np.searchsorted(cum_tf.values, reference_prop_point) + 1 # 정렬된 1-차원 배열에서, 주어진 값이 삽입될 인덱스를 찾아줌
cut_tfidf = np.searchsorted(cum_tfidf.values, reference_prop_point) + 1
cut_tfidf_l2 = np.searchsorted(cum_tfidf_l2.values, reference_prop_point) + 1


### ── 그래프
fig = go.Figure()

# 누적 곡선 두 개
fig.add_trace(go.Scatter(x=ranks, y=cum_tf, mode='lines', name='TF'))
fig.add_trace(go.Scatter(x=ranks, y=cum_tfidf, mode='lines',
                         line=dict(color='firebrick'), name='TF-IDF'))
fig.add_trace(go.Scatter(x=ranks, y=cum_tfidf_l2, mode='lines',
                         line=dict(color='blue'), name='TF-IDF+L2'))

# 80 % 수평선
fig.add_hline(y=reference_prop_point, line_dash="dot", line_color="gray", line_width=2, layer="above")

# 두 랭크 수직선
fig.add_vline(x=cut_tf, line_dash="dash", line_color="blue", line_width=2, layer="above")
fig.add_vline(x=cut_tfidf, line_dash="dash", line_color="red", line_width=2, layer="above")
fig.add_vline(x=cut_tfidf_l2, line_dash="dash", line_color="green", line_width=2, layer="above")

# ── 축·레이아웃
fig.update_xaxes(title='Rank')
fig.update_yaxes(title='Cumulative Share', range=[0, 1])
fig.update_layout(
    width=800, height=450, template="simple_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right",  x=1),
    title=(f"누적 비중 비교 ({reference_prop_point*100}%)  |  "
           f"TF: {cut_tf}개 단어 · TF-IDF: {cut_tfidf}개 단어 · TF-IDF+L2: {cut_tfidf_l2}개 단어")
)
fig.show()
fig.write_image(f"{PATH_to_save}/word_cumulative_share.svg")


#=================================
# 1.2 tf vs tfidf vs tfidf+l2: 브랜드 간 단어분포의 코사인 유사도
#=================================

### 문서간 코사인 유사도 행렬
sim_tf = cosine_similarity(data_tf)
sim_tfidf = cosine_similarity(data_tfidf)
sim_tfidf_l2 = cosine_similarity(data_tfidf_l2)

# sim_tf.shape # 확인용

### 유사도 평균, 표준편차 계산
# 중복을 파하기 위해 유사도 행렬의 상삼각행렬에 대해서 적용
upper_idx = np.triu_indices_from(sim_tf, 1) # 상삼각행렬의 행인덱스, 열인덱스; 1은 상삼각행렬 추출할때 주대각선 1칸 위부터 추출함을 의미함

# tf
sim_tf_mean = sim_tf[upper_idx].mean() # 평균
sim_tf_std = sim_tf[upper_idx].std() # 표준편차

# tfidf
sim_tfidf_mean = sim_tfidf[upper_idx].mean()
sim_tfidf_std = sim_tfidf[upper_idx].std()

# tfidf_l2
sim_tfidf_l2_mean = sim_tfidf_l2[upper_idx].mean()
sim_tfidf_l2_std = sim_tfidf_l2[upper_idx].std()

### 평균, 표준편차 출력
print(f"TF 유사도 mean={sim_tf_mean:.3f}, std={sim_tf_std:.3f}")
print(f"TF-IDF 유사도 mean={sim_tfidf_mean:.3f}, std={sim_tfidf_std:.3f}")
print(f"TF-IDF+L2 유사도 mean={sim_tfidf_l2_mean:.3f}, std={sim_tfidf_l2_std:.3f}")


#=================================
# 1.3 tf vs tfidf vs tfidf+l2: 단어별 브랜드 간 변동성 비교
#=================================

### 브랜드별 단어빈도의 분산 계산
var_tf = data_tf.var(axis=0) # 브랜드별 단어빈도(tf)의 분산 계산
var_tfidf = data_tfidf.var(axis=0) # 브랜드별 단어 tfidf의 분산 계산
var_tfidf_l2 = data_tfidf_l2.var(axis=0) # 브랜드별 단어 tfidf+l2의 분산 계산

### 상위 분산 단어 추출
top_var_tf = var_tf.sort_values(ascending=False).head(20)
top_var_tfidf = var_tfidf.sort_values(ascending=False).head(20)
top_var_tfidf_l2 = var_tfidf_l2.sort_values(ascending=False).head(20)

top_var_words = pd.concat([top_var_tf, top_var_tfidf, top_var_tfidf_l2], axis=1, keys=['var_tf','var_tfidf', 'var_tfidf_l2'])

#################################################
# [인사이트] 단어 수준 분석 결과

# ---

# 1. 단어 출현 빈도 분포

# Linear 그래프에서 상위 소수 단어에 빈도가 극단적으로 집중되고, log-log 그래프에서 직선에 가까운 패턴이 나타나 Zipf의 법칙을 따르는 전형적인 멱함수 분포임을 확인할 수 있음.

# 빈도 기준 단어 구분 결과, 상위 빈도 단어(상위 10%)에는 `chicken`, `pizza`, `tabl`, `server` 등 레스토랑 리뷰에서 보편적으로 등장하는 핵심 단어들이 포함됨. 하위 빈도 단어(하위 20%)에는 `japanes`, `hawaiian`, `dumpl` 등 특정 브랜드에만 집중되는 니치 단어들이 포함됨.

# ---

# 2. 누적 비중 비교

# 전체 가중치의 80%를 커버하는 데 필요한 단어 수가 TF 161개 → TF-IDF 190개 → TF-IDF+L2 194개로 증가함. 이는 TF-IDF 적용 후 소수 고빈도 단어의 지배력이 완화되고 단어 간 가중치가 더 균등하게 분산됨을 의미함.

# ---

# 3. 코사인 유사도 비교

# TF 유사도 평균(0.271)이 TF-IDF(0.189)보다 높은데, 이는 TF 기준에서 모든 브랜드가 `chicken`, `pizza` 등 고빈도 공통어를 공유하여 인위적으로 유사하게 보이기 때문임. 
# TF-IDF 적용 후 공통어 영향이 제거되어 브랜드 간 실질적 차별성이 더 잘 반영됨. TF-IDF와 TF-IDF+L2의 평균이 동일(0.189)한 것은 L2 정규화가 방향(각도)은 바꾸지 않고 크기만 조정하기 때문임.

# ---

# 4. 상위 단어 순위 변화

# TF → TF-IDF 전환 시 `sushi`, `burger`, `taco` 등 특정 브랜드에 집중되는 음식명의 순위가 상승하고, `tabl`, `locat`, `minut` 등 서비스/경험 관련 보편어의 순위가 하락함. 이는 TF-IDF가 브랜드 특화 단어를 부각시키는 효과를 잘 보여줌.

