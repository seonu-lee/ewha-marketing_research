'''
마케팅조사론 과제 1
Yelp 데이터셋 탐색적 분석 (EDA)
분석 대상: business, reviews
'''

import pandas as pd
import numpy as np
import scipy.stats as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from statsmodels.tsa.seasonal import STL  # STL 분해 시 활성화

# =============================================================================
# 설정
# =============================================================================
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\datasets\yelp_dataset"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment1\results"

# =============================================================================
# 0. 데이터 불러오기
# =============================================================================
business_raw = pd.read_csv(f"{PATH_to_data}/yelp_business.csv")
reviews_raw  = pd.read_csv(f"{PATH_to_data}/yelp_review.csv")

business = business_raw.copy()
reviews  = reviews_raw.copy()

# =============================================================================
# [1] BUSINESS 데이터 EDA
# =============================================================================

# -------------------------------------------------------------------
# 1-0. 데이터 개요
# -------------------------------------------------------------------
print("=" * 60)
print("[Business] 데이터 개요")
print("=" * 60)
business.info()
print("\n[결측치 수]")
print(business.isna().sum())
print("\n[결측 비율]")
print((business.isna().mean() * 100).round(2).astype(str) + " %")
print("\n[기초통계량]")
print(business.describe())
print("\n[상위 5개 행]")
print(business.head().to_string(index=False))


# 결측치:
# neighborhood 61% 결측 → 동네 단위 분석은 사실상 불가, city/state로 대체해야 함
# 나머지(city, state, latitude, longitude)는 각 1건 — 거의 무시 가능한 수준


# latitude/longitude 범위:
# latitude max = 89.99 (북극 근처?), min = -36.09 (호주/아르헨티나급)
# longitude max = 115.09 (아시아?), min = -142.47
# 명백히 미국/캐나다 외 데이터가 섞여 있음 → state 필터링이 중요함

# review_count 분포:
# 평균 30.1 vs 중앙값 8 → 평균이 중앙값의 3.8배 → 극단적 우편향 확인
# 75분위 = 23, max = 7,361 → 상위 소수가 분포를 크게 왜곡

# stars:
# 평균 3.63, 중앙값 3.5, 25th = 3.0, 75th = 4.5 → 전반적 고평점 편향

# is_open:
# 평균 0.840 → 영업중 비율 84%, 폐업 16%

# -------------------------------------------------------------------
# 1-1. business_id — 중복 확인
# -------------------------------------------------------------------
dup_bid = business[business.duplicated('business_id', keep=False)]
print(f"\n[business_id 중복 행 수] {len(dup_bid)}")

# 중복 0 → 깔끔하게 PK로 쓸 수 있음

# -------------------------------------------------------------------
# 1-2. name — 중복·대소문자·공백 확인 및 cleaning
# -------------------------------------------------------------------
# 대소문자 표기 차이
name_groups_case = business.groupby(business['name'].str.lower())['name'].unique()
case_variants = name_groups_case[name_groups_case.apply(len) > 1]
print(f"\n[name 대소문자 이표기 그룹 수] {len(case_variants)}")
print(case_variants.head(5))

# 공백 차이
business['_name_clean'] = business['name'].str.lower().str.replace(r'\s+', '', regex=True)
name_groups_space = business.groupby('_name_clean')['name'].unique()
space_variants = name_groups_space[name_groups_space.apply(len) > 1]
print(f"\n[name 공백 이표기 그룹 수] {len(space_variants)}")
business.drop(columns=['_name_clean'], inplace=True)

# 대소문자 이표기 622그룹, 공백 이표기 1,004그룹 

# Cleaning
business['name_ori'] = business['name']
business['name'] = (
    business['name']
    .str.lower() # 대소문자 소문자로 통일
    .str.replace('[^a-z0-9]', '', regex=True) # 특수문자·공백 전부 제거
)

# 대소문자 이표기 0 , 공백 이표기 0 → 깔끔하게 name으로 쓸 수 있음 

# Cleaning 후 name 기준으로 중복이 얼마나 생겼는지 확인
print(f"\n[name 중복 행 수] {business['name'].duplicated(keep=False).sum()}") #[name 중복 행 수] 54820
print(f"\n[name 고유 수] {business['name'].nunique()}")   #[name 고유 수] 130817
print(f"\n[체인 브랜드 수] {len(business[business.duplicated('name', keep=False)]['name'].unique())}") #[체인 브랜드 수] 11070

# 전체 174,567개 비즈니스 중 54,820개(31.4%) 가 체인/중복 name
# 고유 name은 130,817개 (174,567 - 130,817 = 43,750개가 중복으로 흡수된 것)
# 실제 체인 브랜드 종류는 11,070개
# 즉 11,070개 브랜드가 평균적으로 (54,820 / 11,070) ≈ 5개 지점씩 보유하고 있다는 뜻

# -------------------------------------------------------------------
# 1-3. neighborhood
# -------------------------------------------------------------------
print(f"\n[neighborhood 결측 수] {business['neighborhood'].isna().sum()}")
print("\n[neighborhood 상위 10개]")
print(business['neighborhood'].value_counts().head(10))

# 결측치 재확인
# 아까 info()에서 Non-Null Count: 68,015, 지금 결측 수 106,552
# 174,567 - 106,552 = 68,015 → 일치. 전체의 61%가 결측임 => 변수로 활용하기 어려움

# 상위 10개 해석
# Westside, Southeast, Eastside 같은 방위 기반 이름들이 보임, Ville-Marie, Plateau-Mont-Royal은 몬트리올(캐나다) 동네. The Strip은 라스베이거스 대로변.
# → 미국·캐나다 데이터가 섞여 있다는 게 neighborhood에서도 보임. state 필터링이 중요함

# => 지역단위 분석이 필요하다면 city나 state로 대체해야 할 듯
# => EDA에서는 state 기준으로 분리해서 분석해보자

# -------------------------------------------------------------------
# 1-4. state — 국가별 분리 (US / CA)
# -------------------------------------------------------------------
us_states = [
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
    'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
    'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
    'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
]
ca_provinces = ['AB','BC','MB','NB','NL','NT','NS','NU','ON','PE','QC','SK','YT']

us_biz = business[business['state'].isin(us_states)]
ca_biz = business[business['state'].isin(ca_provinces)]
print(f"\n[state] 전체: {len(business)}, 미국: {len(us_biz)}, 캐나다: {len(ca_biz)}")
print("\n[state 상위 15개]")
print(business['state'].value_counts().head(15))

# AZ(애리조나) 52,214개로 압도적으로 많음 → 피닉스, 스코츠데일 등 애리조나 주요 도시의 비즈니스가 대량으로 포함된 것으로 추정
# NV(네바다) 33,086개 → 라스베이거스 스트립과 인근 지역의 비즈니스가 대량으로 포함된 것으로 추정

# -------------------------------------------------------------------
# 1-5. stars — 별점 분포 시각화
# -------------------------------------------------------------------
star_count   = business['stars'].value_counts().sort_index()
total_stars  = star_count.sum()
percent      = (star_count / total_stars * 100).round(1)
text_labels  = [f"{c} ({p}%)" for c, p in zip(star_count.values, percent)]

fig_star = go.Figure(go.Bar(
    x=star_count.index.astype(str),
    y=star_count.values,
    text=text_labels,
    textposition='outside',
    marker_color='lightskyblue',
))
fig_star.update_layout(
    title='Business 별점(Stars) 분포',
    xaxis_title='별점',
    yaxis_title='비즈니스 수',
    xaxis=dict(type='category'),
    bargap=0.2,
    template='plotly_white',
)
fig_star.show()
fig_star.write_image(f"{PATH_to_save}/fig_business_stars.png", scale=2)

print("\n[별점 기초통계]")
print(business['stars'].describe())
print(f"최빈값: {business['stars'].mode()[0]}")

# 별점 분포는 고평점에 데이터가 집중되어 있음
# 4.0점이 19.2%(33,492개)로 최빈값이고, 3.5~5.0 구간이 전체의 67.6% 를 차지함 
# 반면 1.0~2.0의 저평점 비율은 10%에 불과해, Yelp 리뷰어들이 만족한 경우에 리뷰를 남기는 경향이 있음을 시사함
# -------------------------------------------------------------------
# 1-6. review_count — 분포 시각화 + 로그변환 비교
# -------------------------------------------------------------------
print("\n[review_count 기초통계]")
print(business['review_count'].describe())
print(f"\n[리뷰수 3000 초과 비즈니스]")
print(business[business['review_count'] > 3000][['name_ori', 'city', 'state']].head(10))

fig_rc = go.Figure(go.Histogram(
    x=business['review_count'],
    nbinsx=100,
    marker_color='mediumturquoise',
))
fig_rc.update_layout(
    title='Business 리뷰 수 분포',
    xaxis_title='리뷰 수',
    yaxis_title='비즈니스 수',
    bargap=0.05,
    template='plotly_white',
)
fig_rc.show()

# 원본 vs log1p 비교
fig_log = make_subplots(rows=1, cols=2,
    subplot_titles=['원본 review_count', 'log1p(review_count)'])
fig_log.add_trace(go.Histogram(x=business['review_count'], nbinsx=60), row=1, col=1)
fig_log.add_trace(go.Histogram(x=np.log1p(business['review_count']), nbinsx=60), row=1, col=2)
fig_log.update_layout(
    title_text='리뷰 수 분포: 원본 vs log1p 변환',
    bargap=0.05, template='plotly_white',
    showlegend=False, width=900, height=400,
)
fig_log.show()
fig_star.write_image(f"{PATH_to_save}/fig_business_stars.png", scale=2)
fig_log.write_image(f"{PATH_to_save}/fig_business_review_count.png", scale=2)

# 원본 분포는 극단적 우편향으로, 대부분의 비즈니스가 소수의 리뷰에 몰려 있고 일부 outlier가 분포를 크게 왜곡함.
# 평균(30.1)이 중앙값(8)의 3.8배에 달하며, 상위 outlier는 최대 7,361개로 하위 75%(23개)와 300배 이상 차이남. 
# log1p 변환 후에도 여전히 우편향이 남아 있어, 리뷰 수 자체가 멱함수(power-law) 분포를 따르는 전형적인 패턴임을 보여줌.
# 리뷰 수 상위 비즈니스는 라스베이거스(NV) 호텔·카지노 복합시설에 집중되어 있는데, 이는 관광객 유입이 많은 특수 상권의 특성을 반영함. 
# 분석 시 이러한 outlier가 브랜드 수준 집계에 미치는 영향을 고려할 필요가 있음

# -------------------------------------------------------------------
# 1-7. 별점 vs 평균 리뷰수 (라인 + 99% 신뢰구간)
# -------------------------------------------------------------------
agg = business.groupby('stars')['review_count'].agg(['mean','std','count']).sort_index()
agg['ci99']  = 2.58 * (agg['std'] / np.sqrt(agg['count']))
agg['lower'] = agg['mean'] - agg['ci99']
agg['upper'] = agg['mean'] + agg['ci99']

fig_agg = go.Figure()
fig_agg.add_trace(go.Scatter(
    x=agg.index, y=agg['mean'],
    mode='lines+markers',
    line=dict(color='royalblue', width=3),
    marker=dict(size=7),
    name='평균 리뷰 수',
))
fig_agg.add_trace(go.Scatter(
    x=agg.index, y=agg['upper'],
    mode='lines', line=dict(width=0),
    showlegend=False, hoverinfo='skip',
))
fig_agg.add_trace(go.Scatter(
    x=agg.index, y=agg['lower'],
    mode='lines', fill='tonexty',
    fillcolor='rgba(65,105,225,0.2)',
    line=dict(width=0), name='99% CI', hoverinfo='skip',
))
fig_agg.update_layout(
    title='별점별 평균 리뷰 수 (+ 99% 신뢰구간)',
    xaxis_title='별점', yaxis_title='평균 리뷰 수',
    template='plotly_white',
)
fig_agg.show()
fig_agg.write_image(f"{PATH_to_save}/fig_business_review_mean_by_stars.png", scale=2)

# 별점 4.0에서 평균 리뷰 수가 약 50개로 정점을 찍으며, 1.0~4.0 구간은 별점이 높아질수록 리뷰 수도 증가하는 양의 관계임. 
# 단, 5.0점에서 평균 리뷰 수가 약 11개로 급락하는데, 이는 5점 만점 비즈니스가 소규모·신규 가게에 집중되어 있어 리뷰 모수 자체가 적기 때문으로 해석됨. 
# 즉 높은 별점 = 많은 리뷰가 아니며, 4점대가 리뷰 참여도와 품질 모두 가장 균형 잡힌 구간임.

# -------------------------------------------------------------------
# 1-8. is_open — 집단 간 차이 검정
# -------------------------------------------------------------------
print("\n[is_open 분포]")
print(business['is_open'].value_counts())
print("\n[영업/폐업 평균 별점·리뷰수]")
print(business.groupby('is_open').agg({'stars':'mean','review_count':'mean'}).round(3))

mask_open   = business['is_open'] == 1
stars_open  = business.loc[mask_open,  'stars']
stars_close = business.loc[~mask_open, 'stars']
rv_open     = business.loc[mask_open,  'review_count']
rv_close    = business.loc[~mask_open, 'review_count']

# 별점 Welch t-test
t, p = st.ttest_ind(stars_open, stars_close, equal_var=False, nan_policy='omit')
print(f"\n[별점] Welch t-test  t={t:.3f}, p={p:.4g}")

# 리뷰수 log 변환 후 Welch t-test
t, p = st.ttest_ind(np.log1p(rv_open), np.log1p(rv_close), equal_var=False, nan_policy='omit')
print(f"[리뷰수 log] Welch t-test  t={t:.3f}, p={p:.4g}")

# 전체 비즈니스의 84%가 영업 중(1), 16%가 폐업(0)임. 
# 영업 중인 비즈니스가 폐업 비즈니스 대비 평균 별점(3.655 vs 3.513)과 평균 리뷰 수(31.7 vs 22.2) 모두 높음. 
# Welch t-test 결과 두 집단 간 차이는 별점(p=1.53e-121)과 리뷰 수(p=4.80e-69) 모두 통계적으로 유의함. 
# 즉 별점이 낮고 리뷰 수가 적은 비즈니스일수록 폐업 가능성이 높은 경향이 있으며, 리뷰 참여도와 평점이 비즈니스 생존과 연관될 수 있음을 시사함.

# -------------------------------------------------------------------
# 1-9. categories — 다중 카테고리 분해
# -------------------------------------------------------------------
print(f"\n[categories 결측 수] {business['categories'].isna().sum()}")
cat_counts = (
    business['categories']
    .fillna('')
    .str.split(';')
    .explode()
    .str.strip()
    .value_counts()
)
print("\n[카테고리 상위 20개]")
print(cat_counts.head(20))

fig_cat = go.Figure(go.Bar(
    x=cat_counts.head(20).values,
    y=cat_counts.head(20).index,
    orientation='h',
    marker_color='lightskyblue',
    text=cat_counts.head(20).values,
    textposition='outside',
))
fig_cat.update_layout(
    title='Business 카테고리 상위 20개',
    xaxis_title='비즈니스 수',
    yaxis=dict(autorange='reversed'),
    template='plotly_white',
    height=600,
)
fig_cat.show()
fig_cat.write_image(f"{PATH_to_save}/fig_business_TOP20_categories.png", scale=2)
cat_counts.to_csv(f"{PATH_to_save}/business_categories.csv", encoding='utf-8-sig')

# Restaurants가 54,618개로 압도적 1위이며, 2위 Shopping(27,971개)의 약 2배에 달함. 
# 상위 20개 중 Restaurants, Food, Sandwiches, Fast Food, American(Traditional), Pizza, Coffee & Tea 등 식음료 관련 카테고리가 7개를 차지해 Yelp 데이터가 외식업 중심으로 구성되어 있음을 확인할 수 있음. 
# 결측치가 0개로 categories는 분석에 바로 활용 가능한 상태임. 
# 본 분석에서 Restaurants를 분석 대상으로 선정한 것은 데이터 규모와 대표성 측면에서 적절한 선택임.

# =============================================================================
# [2] REVIEWS 데이터 EDA
# =============================================================================

# -------------------------------------------------------------------
# 2-0. 데이터 개요
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("[Reviews] 데이터 개요")
print("=" * 60)
reviews.info()
print("\n[결측치 수]")
print(reviews.isna().sum())
print("\n[기초통계량]")
print(reviews.describe())
print("\n[상위 5개 행]")
print(reviews.head().to_string(index=False))

# 총 526만 건의 리뷰 데이터로 결측치는 전혀 없어 분석 바로 활용 가능한 상태임. 
# 별점 평균은 3.73으로 business 데이터(3.63)보다 소폭 높으며, 중앙값 4.0, 75분위 5.0으로 고평점 편향이 리뷰 단위에서도 확인됨.
# useful/funny/cool의 중앙값이 모두 0으로, 대부분의 리뷰는 투표를 받지 못함. 
# 반면 최댓값은 각각 3,364 / 1,481 / 1,105로 일부 리뷰에 투표가 극단적으로 집중되는 멱함수 분포를 따름. 
# 또한 useful과 cool의 min이 -1로 음수값이 존재해 데이터 이상치 여부를 추가 확인할 필요가 있음.

# -------------------------------------------------------------------
# 2-1. user_id — 사용자별 리뷰 수 분포
# -------------------------------------------------------------------
user_rc = reviews['user_id'].value_counts()
print(f"\n[유저 수] {len(user_rc):,}")
print(f"[1인 평균 리뷰 수] {user_rc.mean():.2f}")
print(f"[1인 중앙값 리뷰 수] {user_rc.median():.0f}")

fig_u = go.Figure(go.Histogram(x=user_rc, nbinsx=50, marker_color='plum'))
fig_u.update_layout(
    title='유저별 리뷰 수 분포',
    xaxis_title='리뷰 수', yaxis_title='유저 수',
    bargap=0.1, template='plotly_white',
)
fig_u.show()

fig_u = go.Figure(go.Histogram(
    x=np.log1p(user_rc), 
    nbinsx=50, 
    marker_color='plum'
))
fig_u.update_layout(
    title='유저별 리뷰 수 분포 (log1p 변환)',
    xaxis_title='log1p(리뷰 수)', 
    yaxis_title='유저 수',
    bargap=0.1, 
    template='plotly_white',
)
fig_u.show()
fig_u.write_image(f"{PATH_to_save}/fig_user_review_count_log.png", scale=2)

# 총 132만 명의 유저 중 중앙값이 1로, 절반 이상이 리뷰를 단 1개만 작성한 일회성 리뷰어임. 
# 평균(3.97)이 중앙값(1)의 약 4배에 달해 소수의 헤비 유저가 전체 리뷰 수를 끌어올리는 구조임. 
# log1p 변환 후에도 여전히 우편향이 강하게 남아 있어, 유저별 리뷰 수가 멱함수 분포를 따름을 확인할 수 있음. 
# 텍스트 분석 시 일회성 리뷰어의 노이즈 영향을 고려할 필요가 있음.

# -------------------------------------------------------------------
# 2-2. business_id — 비즈니스별 리뷰 수 분포
# -------------------------------------------------------------------
biz_rc = reviews['business_id'].value_counts()
print(f"\n[비즈니스별 리뷰 수] 평균: {biz_rc.mean():.2f}, 중앙값: {biz_rc.median():.0f}")

fig_b = go.Figure(go.Histogram(x=biz_rc, nbinsx=100, marker_color='mediumseagreen'))
fig_b.update_layout(
    title='비즈니스별 리뷰 수 분포',
    xaxis_title='리뷰 수', yaxis_title='비즈니스 수',
    bargap=0.1, template='plotly_white',
)
fig_b.show()

fig_b = go.Figure(go.Histogram(
    x=np.log1p(biz_rc),
    nbinsx=100,
    marker_color='mediumseagreen'
))
fig_b.update_layout(
    title='비즈니스별 리뷰 수 분포 (log1p 변환)',
    xaxis_title='log1p(리뷰 수)',
    yaxis_title='비즈니스 수',
    bargap=0.1,
    template='plotly_white',
)
fig_b.show()
fig_b.write_image(f"{PATH_to_save}/fig_business_review_count_log.png", scale=2)

# 비즈니스별 리뷰 수도 극단적 우편향 구조로, 평균(30.14)이 중앙값(8)의 약 3.8배에 달함. 
# 대부분의 비즈니스가 소수의 리뷰만 보유하는 반면, 일부 인기 비즈니스에 리뷰가 집중되는 멱함수 분포를 따름. 
# log1p 변환 후에도 우편향이 유지되어 분포의 불균형이 매우 심함을 확인할 수 있음. 
# 유저 분포(중앙값 1)보다는 비즈니스 분포(중앙값 8)가 상대적으로 완만해, 비즈니스 단위 집계가 유저 단위보다 분석에 더 안정적임.
# -------------------------------------------------------------------
# 2-3. stars (리뷰 별점) — 분포
# -------------------------------------------------------------------
star_r = reviews['stars'].value_counts().sort_index()
total_r = star_r.sum()
pct_r   = (star_r / total_r * 100).round(1)
print("\n[리뷰 별점 분포]")
for s, c, p in zip(star_r.index, star_r.values, pct_r):
    print(f"  {s}점: {c:,} ({p}%)")

# 비즈니스 단위 리뷰수 vs 평균 별점 산점도
df_biz_star = (
    reviews.groupby('business_id')
    .agg(review_count=('stars','count'), avg_stars=('stars','mean'))
    .reset_index()
)
corr_biz = df_biz_star[['review_count','avg_stars']].corr().iloc[0,1]
print(f"\n[비즈니스] 리뷰수-평균별점 상관계수: {corr_biz:.4f}")

fig_scatter = go.Figure(go.Scatter(
    x=df_biz_star['review_count'],
    y=df_biz_star['avg_stars'],
    mode='markers',
    marker=dict(size=4, opacity=0.5, color='steelblue'),
))
fig_scatter.update_layout(
    title=f'비즈니스 단위: 리뷰 수 vs 평균 별점 (r={corr_biz:.3f})',
    xaxis_title='리뷰 수', yaxis_title='평균 별점',
    template='plotly_white',
)
fig_scatter.show()
fig_scatter.write_image(f"{PATH_to_save}/fig_business_review_cnt_VS_stars_mean.png", scale=2)

# 리뷰 단위 별점은 5점이 42.8%로 압도적 1위이며, 4~5점 합산이 전체의 66% 를 차지해 고평점 편향이 뚜렷함. 
# 중간 평점보다 극단적 저평점(1점) 혹은 고평점(5점)을 남기는 경향이 있음을 시사함(U자형 분포).
# 비즈니스 단위 리뷰 수와 평균 별점 간 상관계수는 r=0.030으로 사실상 선형 관계가 없음. 
# 산점도에서도 리뷰가 많은 비즈니스의 평균 별점이 3.5~4.5 구간에 수렴하는 경향이 보이는데, 이는 리뷰가 쌓일수록 평균이 중심으로 회귀하는 통계적 현상으로 해석됨.

# -------------------------------------------------------------------
# 2-4. date — 연도별·월별 추이 + STL 준비
# -------------------------------------------------------------------
reviews['date']  = pd.to_datetime(reviews['date'])
reviews['year']  = reviews['date'].dt.year
reviews['month'] = reviews['date'].dt.month

yearly = (
    reviews.groupby('year', as_index=False)
    .agg(review_count=('review_id','count'))
)
print("\n[연도별 리뷰 수]")
print(yearly.to_string(index=False))

fig_yr = go.Figure(go.Scatter(
    x=yearly['year'], y=yearly['review_count'],
    mode='lines+markers',
    line=dict(color='royalblue', width=2),
    marker=dict(size=6),
))
fig_yr.update_xaxes(dtick=1)
fig_yr.update_layout(
    title='연도별 리뷰 수 추이',
    xaxis_title='연도', yaxis_title='리뷰 수',
    template='plotly_white', height=450,
)
fig_yr.show()
fig_yr.write_image(f"{PATH_to_save}/fig_yearly_review_count.png", scale=2)


monthly = (
    reviews.groupby(['year','month'], as_index=False)
    .agg(review_count=('review_id','count'))
)
monthly['year_month'] = pd.to_datetime(monthly[['year','month']].assign(day=1))

fig_mo = go.Figure(go.Scatter(
    x=monthly['year_month'], y=monthly['review_count'],
    mode='lines+markers',
))
fig_mo.update_layout(
    title='월별 리뷰 수 추이',
    xaxis_title='년-월', yaxis_title='리뷰 수',
    template='plotly_white', height=450,
)
fig_mo.show()
fig_mo.write_image(f"{PATH_to_save}/fig_monthly_review_count.png", scale=2)

# 연도별 리뷰 수는 2004년부터 2017년까지 지속적 성장세를 보이며, 특히 2013년 이후 급격한 증가가 나타남. 2017년 기준 연간 약 112만 건으로 정점에 달함.
# 월별 그래프에서는 전반적 우상향 트렌드 위에 계절성 패턴이 겹쳐 있음을 확인할 수 있음. 
# 2014년 이후 월별 변동폭이 커지며 특정 월에 리뷰가 집중되는 경향이 강해짐. 
# 2018년 초에 리뷰 수가 급락하는데, 이는 데이터 수집이 중간에 truncate된 것으로 해석되며 실제 리뷰 감소가 아님. 
# STL 분해를 통해 트렌드·계절성·잔차를 분리하면 보다 정확한 패턴 파악이 가능함

# STL 분해 (계절성 분석) — (Yelp 월별 리뷰 수)

from statsmodels.tsa.seasonal import STL
ts = monthly.set_index('year_month')['review_count'].asfreq('MS').fillna(0)
stl = STL(ts, period=12, robust=True)
result = stl.fit()

fig_stl = go.Figure()
for name, vals in [('Observed', result.observed), ('Trend', result.trend),
                   ('Seasonal', result.seasonal), ('Residual', result.resid)]:
    fig_stl.add_trace(go.Scatter(x=ts.index, y=vals, name=name))
fig_stl.update_layout(
    title='STL 분해 (Yelp 월별 리뷰 수)',
    xaxis=dict(tickformat='%Y-%m', dtick='M12', tickangle=45),
    template='plotly_white', height=600,
)
fig_stl.show()
fig_stl.write_image(f"{PATH_to_save}/fig_STL.png", scale=2)

# Trend: 2004~2017년 동안 리뷰 수가 지속적·가속적으로 증가하는 장기 우상향 추세가 뚜렷함. 플랫폼 성장과 사용자 유입 확대가 주된 원인으로 해석됨.
# Seasonal: 계절성 진폭이 초기에는 거의 없다가 2014년 이후 점차 확대됨. 이는 절대적 리뷰 수가 늘어남에 따라 계절 효과의 절대적 크기도 함께 커지는 구조임. 특정 월(여름·연말)에 리뷰가 집중되는 패턴이 반복됨.
# Residual: 대부분 구간에서 잔차가 0 근처로 안정적이나, 2017년 말에 -60k 수준의 극단적 이상값이 발생함. 이는 데이터 수집 truncation에 의한 것으로, 실제 리뷰 감소가 아닌 데이터 한계로 해석해야 함.

# -------------------------------------------------------------------
# 2-5. text — 리뷰 텍스트 길이 분포
# -------------------------------------------------------------------
reviews['text_length'] = reviews['text'].str.len()
print("\n[텍스트 길이 기초통계]")
print(reviews['text_length'].describe())

fig_txt = go.Figure(go.Histogram(
    x=reviews['text_length'], nbinsx=50,
    marker_color='sandybrown',
))
fig_txt.update_layout(
    title='리뷰 텍스트 길이 분포',
    xaxis_title='문자 수', yaxis_title='리뷰 수',
    bargap=0.05, template='plotly_white',
)
fig_txt.show()
fig_txt.write_image(f"{PATH_to_save}/fig_text_length.png", scale=2)

# 평균 텍스트 길이는 612자이나 중앙값(434자)보다 크게 높아 우편향 분포임. 
# 대부분의 리뷰가 200~500자 구간에 집중되어 있으며, 1,000자 이상의 장문 리뷰는 급격히 감소함.
# 최솟값 1자~최댓값 5,056자로 편차가 매우 크며, 표준편차(572)가 평균(612)에 육박해 분포가 매우 불균일함. 
# 텍스트 분석 시 지나치게 짧은 리뷰(노이즈 가능성)나 극단적으로 긴 리뷰에 대한 필터링을 고려할 필요가 있음.


# -------------------------------------------------------------------
# 2-6. useful / funny / cool — 분포
# -------------------------------------------------------------------
print("\n[useful/funny/cool 기초통계]")
print(reviews[['useful','funny','cool']].describe())

for col, color in [('useful','steelblue'), ('funny','mediumorchid'), ('cool','mediumseagreen')]:
    filtered = reviews[(reviews[col] >= 0) & (reviews[col] <= 100)]
    fig_soc = go.Figure(go.Histogram(
        x=filtered[col],
        xbins=dict(start=0, size=1),
        marker_color=color,
    ))
    fig_soc.update_layout(
        title=f'{col} 분포 (0~100)',
        xaxis_title=f'{col} 횟수',
        yaxis_title='리뷰 수',
        bargap=0.05, template='plotly_white',
    )
    fig_soc.show()
    fig_soc.write_image(f"{PATH_to_save}/fig_{col}_distribution.png", scale=2)

# 세 변수 모두 중앙값이 0으로, 대부분의 리뷰는 투표를 전혀 받지 못함. 
# 75분위 기준으로도 useful만 2표, funny와 cool은 각각 0표·1표에 불과해 투표 참여 자체가 매우 저조함.
# 세 변수 중 useful의 평균(1.39)이 가장 높아 리뷰어들이 funny·cool보다 useful 투표를 가장 적극적으로 활용하는 경향이 있음. 
# 반면 funny는 평균 0.51로 가장 낮아 유머성 리뷰에 대한 반응은 상대적으로 드묾.
# 최솟값에서 useful과 cool이 -1로 음수값이 존재하는데, 이는 데이터 수집 또는 처리 과정의 오류로 보이며 분석 전 해당 이상값 처리가 필요함. 
# 세 변수 모두 극단적 우편향으로 가중치 변수로 활용 시 log 변환을 고려할 필요가 있음.