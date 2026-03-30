'''
마케팅조사론 과제 1
Yelp 데이터셋 탐색적 분석 (EDA)
분석 대상: business, reviews
'''

import pandas as pd
import numpy as np
import scipy.stats as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# from statsmodels.tsa.seasonal import STL  # STL 분해 시 활성화

# =============================================================================
# 설정
# =============================================================================
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\datasets\yelp_dataset"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\session3_eda\results"

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

# -------------------------------------------------------------------
# 1-1. business_id — 중복 확인
# -------------------------------------------------------------------
dup_bid = business[business.duplicated('business_id', keep=False)]
print(f"\n[business_id 중복 행 수] {len(dup_bid)}")

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

# Cleaning
business['name_ori'] = business['name']
business['name'] = (
    business['name']
    .str.lower()
    .str.replace('[^a-z0-9]', '', regex=True)
)

# -------------------------------------------------------------------
# 1-3. neighborhood
# -------------------------------------------------------------------
print(f"\n[neighborhood 결측 수] {business['neighborhood'].isna().sum()}")
print("\n[neighborhood 상위 10개]")
print(business['neighborhood'].value_counts().head(10))

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

# -------------------------------------------------------------------
# 1-5. latitude / longitude — Mapbox 위치 시각화
# -------------------------------------------------------------------
# 레스토랑 카테고리 필터 및 지도 시각화
biz_map = business_raw.copy()
biz_map = biz_map[biz_map["categories"].str.contains("Restaurants", na=False)]
biz_map = biz_map[["name", "latitude", "longitude", "stars", "review_count", "is_open"]]

fig_map = px.scatter_mapbox(
    biz_map,
    lat="latitude",
    lon="longitude",
    color="stars",            # 별점에 따라 색상
    size="review_count",      # 리뷰 수로 마커 크기
    size_max=30,
    zoom=3,                   # 전체 분포 확인 (북미 레벨)
    hover_data={
        "name":         True,
        "stars":        True,
        "review_count": True,
        "is_open":      True,
        "latitude":     False,   # hover에 lat/lon 숨기기
        "longitude":    False,
    },
)
fig_map.update_layout(
    title="Yelp Restaurant Businesses — 위치 분포 (별점·리뷰수)",
    mapbox_style="carto-positron",   # Mapbox 토큰 없이 사용 가능
    margin=dict(l=0, r=0, t=50, b=0),
)
fig_map.show(config={"scrollZoom": True, "doubleClick": "reset"})
# fig_map.write_html(f"{PATH_to_save}/fig_business_map.html")  # 인터랙티브 저장

# 전체 카테고리 지도 (비교용)
biz_map_all = business_raw[["name","latitude","longitude","stars","review_count","is_open"]].copy()
fig_map_all = px.scatter_mapbox(
    biz_map_all,
    lat="latitude",
    lon="longitude",
    color="stars",
    size="review_count",
    size_max=20,
    zoom=3,
    hover_data={"name":True,"stars":True,"review_count":True,
                "is_open":True,"latitude":False,"longitude":False},
)
fig_map_all.update_layout(
    title="Yelp All Businesses — 전체 위치 분포",
    mapbox_style="carto-positron",
    margin=dict(l=0, r=0, t=50, b=0),
)
fig_map_all.show(config={"scrollZoom": True, "doubleClick": "reset"})

# -------------------------------------------------------------------
# 1-6. stars — 별점 분포 시각화
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
# fig_star.write_image(f"{PATH_to_save}/fig_business_stars.png", scale=2)

print("\n[별점 기초통계]")
print(business['stars'].describe())
print(f"최빈값: {business['stars'].mode()[0]}")

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
# cat_counts.to_csv(f"{PATH_to_save}/business_categories.csv", encoding='utf-8-sig')


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

# STL 분해 (계절성 분석) — 필요 시 주석 해제
'''
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
'''

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