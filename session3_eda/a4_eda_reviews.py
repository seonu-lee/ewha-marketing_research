import pandas as pd
import re
import plotly.graph_objects as go
from collections import Counter
from statsmodels.tsa.seasonal import STL


#=================================
# 설정
#=================================
PATH_to_data = ""
PATH_to_save = ""

#=================================
# 0. 데이터 불러오기
#=================================
# business_raw = pd.read_csv(f"{PATH_to_data}/yelp_business.csv")
reviews_raw = pd.read_csv(f"{PATH_to_data}/yelp_review.csv")
# users_raw = pd.read_csv(f"{PATH_to_data}/yelp_user.csv")
# hours_raw = pd.read_csv(f"{PATH_to_data}/yelp_business_hours.csv")

# business = business_raw.copy()
# users = users_raw.copy()
# hours = hours_raw.copy()
reviews = reviews_raw.copy()

#=================================
# 변수별 eda
#=================================
reviews.info()
reviews

#-------------------------
# user_id
#-------------------------
# 사용자별 리뷰수 분포
user_review_counts = reviews['user_id'].value_counts()

# 히스토그램
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=user_review_counts,
    nbinsx=50
))
fig.update_layout(
    title='유저별 리뷰 수 분포',
    xaxis_title='리뷰 수',
    yaxis_title='유저 수',
    bargap=0.1
)
fig.show()

#-------------------------
# business_id
#-------------------------
# business id 별 리뷰수 분포
business_review_counts = reviews['business_id'].value_counts()

# 히스토그램
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=business_review_counts,
    nbinsx=100  # 구간 수 조정 가능
))
fig.update_layout(
    title='비즈니스별 리뷰 수 분포',
    xaxis_title='리뷰 수',
    yaxis_title='비즈니스 수',
    bargap=0.1
)
fig.show()

#-------------------------
# stars 
#-------------------------

### 리뷰어 별점 분포
star_counts = reviews['stars'].value_counts().sort_index()
df_star_counts = star_counts.reset_index()
df_star_counts.columns = ['stars', 'count']
df_star_counts['prop'] = df_star_counts['count']/df_star_counts['count'].sum()


### 가게 별점 분포: 가게 리뷰수 vs 평균별점
# Q. 맛집평가 - 리뷰수 vs 평균별점?

df_star_perBusiness = reviews.groupby(['business_id']).agg({'business_id': 'count', 'stars': 'mean'}).rename(columns={'business_id': 'review_count', 'stars': 'avg_stars'})
df_star_perBusiness = df_star_perBusiness.reset_index()

# 산점도 그래프 생성
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_star_perBusiness['review_count'],
        y=df_star_perBusiness['avg_stars'],
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.6,
            color='blue'
        ),
    )
)
fig.update_layout(
    title='리뷰 수 vs 평균 별점 (Business 단위)',
    xaxis_title='리뷰 수 (review_count)',
    yaxis_title='평균 별점 (avg_stars)',
    template='plotly_white',
    # height=600,
    # width=800
)
fig.show()

#-------------------------
# date
#-------------------------
### Q 
# seasonality 존재 여부, 나타나는 이유?
# 분석시 샘플 기간은 어떨게 설정해야할까?

### cleaning
reviews['date'] = pd.to_datetime(reviews['date']) # 날짜 컬럼이 datetime 형식이 아닌 경우 변환
reviews['year'] = reviews['date'].dt.year # 연도 추출
reviews['month'] = reviews['date'].dt.month # 월 추출

# 특정 기간의 데이터 추출
reviews_2016 = reviews[(reviews['date']>='2016-01-01') & (reviews['date']<='2016-12-31')]
reviews_2016.sort_values(by='date')

### 년도별 리뷰수
yearly_review_count = reviews.groupby("year", as_index=False).agg({'review_id': 'count'}).rename(columns={'review_id': 'review_count'})

## 라인 그래프 생성
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=yearly_review_count['year'],
    y=yearly_review_count['review_count'],
    mode='lines+markers',
    line=dict(color='royalblue', width=2),
    marker=dict(size=6),
    name='리뷰 수'
))
fig.update_xaxes(
    title='연도',
    # tickmode='linear',  # 모든 년도 표시
    # tick0=yearly_review_count['year'].min(),  # 시작값
    dtick=1  # 1년 단위 간격
)
fig.update_layout(
    title='연도별 리뷰 수 추이',
    xaxis_title='연도',
    yaxis_title='리뷰 수',
    template='plotly_white',
    height=900,
    # width=800
)
fig.show()

### 년-월별 리뷰수
monthly_review_counts = reviews.groupby(['year', 'month'], as_index=False).agg({'review_id': 'count'}).rename(columns={'review_id': 'review_count'}) # 년-월 기준 agg
monthly_review_counts['year_month'] = pd.to_datetime(monthly_review_counts[['year', 'month']].assign(day=1)) # day라는 이름의 컬럼을 추가(값은 1), 이를 년-월-일 datetime 형식으로 변환 (시계열 분석용)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly_review_counts['year_month'],
    y=monthly_review_counts['review_count'],
    mode='lines+markers'
))
fig.update_layout(
    title='월별 리뷰 수 추이',
    xaxis_title='년-월',
    yaxis_title='리뷰 수',
    template='plotly_white',
    height=900,
    # width=800
)
fig.show()

### 리뷰수 trend, seasonality 분해 (STL 라이브러리 적용)
'''
# Seasonal-Trend decomposition using Loess, Loess (Locally Estimated Scatterplot Smoothing)
## 시계열로 변환
monthly_review_ts = monthly_review_counts.set_index('year_month')['review_count']
monthly_review_ts = monthly_review_ts.asfreq('MS')  # 시계열 데이터의 빈도(frequency) 를 명시적으로 'MS'로 설정: 월의 시작일 기준으로 frequency 맞추기 -- 빠진 달의 값은 NaN으로 들어옴
monthly_review_ts = monthly_review_ts.fillna(0) # 빠진 달의 값을 0 으로 변환

## STL 분해 (주기: 12개월)
stl = STL(monthly_review_ts, period=12, robust=True) # 계절 주기(period): 월별 데이터이므로 계절 반복 주기를 12개월로 지정, Outliers에 대해 덜민감하게 설정
result = stl.fit()

## 결과 분리
trend = result.trend
seasonal = result.seasonal
resid = result.resid
observed = result.observed
time_index = monthly_review_ts.index

## 그래프
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_index, y=observed, name='Observed'))
fig.add_trace(go.Scatter(x=time_index, y=trend, name='Trend'))
fig.add_trace(go.Scatter(x=time_index, y=seasonal, name='Seasonal'))
fig.add_trace(go.Scatter(x=time_index, y=resid, name='Residual'))
fig.update_layout(
    title='STL 분해 결과 (Yelp 월별 리뷰 수)',
    xaxis_title='날짜',
    yaxis_title='리뷰 수',
    height=900,
)
fig.update_xaxes(  ## x축 년월 라벨 표시 설정
    tickformat='%Y-%m',  # 라벨 형식: 연-월
    tickmode='linear',     # 또는 'linear'
    dtick='M12',          # 년 단위 간격
    tickangle=45         # 라벨이 겹치지 않도록 기울이기
)
fig.show()
'''

#-------------------------
# text
#-------------------------
### Q 
# 리뷰글자수가 너무 적으면 텍스트 분석의 대상으로 적절한가? 너무 길면?

### 리뷰 길이(글자수)
reviews['text_length'] = reviews['text'].str.len()

# 히스토그램 생성
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=reviews['text_length'],
    nbinsx=50  # 구간 수 조절 가능
))
fig.update_layout(
    title='리뷰 텍스트 길이 분포',
    xaxis_title='텍스트 길이 (문자 수)',
    yaxis_title='리뷰 수',
    bargap=0.05
)
fig.show()

#-------------------------
# userful, funny, cool
#-------------------------
reviews.describe()

### useful 분포
# 0~100 범위로 필터링
filtered = reviews[(reviews['useful'] >= 0) & (reviews['useful'] <= 100)]

# 히스토그램 생성
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=filtered['useful'],
    xbins=dict(
        start=0,
        # end=100,
        size=1
    )
))
fig.update_layout(
    title='useful 수 분포 (0~100, 간격 1)',
    xaxis_title='useful 횟수',
    yaxis_title='리뷰 수',
    bargap=0.05
)
fig.show()



