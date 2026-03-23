'''
1. yelp business data
1.1 변수, 데이터타입, 결측치, 기초통계량
2.2 id, name 등 잠재 기준값 변수 - 중복, 대소문자, 공백 등 확인

2. 변수
2.1 business_id 
2.2 name 
2.3 neighborhood
2.4 address, city, state, postal code
2.5 latitude, longitude
2.6 stars
2.7 review_count
2.8 is_open
2.9 categories


'''

import pandas as pd
import numpy as np
import plotly.graph_objects as go #처음써봥 
import scipy.stats as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#=================================
# 설정
#=================================
# pd.set_option('display.max_columns', 20)  # 예: 출력하는 최대 컬럼수 지정
# pd.set_option('display.expand_frame_repr', False)  # 줄바꿈 없이 한 줄로 출력

PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\datasets\yelp_dataset"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\session3_eda\results"

#=================================
# 0. 데이터 불러오기
#=================================
business_raw = pd.read_csv(f"{PATH_to_data}/yelp_business.csv")
# reviews_raw = pd.read_csv(f"{PATH_to_data}/yelp_review.csv")
# users_raw = pd.read_csv(f"{PATH_to_data}/yelp_user.csv")
# hours_raw = pd.read_csv(f"{PATH_to_data}/yelp_business_hours.csv")

business = business_raw.copy()
# reviews = reviews_raw.copy()
# users = users_raw.copy()
# hours = hours_raw.copy()


#=================================
# business 데이터
#=================================
business.info() # data타입, missing이 많은 변수 확인
business.isna().sum() # missing 갯수 확인
business.describe() # 기초통계량
business.head() # tail, sample, head(10)


#-------------------------
# business_id 
#-------------------------
# 중복 확인 - 중복없음 --> id별로 분석하려면 이대로 하면 됨.
business[business.duplicated('business_id', keep=False)] # business_id 기준으로 중복된 행 추출 (keep=False: 모든 중복 행 표시) # 'first' : 첫번쨰 항목은 중복 아님으로 표시
# keep=False → 중복된 행 모두 True (전부 보여줌)
# keep='first' → 첫 번째 행은 False, 나머지만 True
# keep='last' → 마지막 행은 False, 나머지만 True

#-------------------------
# name 
#-------------------------
# name은 중복, 대소문자, 공백 등 오류의 기능성이 높음

### 중복 확인 - 중복 존재 (chain) --> 브랜드 단위에서 분석하려면 agg필요함
duplicate_names = business[business.duplicated('name', keep=False)] 
# duplicate_names = business[business.duplicated('name', keep='first')] 
duplicate_names.sort_values(by="name").head(10) # name 기준으로 중복된 행 추출 후, 이름순으로 정렬하여 상위 10개 출력 (keep='first': 첫 번째 항목은 중복 아님으로 표시)

### name 대소문자 표기 확인 --> 이름은 소문자로 통일 필요
name_groups = business.groupby(business['name'].str.lower())['name'].unique() # 소문자 변환된 name을 기준값으로 그룹화하여, 그룹별로 원래 name 값들의 고유값 리스트 생성
case_variants = name_groups[name_groups.apply(len) > 1]
case_variants.head(10)
# str.lower() 기준으로 그룹화 → 같은 이름인데 표기만 다른 것들 묶기
# unique() → 그룹 내 원래 표기들 수집
# len > 1 → 표기 방식이 2개 이상인 것만 필터링

### name 공백/형식 차이 확인
business['name_clean'] = business['name'].str.lower().str.replace(r'\s+', '', regex=True) # 기준 문자열 만들기 (소문자 + strip + 내부 공백 제거)
name_groups_space = business.groupby('name_clean')['name'].unique() # name_clean 기준으로 다양한 표현 집계
space_variants = name_groups_space[name_groups_space.apply(len) > 1] # 표현이 여러 개인 경우 (공백, 대소문자 등으로 인한 차이)
space_variants.head(10)
# \s+ → 공백 문자(스페이스, 탭 등) 1개 이상
# '' → 빈 문자열로 교체 (= 삭제)

### 이름 cleaning
business = business_raw.copy()
business['name_ori'] = business['name']
business['name'] = business['name'].str.lower() # 소문자화
# business['name'] = business['name'].str.strip() # 앞뒤 공백 제거
# business['name'] = business['name'].str.replace('\s+', '', regex=True) # 모든 공백 제거
business['name'] = business['name'].str.replace('[^a-z0-9]', '', regex=True) # 알파벳/숫자만 남김(특수문자 제거)

#-------------------------
# neighborhood
#-------------------------
# 동네, 결측치 많음
business.info()

business['neighborhood'].unique() # 동네 리스트
business['neighborhood'].value_counts().head(50) # 동네별 가게수
business.groupby('neighborhood')['stars'].mean().sort_values(ascending=False).head(10)
business.groupby('neighborhood').agg({'business_id': 'count', 'stars': 'mean'}).sort_values(by='business_id', ascending=False) # 동네별, 가게수, 평균별점

#-------------------------
# address, city, state, postal code
#-------------------------
### postal code
business['postal_code'].value_counts().head(20) # 우편번호별 business 갯수

### state
set(business['state'].to_list()) # state list
business['state'].value_counts().head(50) # 오타, 영국 지역도 일부 있음

# us
us_state_abbr = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]
us_business = business[business['state'].isin(us_state_abbr)] # state가 us_state_abbr에 포함된 행만 추출
us_business['state'].value_counts()

# canada
ca_province_abbr = [
    'AB', 'BC', 'MB', 'NB', 'NL', 'NT', 'NS',
    'NU', 'ON', 'PE', 'QC', 'SK', 'YT'
]
ca_business = business[business['state'].isin(ca_province_abbr)]
ca_business['state'].value_counts()


#-------------------------
# latitude, longitude
#-------------------------
# **mapbox 이용하여 시각화**
business[['longitude', 'latitude']]


#-------------------------
# stars
#-------------------------
star_count = business['stars'].value_counts().sort_index() # 별점 분포
total_star_count = star_count.sum() # 총 별점 갯수


### 별점 분포 그래프 - x축: 별점, y축: 개수(비지니스 수)
x = star_count.index.astype(str) # 별점
y = star_count.values # 별점별 갯수
percent = (star_count / total_star_count * 100).round(1) # % 계산
text_labels = [f"{count} ({pct}%)" for count, pct in zip(y, percent)] # 텍스트: "개수 (xx%)" 형식

# 그래프 생성
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=x,
        y=y,
        text=text_labels,
        textposition='outside',
        marker_color='lightskyblue',
    )    
)
# 레이아웃 설정
fig.update_layout(
    title='Business Stars Distribution',
    xaxis_title='별점 (Stars)',
    yaxis_title='비즈니스 수',
    xaxis=dict(type='category'),
    bargap=0.2,
    template='plotly_white'
)
fig.show()


#-------------------------
# review_count
#-------------------------
# review count가 너무 적으면 텍스트 분석에 부적할 수 있음. 
business['review_count']

### review count 분포 히스토그램
fig = go.Figure() # Figure 생성
fig.add_trace( # 히스토그램 trace 추가
    go.Histogram(
        x=business['review_count'],
        nbinsx=100, # 히스토그램 막대 개수 조절 (너무 적으면 요약이 부족하고, 너무 많으면 과도하게 나뉨)
        marker_color='mediumturquoise',
        name='리뷰 수'
    )
)
fig.update_layout( # 레이아웃 설정
    title='Business 리뷰 수 분포',
    xaxis_title='리뷰 수 (Review Count)',
    yaxis_title='비즈니스 수',
    bargap=0.05,
    template='plotly_white'
)
fig.show()

# 리뷰수가 매우 높은 샘플 확인
business[business['review_count'] > 3000][['name_ori', 'address', 'city', 'state']] 


### 별점 vs 리뷰수 산점도 ------------------

# 별점 값에 지터 추가
jitter_strength = 0.15   
rng = np.random.default_rng(42) # 재현 가능성용 시드
x_jittered = business['stars'] + rng.uniform(-jitter_strength, jitter_strength, size=len(business))

# 산점도 그래프
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=business['stars'], # x_jittered 
        y=business['review_count'],
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.5,
            color='indianred'
        ),
        name='비즈니스',
        hoverinfo='x+y'
    )
)
fig.update_layout(
    title='별점 vs 리뷰 수 산점도',
    xaxis_title='별점 (Stars)',
    yaxis_title='리뷰 수 (Review Count)',
    template='plotly_white'
)
fig.show()


### 별점 vs 평균 리뷰수 line 그래프 ------------------
## 1. 별점별 리뷰수 평균, 표준편차, 갯수 계산
agg_df = business.groupby('stars')['review_count'].agg(['mean', 'std', 'count']).sort_index()

## 2. 99% 신뢰구간 계산
agg_df['ci95'] = 2.58 * (agg_df['std'] / np.sqrt(agg_df['count']))
agg_df['lower'] = agg_df['mean'] - agg_df['ci95']
agg_df['upper'] = agg_df['mean'] + agg_df['ci95']

## 3. 라인그래프 그리기
fig = go.Figure()

# 평균선
fig.add_trace(
    go.Scatter(
        x=agg_df.index,
        y=agg_df['mean'],
        mode='lines+markers',
        line=dict(color='royalblue', width=3),
        marker=dict(size=7),
        name='별점별 평균 리뷰 수'
    )
)
# 신뢰구간 상한선 (투명 fill을 위해 순서 중요)
fig.add_trace(
    go.Scatter(
        x=agg_df.index,
        y=agg_df['upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    )
)
# 신뢰구간 하한선 + 영역 채우기
fig.add_trace(
    go.Scatter(
        x=agg_df.index,
        y=agg_df['lower'],
        mode='lines',
        fill='tonexty',  # 위 trace와 영역 채우기
        fillcolor='rgba(65, 105, 225, 0.2)',  # royalblue with opacity
        line=dict(width=0),
        name='95% 신뢰구간',
        hoverinfo='skip'
    )
)
# 레이아웃
fig.update_layout(
    title='별점 vs 평균 리뷰수 (+ 99% 신뢰구간)',
    xaxis_title='별점 (Stars)',
    yaxis_title='평균 리뷰 수',
    template='plotly_white'
)
fig.show()


#-------------------------
# is_open
#-------------------------
# 폐업한 가게 포함여부 - 연구범위, 주제에 따라 결정. 경우에 따라, 불포함시 selection bias 가능성
business['is_open'].value_counts() # 생존, 폐업한 가게 수 확인

### 폐업 여부에 따른 평균 별점, 리뷰수 계산
business.groupby('is_open').agg({'stars': 'mean', 'review_count': 'mean'})
# business.groupby('is_open')['stars', 'review_count'].mean() # 위와 동일

### 오픈 여부에 따른 별점 차이 유의도 검증 ---------------------
## 영업/폐업 두 집단 분리
open_mask  = business['is_open'] == 1
closed_mask = ~open_mask # 또는 business['is_open'] == 0

stars_open   = business.loc[open_mask, 'stars']
stars_closed = business.loc[closed_mask, 'stars']

# Student's t-test (정규 and 등분산이라 가정)
t_stat, p_val = st.ttest_ind(stars_open, stars_closed, equal_var=True, nan_policy='omit')
print(f"Student's t-test (stars) t={t_stat:.3f}, p={p_val:.4g}")

# Welch t-test (정규 but 이분산이라 가정)
t_stat, p_val = st.ttest_ind(stars_open, stars_closed, equal_var=False, nan_policy='omit')
print(f"Welch t-test (stars) t={t_stat:.3f}, p={p_val:.4g}")


### 오픈 여부에 따른 리뷰수 차이 유의도 검증 ---------------------
rv_open   = business.loc[open_mask, 'review_count']
rv_closed = business.loc[closed_mask, 'review_count']

# count변수는 한쪽으로 치우쳐 있으므로 로그변환을 통해 완화한 후 t-test 적용
rv_open_log   = np.log1p(rv_open) 
rv_closed_log = np.log1p(rv_closed)

t_stat, p_val = st.ttest_ind(rv_open_log, rv_closed_log, equal_var=False, nan_policy='omit')
print(f"Welch t-test (log review_count) t={t_stat:.3f}, p={p_val:.4g}")


### 참고: 로그 변환 전후 분포 모양변환
# 1) 서브플롯(1×2) 생성 ---------------
fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['원본 review_count', 'log1p(review_count)']
      )
# 왼쪽: 원본
fig.add_trace(
    go.Histogram(
        x=rv_open,
        nbinsx=60, # x 값을 60개 구간으로 나눔
    ),
    row=1, col=1
)
# 오른쪽: 로그변환
fig.add_trace(
    go.Histogram(
        x=rv_open_log,
        nbinsx=60,
    ),
    row=1, col=2
)

# 2) 레이아웃‧축 설정 ---------------
fig.update_layout(
    title_text='리뷰 수 분포 비교: 원본 vs log1p 변환',
    bargap=0.05,
    template='plotly_white',
    showlegend=False,
    width=900, height=400
)

# 3) 레이블링
fig.update_xaxes(title_text='리뷰 수', row=1, col=1)
fig.update_xaxes(title_text='log1p(리뷰 수)', row=1, col=2)
fig.update_yaxes(title_text='빈도', row=1, col=1)
fig.update_yaxes(title_text='빈도', row=1, col=2)

fig.show()


#-------------------------
# categories
#-------------------------
# 식당외 다른 카테고리 중 충분한 데이터가 있는 업종 확인 - 기업수, 리뷰수 확인
business['categories'].isna().sum() # na 갯수 확인

### categories 분포
category_counts = business['categories'].fillna('').str.split(';').explode().str.strip().value_counts().reset_index()
category_counts.to_csv(f"{PATH_to_save}/business_categories.csv", encoding='utf-8-sig', index=False)
