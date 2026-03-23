'''
users
'''
import pandas as pd
import plotly.graph_objects as go

#=================================
# 설정
#=================================
PATH_to_data = ""
PATH_to_save = ""

#=================================
# 0. 데이터 불러오기
#=================================
# business_raw = pd.read_csv(f"{PATH_to_data}/yelp_business.csv")
# reviews_raw = pd.read_csv(f"{PATH_to_data}/yelp_review.csv")
users_raw = pd.read_csv(f"{PATH_to_data}/yelp_user.csv")
# hours_raw = pd.read_csv(f"{PATH_to_data}/yelp_business_hours.csv")

# business = business_raw.copy()
# reviews = reviews_raw.copy()
users = users_raw.copy()
# hours = hours_raw.copy()

#=================================
# 각 변수 분석
#=================================
users.info()

### 전체 컬럼 목록 확인
users.columns.tolist() # 컬럼 리스트
users.describe() # 숫자형 변수 요약 통계

#-------------------------
# review_count
#-------------------------
users['review_count']

users_slted = users.copy()
# users_slted = users[users['review_count'] <= 1000]

# review count 분포 히스토그램
fig = go.Figure() # Figure 생성
fig.add_trace( # 히스토그램 trace 추가
    go.Histogram(
        x=users_slted['review_count'],
        nbinsx=200, # 히스토그램 막대 개수 조절 (너무 적으면 요약이 부족하고, 너무 많으면 과도하게 나뉨)
        marker_color='mediumturquoise',
        name='리뷰 수'
    )
)
fig.update_layout( # 레이아웃 설정
    title='Users 리뷰 수 분포',
    xaxis_title='리뷰 수 (Review Count)',
    yaxis_title='유저 수',
    bargap=0.05,
    template='plotly_white'
)
fig.show()

#-------------------------
# yelping_since 
#-------------------------
# 가입 시점
users['yelping_since'] = pd.to_datetime(users['yelping_since']) # 데이터타입을 날짜형으로 변환
users['yelping_since'].dt.year.value_counts().sort_index() # 가입년도 분포

#-------------------------
# influence 
#-------------------------
### friends
# 친구 수
users['friend_count'] = users['friends'].fillna('').apply(lambda x: 0 if x=="" else len(x.split(',')))
users['friend_count'].describe() # 친구수 요약

### fans
# 팬 수 상위 사용자
users[['user_id', 'name', 'fans']].sort_values(by='fans', ascending=False).head(20) #  팬 수 상위 유저

### elite
# 엘리트 유저 수 확인
users[users['elite']!="None"]['elite']
users['elite_count'] = users['elite'].fillna('').apply(lambda x: 0 if x == '' else len(x.split(',')))
users['elite_count'].value_counts().sort_index() # 엘리트 유저 분포

#-------------------------
# average_stars
#-------------------------
# 평균 별점 분포
users['average_stars'].value_counts(bins=10, sort=False) # 평균별점 분포

users[['user_id', 'review_count', 'average_stars']].sort_values(by='review_count', ascending=False).head(30)

#=================================
# 변수들간의 상관관계
#=================================
numeric_cols = ['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars'] # 수치형 변수 목록
df_corr = users[numeric_cols].dropna() # 결측치 제거

### 상관계수
corr_matrix = df_corr.corr() # 상관계수 계산

### 히트맵 생성
fig = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title='Correlation')
    )
)
fig.update_layout(
    title='Correlation Matrix of Yelp User Variables',
    xaxis_title='Variables',
    yaxis_title='Variables',
    width=700,
    height=700,
    template='plotly_white'
)
fig.show()

