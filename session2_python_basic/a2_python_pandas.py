'''
Series/ Dataframe

index, column, value
column - 선택, 추가, 삭제
row - 선택, 추가, 삭제
정렬

Series.str
apply

조건 검색
결측치
중복

그룹핑 - groupby, agg, pivot_table
합치기 - concat/merge

문자열 --> 날짜/시간 (pandas Series)

file 불러오기, 저장하기

'''


import pandas as pd
import numpy as np

#=================================
# 실습용 데이터 생성 (DataFrame) 
#=================================
np.random.seed(42) # 랜덤 시드 고정 (재현 가능성)
n_students = 30 # 학생 수

ids = range(1, n_students + 1) # ID 생성
names = [f"Student_{i}" for i in ids] # 이름 생성
region = np.random.choice(["seoul", "busan", "jinju"], size=n_students) # 지역 생성

# 과목별 점수 (50~100)
math = np.random.randint(50, 101, n_students)
lang = np.random.randint(50, 101, n_students)
sci = np.random.randint(50, 101, n_students)

# DataFrame 생성
score_df = pd.DataFrame({
    "id": ids,
    "name": names,
    "region": region,
    "math": math,
    "lang": lang,
    "science": sci
})

score_df

#=================================
# DataFrame - 기초
#=================================
df = score_df.copy()

#--------------------------
# index, column, value 확인
#--------------------------
df.index
df.columns
df.values # array 반환

### 기초 정보
df.info()
df.describe()
df.shape

### index 설정
df = df.set_index('id') # column를 index로 변경하기
df.index.name = '번호' # index 이름 변경

df1 = df.reset_index() # index 데이터를 column으로 변경하기
df2 = df.reset_index(drop=True) # index를 0부터 시작하여 다시 매길때 (기존 index 삭제)

#--------------------------
# column 선택, 추가, 삭제
#--------------------------
df = score_df.copy()
df['math'] # 특정 컬럼 series로 가져오기
df[['math', 'science']] # 여러 개 컬럼 선택
df['enrolled'] = True # 컬럼추가
del df['enrolled'] # 컬럼삭제 - 원본데이터에 즉시 적용됨
df1 = df.drop(["math", "lang"], axis=1, inplace=False) # 컬럼삭제, inplace=True이면 원본데이터에 적용
df2 = df.drop(columns=['math', 'lang'])

### column 타입 변경
df = score_df.copy()
df['math'] = df['math'].astype('float') # 선택 컬럼의 type을 변경함

score_cols = ['math', 'lang', 'science'] # 여러 개 컬럼에 동시 적용
df[score_cols] = df[score_cols].astype('float')

df = score_df.copy()
df1 = df.astype({'lang': 'float', 'id': 'str'}) # 타입 변경할 column들을 사전형태로 표시
df.info()
df1.info()

### column 이름 변경
df = score_df.copy()
df.columns = ['번호', '이름', '지역', '수학', '언어', '과학'] # 이름 변경

df = score_df.copy()
df.rename(columns={'id': '번호'}) # 이름을 변경할 column을 사전형태로 표시

#--------------------------
# row 선택, 추가, 삭제
#--------------------------
df = score_df.copy()
df1 = df.set_index('name')

df1.loc['Student_10'] # 라벨(index 값) 이용하여 행 가져오기
df1.loc["Student_10", 'math'] # "Student_10" 행에서 'math' 컬럼의 값을 가져옴

df1.iloc[9] # index 번호 이용하여 행 가져오기
df1.iloc[9, 1] # 9번째(0부터 시작) 행, 1번째(0부터 시작) 컬럼의 값을 가져옴

df[0:10] # 0에서 9번째 행까지의 데이터 선택


#--------------------------
# df.copy() 역할
#--------------------------
# copy() 를 통해 복사본을 만들어서 조작하여, 원본 데이터프레임은 보존 가능
df = score_df.copy() 

### copy()를 사용하지 않을 경우 발생하는 문제 예시
df2 = df 
print(id(df), id(df2))
df2['science'] = 100
print(df2)
print(df) 

### copy()를 사용하는 경우
df = score_df.copy() # copy() 를 통해 복사본을 만들어서 조작하여, 원본 데이터프레임은 보존 가능

df2 = df.copy()
print(id(df), id(df2))
df2['science'] = 100
print(df2)
print(df) 

#--------------------------
# 정렬
#--------------------------
df = score_df.copy() 

df.sort_values(by='math', ascending=False)
df.sort_values(by=['region', 'math'], ascending=[True, False])


#=================================
# 판다스 문자열 처리 - Series.str 
#=================================

#--------------------------
# comment 생성 (100자 내외 임의의 문장)
#--------------------------
import random

fragments = [
    "오늘 수업은 흥미로웠다.", "내용이 어렵지만 재미있었다.", "선생님의 설명이 도움이 되었다.",
    "조금 더 예제를 보고 싶다.", "수학은 생각보다 점수가 잘 나왔다.", "언어 과목은 더 공부해야겠다.",
    "과학은 실험이 많아 흥미롭다.", "다음 시험을 위해 복습을 열심히 해야겠다.", 
    "친구들과 같이 공부하니 도움이 된다.", "온라인 강의도 병행하면 좋을 것 같다."
]
fragments = [
    "오늘 수업은 흥미로웠다.", "내용이 어렵지만 재미있었다.", "선생님의 설명이 도움이 되었다.",
    "조금 더 예제를 보고 싶다.", "수학은 생각보다 점수가 잘 나왔다.", "언어 과목은 더 공부해야겠다.",
    "과학은 실험이 많아 흥미롭다.", "다음 시험을 위해 복습을 열심히 해야겠다.", 
    "친구들과 같이 공부하니 도움이 된다.", "온라인 강의도 병행하면 좋을 것 같다.",
    "The class was very interesting today.", "I need to review more before the exam.",
    "Studying with friends helps a lot.", "Math is challenging but fun.", 
    "I enjoyed the science experiment session."
]
### comment 생성함수
def make_comment():
    text = ""
    while len(text) < 30:
        text += random.choice(fragments) + " "
    return text

df = score_df.copy() 
df['comment'] = [make_comment() for _ in range(n_students)]

#--------------------------
# Series.str
#--------------------------
# https://pandas.pydata.org/docs/reference/api/pandas.Series.str.html

df['comment'].str.replace(' ', '')
df['comment'].str.replace('수학|과학', 'STEM', regex=True)
df['comment'].str.replace(pat='[^A-Za-z]', repl='', regex=True) # 대소문자만 남기기
df['comment'].str.replace(pat='[^가-힣]', repl='', regex=True) # 한글만 남기기

df[df['comment'].str.startswith('수학')]
df[df['comment'].str.contains('수학')]


#=================================
# apply
#=================================
# apply - dataframe 또는 series의 각 행에 대해 지정한 함수를 적용

### 예) 개별과목점수을 이용하여 새로운 점수를 계산하려고 함
# 방법1
def calc_uni_score(row):
    # print(row) # row 확인용
    uni_score = row['math']*0.7 + row['science']*0.2 + row['lang']*0.1
    return uni_score
df.apply(calc_uni_score, axis=1)
df.apply(lambda row: calc_uni_score(row), axis=1) # 위와 동일
# 방법2
df.apply(lambda row: row['math']*0.7 + row['science']*0.2 + row['lang']*0.1, axis=1) # 위와 동일한 결과 - 간결

# 계산이 복잡한 경우, lamda함수로는 구현이 어려움
def calc_uni_score2(row):
    # print(row) # row 확인용
    uni_score = row['math']*0.7 + row['science']*0.2 + row['lang']*0.1
    if uni_score > 90:
        grade = 'A'
    elif uni_score >80:
        grade = 'B'
    else:
        grade = 'C'
    return grade
df_w_grade = score_df.copy() 
df_w_grade['grade'] = df_w_grade.apply(calc_uni_score2, axis=1)


#=================================
# 조건 검색
#=================================
df = score_df.copy() 

df[df['region']=='jinju']
df[(df['region']=='jinju') & (df['math']>90)]
df[(df['region']=='jinju') | (df['region']=='busan')]
df[df['region'].isin(['jinju', 'busan'])]


#=================================
# 결측치
#=================================
df = score_df.copy() 

### 결측치가 있는 데이터 생성
score_cols = ['math', 'lang', 'science']
df[score_cols] = df[score_cols].mask(df[score_cols] <= 60) # mask(조건) 조건이 True이면, 결측치로 바꿔줌 

### 결측치 확인
df.isnull() # 모든 value에 대해서 missing여부 T/F반환, .isna()와 동일
df.isna()

df.isnull().sum() # df의 각 열별로 값이 없는 셀 갯수 반환
df.isnull().sum(axis=1) # df의 각 행별로 값이 없는 셀 갯수 반환
df.isnull().sum().sum() #df 전체에 걸쳐 값이 없는 셀 갯수 반환

### 결측치 행 삭제
df.dropna() # 결측치를 가진 행을 모두 삭제
df.dropna(subset = ['math']) # 특정 column에 대해 결측치를 가진 행 삭제
df.dropna(how = 'any') # 어느 한 컬럼이라도 na인 행 삭제
df[score_cols].dropna(how = 'all') # 모든 컬럼이 na인 행 삭제

### 결측치 값 대체
df.fillna(0) # 전체에 걸쳐 결측치를 0으로 대체
df.fillna({'math': 0, 'lang': 999}) # column별 변경


#=================================
# 중복
#=================================
df = score_df.copy() 

### 중복 행 존재여부 확인
df['region'].value_counts() # 각 값의 갯수, 중복을 확인할 때도 많이 사용

### 중복 행 확인 및 제거
df.duplicated() # 중복행 여부 확인, 행별 T(중복)/F(중복아님)
df.drop_duplicates(subset= 'region', keep = 'last') # subset기준 중복행 마지막 행만 남김
df.drop_duplicates(subset= 'region', keep = 'first') # subset기준 중복행 첫행만 남김
df.drop_duplicates(subset= 'math', keep = False) # subset기준 중복행 모두 제거


#=================================
# 그룹핑 - groupby, agg, pivot_table
#=================================
### 실습용 데이터 준비
df_w_grade = score_df.copy() 
df_w_grade['grade'] = df_w_grade.apply(calc_uni_score2, axis=1)

#--------------------------
# groupby()
#--------------------------
df_w_grade.groupby('region').mean()

# groupby 내용 확인
for group_each in df_w_grade.groupby('region'):
    print(group_each)

#--------------------------
# agg
#--------------------------
df_w_grade.agg('mean')

#--------------------------
# groupby + agg
#--------------------------
df_agged = df_w_grade.groupby('region').agg({'id': 'count', 'math': 'mean', 'lang': 'median'})
df_agged = df_agged.rename(columns={'id': '숫자', 'math': '수학평균값', 'lang': '언어중간값'})

df_agged = df_w_grade.groupby(['region', 'grade']).agg({'id': 'count', 'math': 'mean', 'lang': 'median'})
df_agged = df_w_grade.groupby(['region', 'grade'], as_index=False).agg({'id': 'count', 'math': 'mean', 'lang': 'median'})

df_w_grade.groupby('region')['grade'].agg(lambda row: row.to_list()) # region 기준 grouping한 후, 각 region의 grade를 리스트로 반환

#--------------------------
# pivot table
#--------------------------
# 지역별 평균 성적 구하기
# pivot_table: index=행, values=계산할 컬럼, aggfunc=집계 방식
pivot1 = pd.pivot_table(
    df_w_grade,
    index="region",
    values=["math", "lang", "science"],
    aggfunc="mean"
)
print(pivot1)

# 동일 기능 groupby+agg 로 구현
grouped1 = df_w_grade.groupby("region")[["math", "lang", "science"]].agg("mean")
print(grouped1)


#=================================
# 합치기 - concat/merge
#=================================
### 실습용 데이터 준비
df1 = score_df.head(20)

df2 = score_df.copy() 
df2['grade'] = df2.apply(calc_uni_score2, axis=1)
df2 = df2[['name', 'grade']].sample(20)


### df와 df_grade 합치기
pd.merge(df1, df2, on='name', how='inner') # name값이 공통으로 있는 행만 가져옴
pd.merge(df1, df2, on='name', how='outer') # name값이 적어도 한군데만 있으면 다 가져옴

pd.merge(df1, df2, on='name', how='left') # 왼쪽 df을 기준으로 함
pd.merge(df1, df2, on='name', how='right') # 오른쪽 df을 기준으로 함


#=================================
# 기타
#=================================
df = score_df.copy() 

### 데이터 타입변환
df['math'].astype('float')
df['id'].astype('str')


#=================================
# 문자열 --> 날짜/시간 (pandas Series)
#=================================
df = score_df.copy() 

# -------------------------------
# 실습용 birth_date 컬럼 생성
# -------------------------------
from datetime import datetime, timedelta
start_date = datetime(2000, 1, 1) # 기준 시작일 (예: 2000년 1월 1일)
birth_dates = [(start_date + timedelta(days=random.randint(0, 365*5))).strftime("%Y--%m--%d") for _ in range(len(df))] # 무작위 날짜 생성 (2000년 ~ 2005년 사이)
df['birth_date'] = birth_dates


### 문자열 날짜/시간 --> datetime 날짜/시간으로 변환
df['birth_date1']= pd.to_datetime(df['birth_date'], format='%Y--%m--%d', errors='raise') # datetime64, format에는 원본데이터에 사용된 날짜형식을 적음
df.info()

# 시간 변수 추출
df['birth_date1'].dt.date # 날짜만 추출 (년도-월-일)
df['birth_date1'].dt.year # 년도
df['birth_date1'].dt.month # 월
df['birth_date1'].dt.day # 일
df['birth_date1'].dt.weekday # 요일(숫자) Monday=0, Sunday=6
df['birth_date1'].dt.dayofweek # 요일(숫자) Monday=0, Sunday=6, weekday와 동일
df['birth_date1'].dt.day_name() # 요일 이름

# 날짜/시간 일부만 원하는대로 지정하여 추출 : 문자열로 추출
df['birth_date1'].dt.strftime('%Y-%m') # 년-월
df['birth_date1'].dt.strftime('%Y_%m_%d %H') # 년-월-일 시(Hour)


#===================================
# file 불러오기, 저장하기
#===================================
df = score_df.copy() 

### datafram데이터 csv 파일로 저장하기
# 인코딩: 문자, 숫자, 기호 등 --> 숫자로 변환하는 규칙
# 데이터 저장, 로딩 시 동일한 인코딩 방식을 사용해야 함

# utf-8 전세계 대부분의 문자를 표현할 수 있는 인코딩 방식
# sig는 엑셀 호환성위해 사용: 'utf-8'로만 하면 엑셀에서 제대로 인지하지 못하여 한글이 깨지는 경우가 있음. sig (signature)를 붙여 utf-8임을 인지할 수 있도록 함

df.to_csv('', encoding='utf-8-sig', index=False) # index=False 인덱스는 저장하지 않음

### csv 파일 dataframe으로 불러오기
df_loaded = pd.read_csv('', encoding='utf-8-sig')

### 폴더생성
import os

PATH_to_save = ""
os.makedirs(PATH_to_save)

