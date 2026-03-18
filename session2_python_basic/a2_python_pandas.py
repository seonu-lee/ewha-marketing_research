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
region = np.random.choice(["seoul", "busan", "jinju"], size=n_students) # 지역 생성 # choice(a, size) a에서 size개 만큼 랜덤하게 선택하여 배열로 반환

# 과목별 점수 (50~100)
math = np.random.randint(50, 101, n_students) # randint(a, b, size) a이상 b미만의 정수 중에서 size개 만큼 랜덤하게 생성
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
df = score_df.copy() # copy() 를 통해 복사본을 만들어서 조작하여, 원본 데이터프레임은 보존 가능

#--------------------------
# index, column, value 확인
#--------------------------
df.index # RangeIndex(start=0, stop=30, step=1) # index는 행의 위치를 나타내는 레이블, 기본적으로 0부터 시작하는 정수로 구성됨
df.columns # Index(['id', 'name', 'region', 'math', 'lang', 'science'], dtype='object') # column은 열의 이름을 나타내는 레이블
df.values # array 반환 # 2차원 배열로 반환, 각 행은 하나의 리스트로 표현됨, 각 열의 값이 리스트에 포함됨 
#행렬은 2차원 배열로 표현되는 데이터 구조로, 행과 열로 구성되어 있습니다. 각 행은 하나의 데이터 레코드를 나타내며, 각 열은 특정 속성이나 변수를 나타냅니다. 예를 들어, 학생들의 점수 데이터를 행렬로 표현하면, 각 행은 한 학생의 점수 정보를 담고 있고, 각 열은 수학, 언어, 과학 등의 과목을 나타낼 수 있습니다. 행렬은 데이터 분석과 머신러닝에서 자주 사용되는 데이터 구조입니다.

### 기초 정보
df.info() # 데이터프레임의 각 열에 대한 정보(데이터 타입, 결측치 여부 등)를 보여줌
df.describe() # 수치형 데이터의 요약 통계량을 보여줌 (평균, 표준편차, 최소값, 25%값, 50%값, 75%값, 최대값)
df.shape # (행의 수, 열의 수) 형태로 데이터프레임의 크기를 반환

### index 설정
df = df.set_index('id') # column를 index로 변경하기 # id 컬럼을 index로 설정하여, 행의 레이블로 사용함
df.index.name = '번호' # index 이름 변경 

df1 = df.reset_index() # index 데이터를 column으로 변경하기
df2 = df.reset_index(drop=True) # 기존 index 삭제하고 index를 0부터 시작하여 다시 매길때 

#--------------------------
# column 선택, 추가, 삭제
#--------------------------
df = score_df.copy()
df['math'] # 특정 컬럼 series로 가져오기 
df[['math', 'science']] # 여러 개 컬럼 선택
df['enrolled'] = True # 컬럼추가
del df['enrolled'] # 컬럼삭제 - 원본데이터에 즉시 적용됨 
df1 = df.drop(["math", "lang"], axis=1, inplace=False) # 컬럼삭제, inplace=True이면 원본데이터에 적용
df2 = df.drop(columns=['math', 'lang']) # 컬럼삭제, columns=삭제할 컬럼명, inplace=False이면 원본데이터에 적용

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
print(df)  # df2에서 'science' 컬럼의 값을 100으로 변경했지만, df에서도 'science' 컬럼의 값이 100으로 변경됨. 이는 df와 df2가 같은 객체를 참조하고 있기 때문임. 따라서 df와 df2는 동일한 데이터프레임을 가리키고 있으며, 하나를 변경하면 다른 하나도 변경되는 결과가 나타남. copy()를 사용하여 df2를 생성하면, df와 df2는 서로 다른 객체가 되어, 하나를 변경해도 다른 하나에는 영향을 미치지 않음.

### copy()를 사용하는 경우
df = score_df.copy() # copy() 를 통해 복사본을 만들어서 조작하여, 원본 데이터프레임은 보존 가능

df2 = df.copy()
print(id(df), id(df2))
df2['science'] = 100
print(df2)
print(df)  # df2에서 'science' 컬럼의 값을 100으로 변경했지만, df에서는 'science' 컬럼의 값이 변경되지 않음. 이는 df와 df2가 서로 다른 객체를 참조하고 있기 때문임. 따라서 df와 df2는 서로 다른 데이터프레임을 가리키고 있으며, 하나를 변경해도 다른 하나에는 영향을 미치지 않음.

#--------------------------
# 정렬
#--------------------------
df = score_df.copy() 

df.sort_values(by='math', ascending=False) # by=정렬할 컬럼명, ascending=True(오름차순), False(내림차순)
df.sort_values(by=['region', 'math'], ascending=[True, False]) # region을 오름차순으로 정렬한 후, region이 같은 경우 math를 내림차순으로 정렬함. by=정렬할 컬럼명 리스트, ascending=각 컬럼별 정렬 방식 리스트


#=================================
# 판다스 문자열 처리 - Series.str # Series.str - 문자열 데이터를 다루는 다양한 메서드를 제공하는 속성
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
    return text # 30자 이상이 될 때까지 fragments에서 무작위로 문장을 선택하여 text에 추가하는 함수

df = score_df.copy() 
df['comment'] = [make_comment() for _ in range(n_students)] # make_comment() 함수를 n_students 만큼 호출하여 comment 컬럼에 저장

#--------------------------
# Series.str
#--------------------------
# https://pandas.pydata.org/docs/reference/api/pandas.Series.str.html

df['comment'].str.replace(' ', '')
df['comment'].str.replace('수학|과학', 'STEM', regex=True) # '수학' 또는 '과학'이 포함된 부분을 'STEM'으로 대체함. regex=True는 패턴이 정규표현식임을 나타냄. '|'는 OR 연산자 역할을 함.
df['comment'].str.replace(pat='[^A-Za-z]', repl='', regex=True) # 대소문자만 남기기 # 정규표현식에서 ^는 부정의 의미, [] 안에 A-Z와 a-z를 제외한 모든 문자를 의미함. 따라서 대소문자만 남기고 나머지 문자는 제거됨.
df['comment'].str.replace(pat='[^가-힣]', repl='', regex=True) # 한글만 남기기 # 정규표현식에서 ^는 부정의 의미, [] 안에 가-힣을 제외한 모든 문자를 의미함. 따라서 한글만 남기고 나머지 문자는 제거됨.

df[df['comment'].str.startswith('수학')] # comment가 '수학'으로 시작하는 행 선택
df[df['comment'].str.contains('수학')]


#=================================
# apply
#=================================
# apply - dataframe 또는 series의 각 행에 대해 지정한 함수를 적용

### 예) 개별과목점수을 이용하여 새로운 점수를 계산하려고 함
# 방법1
def calc_uni_score(row):
    print(row) # row 확인용
    uni_score = row['math']*0.7 + row['science']*0.2 + row['lang']*0.1
    return uni_score # row의 math, science, lang 컬럼의 값을 이용하여 uni_score 계산하는 함수

df.apply(calc_uni_score, axis=1) # df의 각 행(row)에 대해 calc_uni_score 함수를 적용하여 uni_score 계산, axis=1은 행 단위로 함수를 적용하겠다는 의미, axis=0은 열 단위로 함수를 적용하겠다는 의미
df.apply(lambda row: calc_uni_score(row), axis=1) # 위와 동일 # lambda함수로 calc_uni_score 함수를 간결하게 표현, lambda row: calc_uni_score(row) 는 입력값 row에 대해 calc_uni_score 함수를 적용하는 익명 함수(람다 함수)를 정의하는 표현식
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

df[df['region']=='jinju'] # region이 jinju인 행 선택
df[(df['region']=='jinju') & (df['math']>90)] # region이 jinju이고 math 점수가 90보다 큰 행 선택
df[(df['region']=='jinju') | (df['region']=='busan')] # region이 jinju이거나 busan인 행 선택
df[df['region'].isin(['jinju', 'busan'])] # region이 jinju 또는 busan인 행 선택


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
df_w_grade.groupby('region').mean() # region별로 그룹핑한 후, 각 그룹의 수치형 컬럼의 평균값 계산
# 최신 버전의 pandas에서는 숫자가 아닌 컬럼이 섞여 있을 때 mean()을 호출하면 자동으로 걸러주지 않고 에러를 발생시킵니다.
df_w_grade.groupby('region').mean(numeric_only=True) # region별로 그룹핑한 후, 각 그룹의 수치형 컬럼의 평균값 계산, numeric_only=True는 숫자형 컬럼에 대해서만 평균을 계산하겠다는 의미
df_w_grade.groupby('region')['science'].mean() # 'region'으로 묶고, 'science' 컬럼에 대해서만 평균 구하기


# groupby 내용 확인
for group_each in df_w_grade.groupby('region'):
    print(group_each) # groupby('region')으로 묶은 각 그룹에 대해서, 그룹의 이름과 그룹에 속한 데이터프레임이 튜플 형태로 반환됨. 예를 들어, ('seoul', seoul_group_df) 는 'seoul' 그룹의 이름과 'seoul' 그룹에 속한 데이터프레임을 나타냄. 이 반복문을 통해 각 그룹의 이름과 해당 그룹의 데이터를 확인할 수 있음.

#--------------------------
# agg # 여러 개의 집계 함수를 한 번에 적용할 때 사용
#--------------------------
df_w_grade.agg('mean')

#--------------------------
# groupby + agg
#--------------------------
df_agged = df_w_grade.groupby('region').agg({'id': 'count', 'math': 'mean', 'lang': 'median'})
df_agged = df_agged.rename(columns={'id': '숫자', 'math': '수학평균값', 'lang': '언어중간값'})

df_agged = df_w_grade.groupby(['region', 'grade']).agg({'id': 'count', 'math': 'mean', 'lang': 'median'})
df_agged = df_w_grade.groupby(['region', 'grade'], as_index=False).agg({'id': 'count', 'math': 'mean', 'lang': 'median'}) # as_index=False는 그룹핑한 컬럼이 index로 설정되지 않도록 하는 옵션, 그룹핑한 컬럼이 index로 설정되지 않으면, 그룹핑한 컬럼이 일반 컬럼으로 남아있게 되어, 이후에 그룹핑한 컬럼을 기준으로 정렬하거나 다른 연산을 수행할 때 편리함

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
df2 = df2[['name', 'grade']].sample(20) # df2는 name과 grade 컬럼만 가지고 있고, name은 score_df의 name 컬럼에서 무작위로 20개 선택한 값으로 구성되어 있음. grade는 calc_uni_score2 함수를 적용하여 계산된 값임. df1과 df2는 name 컬럼을 기준으로 합칠 수 있음.


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
start_date = datetime(2000, 1, 1) # 기준 시작일 (예: 2000년 1월 1일) # datetime(year, month, day) 특정 날짜를 나타내는 datetime 객체를 생성하는 함수
birth_dates = [(start_date + timedelta(days=random.randint(0, 365*5))).strftime("%Y--%m--%d") for _ in range(len(df))] # 무작위 날짜 생성 (2000년 ~ 2005년 사이)
df['birth_date'] = birth_dates #문자열임 


### 문자열 날짜/시간 --> datetime 날짜/시간으로 변환
df['birth_date1']= pd.to_datetime(df['birth_date'], format='%Y--%m--%d', errors='raise') # datetime64, format에는 원본데이터에 사용된 날짜형식을 적음 # errors='raise'는 변환할 수 없는 값이 있을 때 에러를 발생시키겠다는 의미, errors='coerce'는 변환할 수 없는 값을 NaT로 대체하겠다는 의미, errors='ignore'는 변환할 수 없는 값을 원래 값으로 유지하겠다는 의미
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

df.to_csv('C:\\Users\\seonu\\Documents\\ewha-marketing_research\\session2_python_basic\\result_data\\score_data.csv', encoding='utf-8-sig', index=False) # index=False 인덱스는 저장하지 않음

### csv 파일 dataframe으로 불러오기
df_loaded = pd.read_csv('C:\\Users\\seonu\\Documents\\ewha-marketing_research\\session2_python_basic\\result_data\\score_data.csv', encoding='utf-8-sig')
df_loaded

### 폴더생성
import os

# 1. 만들고 싶은 폴더 이름을 정합니다. (예: 'result_data')
PATH_to_save = "C:\\Users\\seonu\\Documents\\ewha-marketing_research\\session2_python_basic\\result_data"

# 2. 만약 해당 폴더가 이미 있으면 에러가 날 수 있으므로, 
#    'exist_ok=True' 옵션을 넣어주는 것이 좋습니다.
os.makedirs(PATH_to_save, exist_ok=True)

# 3. 이제 아까 작성했던 코드를 이 폴더 안에 저장하도록 연결합니다.

