'''
yelp business hours 데이터

'''
import pandas as pd
import numpy as np
from datetime import datetime

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
# users_raw = pd.read_csv(f"{PATH_to_data}/yelp_user.csv")
hours_raw = pd.read_csv(f"{PATH_to_data}/yelp_business_hours.csv")

# business = business_raw.copy()
# reviews = reviews_raw.copy()
# users = users_raw.copy()
df_hours = hours_raw.copy()

#=================================
# 전처리
#=================================
df_hours.info()

# 시간 컬럼에 있는 비정상적인 문자열 필터링 함수
def is_valid_time_string(x):
    return isinstance(x, str) and '-' in x and x.lower() not in ['none', 'nan', 'null', '']

# 시간 문자열을 time 객체로 변환하는 함수
def parse_time(t):
    try:
        return datetime.strptime(t, "%H:%M").time()
    except Exception as e:
        print(e)
        return np.nan

# 요일 리스트
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

# 오픈/클로즈 컬럼 분리
for day in days:
    df_hours[f"{day}_open"] = df_hours[day].apply(
        lambda x: parse_time(x.split('-')[0]) if is_valid_time_string(x) else np.nan
    )
    df_hours[f"{day}_close"] = df_hours[day].apply(
        lambda x: parse_time(x.split('-')[1]) if is_valid_time_string(x) else np.nan
    )

df_hours.columns.to_list()
columns_to_keep = ['business_id', 'monday_open', 'monday_close', 'tuesday_open', 'tuesday_close', 'wednesday_open', 'wednesday_close', 'thursday_open', 'thursday_close', 'friday_open', 'friday_close', 'saturday_open', 'saturday_close', 'sunday_open', 'sunday_close']

df_hours = df_hours[columns_to_keep]

#=================================
# 모든 요일에 영업하지 않는 경우 확인 및 제거
#=================================
# 처음에는 이부분 skip하고 해볼것 -- 분석결과를 통해 문제점 발견

days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

### 모든 요일에 영업하지 않는 비율 (페업으로 추정)
open_cols = [f'{day}_open' for day in days] # 요일 open 컬럼 리스트
all_closed_mask = df_hours[open_cols].isna().all(axis=1) # 모든 요일의 open 값이 NaN인 행만 선택

closed_all_days_ratio = all_closed_mask.sum() / len(df_hours) # 비율 계산
print(f"모든 요일에 영업시간 정보가 없는 가게 비율: {closed_all_days_ratio:.2%}") #25.95%

df_hours = df_hours[~df_hours[open_cols].isna().all(axis=1)] # 모든 요일이 NaN인 가게는 제거

#=================================
# 요일별 영업하지 않는 비율
#=================================
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

closed_ratios = {}
for day in days:
    open_col = f'{day}_open'
    total = len(df_hours) # 전체 갯수
    closed = df_hours[open_col].isna().sum() # 선택 요일에 _open이 NA 즉 close인 갯수
    closed_ratios[day.capitalize()] = closed / total  

# 결과를 DataFrame으로 정리
df_closed_ratio = pd.DataFrame.from_dict(closed_ratios, orient='index', columns=['closed_ratio'])




