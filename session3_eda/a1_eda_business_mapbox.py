'''
위치데이터 시각화

'''


import pandas as pd
import plotly.express as px

#=================================
# 설정
#=================================
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\datasets\yelp_dataset"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\session3_eda\results"

#=================================
# 0. 데이터 불러오기
#=================================
business_raw = pd.read_csv(f"{PATH_to_data}/yelp_business.csv")


# 1) 데이터 준비 ───────────────────────────────────────────
biz_map = business_raw.copy()

biz_map = biz_map[biz_map['categories'].str.contains('Restaurants')] # Health

biz_map = biz_map[["name", "latitude", "longitude", "stars", "review_count", "is_open"]]


# 2) Scatter Mapbox 그리기 ───────────────────────────────
fig = px.scatter_mapbox(
    biz_map,
    lat="latitude",
    lon="longitude",
    color="stars", # 별점에 따라 색상
    size="review_count", # 리뷰 수로 마커 크기
    size_max=30,
    zoom=10,   # 1(전세계) ~ 20(건물) 사이
    hover_data={
        "name": True,
        "stars": True,
        "review_count": True,
        "is_open": True,
        "latitude": False,    # hover에 lat/lon 숨기기
        "longitude": False,
    },
)
fig.update_layout(
    title="Yelp Businesses",
    mapbox_style="carto-positron",   # Mapbox 토큰 없어도 되는 오픈 스타일
    margin=dict(l=0, r=0, t=50, b=0)
)
fig.show()
fig.show(config={"scrollZoom": True, "doubleClick": "reset"})


