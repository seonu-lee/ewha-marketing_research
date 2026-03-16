
'''
그래프
https://plotly.com/python/

1. plotly 그래프 
1.1 plotly express
    - 코드 간결, 직관적
    - 세밀한 제어는 제한적임
1.2 plotly graph_objects
    - 세밀한 제어 가능
    - 코드가 길어짐
1.3 Plotly figure_factory
    - 특수한 그래프 모듈
    - 예, heatmap, dendrogram
'''

import pandas as pd
import plotly.graph_objects as go


### csv 파일 dataframe으로 불러오기
df_loaded = pd.read_csv('', encoding='utf-8-sig')

#==============================
# 지역별 평균 성적 (막대그래프)
#==============================
# 지역별 평균 성적 집계
region_mean = df_loaded.groupby("region")[["math", "lang", "science"]].mean().reset_index()

#---------------------
# 1) 그래프 객체 생성 (빈 그래프 틀 만들기)
#---------------------
fig = go.Figure()

#---------------------
# 2) 각 과목별로 Bar 추가
#---------------------
fig.add_trace(go.Bar(
    x=region_mean["region"],
    y=region_mean["math"],
    name="Math",
    opacity=0.6, # 투명도 조정
    marker_color="blue"
))

fig.add_trace(go.Bar(
    x=region_mean["region"],
    y=region_mean["lang"],
    name="Lang",
    opacity=0.6, # 투명도 조정
    marker_color="red"
))

fig.add_trace(go.Bar(
    x=region_mean["region"],
    y=region_mean["science"],
    name="Science",
    opacity=0.6, # 투명도 조정
    marker_color="green"
))

#---------------------
# 3) 레이아웃 설정
#---------------------
fig.update_layout(
    barmode="group", # 과목별 막대를 나란히 표시, group(기본값), stack, overlay
    title="지역별 평균 성적 비교",
    xaxis_title="Region",
    yaxis_title="Average Score",
    width=800, # 그래프 가로 크기 (픽셀 단위)
    height=500 # 그래프 세로 크기 (픽셀 단위)
)

#---------------------
# 결과 출력, 저장
#---------------------
fig.show() # 브러우저에 보여주기
fig.show(renderer="browser") # 브라우저 지정

import plotly.io as pio
pio.renderers.default = "browser"  # 항상 브라우저로 열기 설정

# 저장하기
PATH_to_save = ""
fig.write_html(f"{PATH_to_save}/overlay_bar_chart.html") # 웹공유, interactive
fig.write_image(f"{PATH_to_save}/overlay_bar_chart.png", width=800, height=500) # 정적이미지 - png, pdf, jpg
fig.write_image(f"{PATH_to_save}/overlay_bar_chart.svg") # 벡터이미지


#==============================
# 개별 학생 성적 분포 (산점도)
#==============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_loaded["math"],
    y=df_loaded["science"],
    mode="markers+text",
    text=df_loaded["name"],
    textposition="top center",
    marker=dict(size=10, color="blue", opacity=0.7)
))

fig.update_layout(
    title="수학 vs 과학 점수 (학생별)",
    xaxis_title="Math Score",
    yaxis_title="Science Score"
)
fig.show()

#==============================
# 과목별 점수 분포 (박스플롯)
#==============================
fig = go.Figure()

for col in ["math", "lang", "science"]:
    fig.add_trace(go.Box(
        y=df_loaded[col], name=col
    ))

fig.update_layout(
    title="과목별 점수 분포 (Boxplot)"
)
fig.show()

