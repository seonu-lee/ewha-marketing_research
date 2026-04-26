import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
import plotly.io as pio
pio.renderers.default = "vscode"

import sys
sys.path.append(r"C:\Users\seonu\Documents\ewha-marketing_research")
from lib.lib_dtm import lib_filtering_dtm as lfd

#=================================
# 공통 설정
#=================================
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment4\results"
meta_cols_pool = ['name', 'review_count', 'avg_stars', 'useful_count', 
                  'funny_count', 'cool_count', 'categories']


#=================================
# PCA 함수 정의
#=================================
def pca_scree_plot_perBrand(data_w_meta_cols, apply_stdscaler, apply_l2, input_data_filtering_conditions):
    df = data_w_meta_cols.set_index('name')
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    X_scaled = X.copy()
    if apply_stdscaler:
        X_scaled = StandardScaler().fit_transform(X)
    if apply_l2:
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    pca = PCA()
    pca.fit(X_scaled)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    subtitle = '<br>'.join([
        f'input_file_name: {input_data_filtering_conditions.get("input_file_name")}',
        f'apply_stdscaler: {apply_stdscaler}',
        f'apply_l2_norm: {apply_l2}',
        f'brand_categories_slted: {input_data_filtering_conditions.get("brand_categories_slted")}',
        f'words_to_include_exclusively: {input_data_filtering_conditions.get("words_to_include_exclusively")}',
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_var)+1)), y=cumulative_var,
        mode='lines+markers', name='누적 설명 분산', line=dict(color='blue')))
    fig.add_trace(go.Bar(x=list(range(1, len(explained_var)+1)), y=explained_var,
        name='개별 설명 분산', marker=dict(color='lightgray'), opacity=0.7))
    fig.add_shape(type='line', x0=1, x1=len(cumulative_var), y0=0.9, y1=0.9,
        line=dict(color='red', dash='dot'))
    fig.update_layout(
        title=dict(text=f'Scree Plot<br><span style="font-size:11px">{subtitle}</span>',
            x=0.5, xanchor='center'),
        xaxis_title='주성분 개수', yaxis_title='설명력',
        xaxis=dict(dtick=1), yaxis=dict(range=[0, 1.05]),
        plot_bgcolor='white',
        width=1200,   
        height=600
    )
    return fig


def calculate_pca_coeff_score_perBrand(data_w_meta_cols, apply_stdscaler, apply_l2, num_comp_to_extract):
    df = data_w_meta_cols.copy()
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    X = df.drop(columns=meta_cols)

    X_scaled = X.copy()
    if apply_stdscaler:
        X_scaled = StandardScaler().fit_transform(X)
    if apply_l2:
        X_scaled = normalize(X_scaled, norm="l2", axis=1)

    pca = PCA(n_components=num_comp_to_extract)
    score = pca.fit_transform(X_scaled)
    coeff = np.transpose(pca.components_)
    print('각 주성분 설명 분산(%):', pca.explained_variance_ratio_.round(4)*100)
    print(f'입력 데이터: {X_scaled.shape} → score: {score.shape}, coeff: {coeff.shape}')

    pca_score_columns = [f'x{i}' for i in range(num_comp_to_extract)]
    pca_df = pd.DataFrame(score, columns=pca_score_columns)
    pca_score_w_meta_cols = pd.concat([data_w_meta_cols[meta_cols].reset_index(drop=True), pca_df], axis=1)

    pca_coeff_df = pd.DataFrame(coeff, columns=pca_score_columns)
    pca_coeff_df['word'] = X.columns
    pca_coeff_df = pca_coeff_df.set_index('word').reset_index()

    return pca_score_w_meta_cols, pca_coeff_df


def graph_pca_biplot_perBrand(pca_score_w_meta_cols, pca_coeff_df, num_words_to_display, scale_factor, apply_stdscaler, apply_l2, input_data_filtering_conditions):
    pca_df = pca_score_w_meta_cols.copy()
    coeff_df = pca_coeff_df.copy()

    if scale_factor == "auto":
        pca_score_scope = np.percentile(np.abs(np.array(pca_df[['x0', 'x1']])), 85)
        coeff_scope = np.percentile(np.abs(np.array(coeff_df[['x0', 'x1']])), 90)
        scale_factor = pca_score_scope / coeff_scope

    brand_circle_size = 5 + pca_df['review_count'] * 0.01

    coeff_df['loading_strength'] = np.linalg.norm(coeff_df[['x0', 'x1']].values, axis=1)
    coeff_df = coeff_df.sort_values(by='loading_strength', ascending=False)
    top_coeff_df = coeff_df.head(num_words_to_display)

    subtitle = '<br>'.join([
        f'input_file_name: {input_data_filtering_conditions.get("input_file_name")}',
        f'apply_stdscaler: {apply_stdscaler}',
        f'apply_l2_norm: {apply_l2}',
        f'brand_categories_slted: {input_data_filtering_conditions.get("brand_categories_slted")}',
        f'words_to_include_exclusively: {input_data_filtering_conditions.get("words_to_include_exclusively")}',
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pca_df['x0'], y=pca_df['x1'], mode='markers',
        marker=dict(size=brand_circle_size, color=pca_df['avg_stars'],
            colorscale='RdBu_r', colorbar=dict(title='Avg Stars')),
        text=pca_df['name'],
        hovertemplate='Brand: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    for _, row in top_coeff_df.iterrows():
        x_end = row['x0'] * scale_factor
        y_end = row['x1'] * scale_factor
        fig.add_trace(go.Scatter(
            x=[0, x_end], y=[0, y_end],
            mode='lines+text', line=dict(color='black', width=1),
            text=[None, row['word']], textposition='top center',
            textfont=dict(size=12), showlegend=False
        ))
    fig.update_layout(
        title=dict(text=f'PCA Biplot<br><span style="font-size:11px">{subtitle}</span>',
            x=0.5, xanchor='center'),
        xaxis_title='PC1', yaxis_title='PC2',
        plot_bgcolor='white', showlegend=False,
         width=1200,   # 추가
        height=800     # 추가
    )
    return fig


def get_top_words_per_pc(pca_coeff_df, top_n=10):
    rows_list = []
    pc_cols = [col for col in pca_coeff_df.columns if col != 'word']
    for pc in pc_cols:
        top_rows = pca_coeff_df.reindex(
            pca_coeff_df[pc].abs().sort_values(ascending=False).index).head(top_n)
        words_signed = [f"{w}(+)" if loading > 0 else f"{w}(-)"
            for w, loading in zip(top_rows['word'], top_rows[pc])]
        row_dict = {'pc': pc}
        row_dict.update({i: word for i, word in enumerate(words_signed)})
        rows_list.append(row_dict)
    return pd.DataFrame(rows_list)


#=================================
# CASE 1: 전체 브랜드 + 전체 단어
# 조합 7가지 PCA
#=================================
cases = [
    # (input_file_name,                                          apply_stdscaler, apply_l2)
    ('reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm',         False, False),  # 1) tf
    ('reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf',   False, False),  # 2) tfidf
    ('reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf',   True,  False),  # 3) tfidf + std
    ('reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf',   False, True ),  # 4) tfidf + l2
    ('reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf',   True,  True ),  # 5) tfidf + std + l2
    ('reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2',True,  False),  # 6) tfidf_l2 + std
    ('reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2',True,  True ),  # 7) tfidf_l2 + std + l2
]

if __name__ == '__main__':
    for input_file_name, apply_stdscaler, apply_l2 in cases:
        print(f"\n{'='*60}")
        print(f"input: {input_file_name} | std: {apply_stdscaler} | l2: {apply_l2}")
        print(f"{'='*60}")

        input_data_filtering_conditions = dict(
            input_file_name=input_file_name,
            remove_brand_w_word_in_name=False,
            brand_categories_slted=[],
            words_to_delete=[],
            words_to_include_exclusively=[],
        )

        # 1) 데이터 추출
        data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)

        # 2) Scree plot
        fig_scree = pca_scree_plot_perBrand(data_w_meta_cols, apply_stdscaler, apply_l2, input_data_filtering_conditions)
        fig_scree.show()

        # 3) PCA coeff, score
        pca_score_w_meta_cols, pca_coeff_df = calculate_pca_coeff_score_perBrand(
            data_w_meta_cols, apply_stdscaler, apply_l2, num_comp_to_extract=2)

        # 4) Biplot
        fig_biplot = graph_pca_biplot_perBrand(
            pca_score_w_meta_cols, pca_coeff_df,
            num_words_to_display=30, scale_factor='auto',
            apply_stdscaler=apply_stdscaler, apply_l2=apply_l2,
            input_data_filtering_conditions=input_data_filtering_conditions)
        fig_biplot.show()

        # 5) 주성분별 상위 단어
        top_words_df = get_top_words_per_pc(pca_coeff_df, top_n=10)
        print(top_words_df)

# 그래프 브라우저 오류-> vscode로 6번만 다시 실행
input_data_filtering_conditions = dict(
    input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2',
    remove_brand_w_word_in_name=False,
    brand_categories_slted=[],
    words_to_delete=[],
    words_to_include_exclusively=[],
)
data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
pca_score_w_meta_cols, pca_coeff_df = calculate_pca_coeff_score_perBrand(
    data_w_meta_cols, apply_stdscaler=True, apply_l2=False, num_comp_to_extract=2)
fig_biplot = graph_pca_biplot_perBrand(
    pca_score_w_meta_cols, pca_coeff_df,
    num_words_to_display=30, scale_factor='auto',
    apply_stdscaler=True, apply_l2=False,
    input_data_filtering_conditions=input_data_filtering_conditions)
fig_biplot.show()

# 최적 조합: 6번 (TF-IDF_l2 + std)
# 단어벡터 해석:

# PC1(+) 방향: dinner, tabl, server, seat, waiter → 풀서비스 다이닝 경험 축
# PC2(+) 방향: rice, noodl, soup, beef, dish → 아시아 음식 축
# PC2(-) 방향: bar, bartend, game, watch, play → 바·엔터테인먼트 축
# 단어 벡터들이 여러 방향으로 고르게 분산되어 각 축의 의미가 명확하게 해석됨

# 브랜드 위치 해석:

# 전체 공간에 고르게 분포하여 브랜드 간 포지셔닝 비교가 용이함
# 원 크기(리뷰 수)와 무관하게 단어 패턴만으로 포지셔닝됨
# 오른쪽에 위치한 브랜드일수록 풀서비스 다이닝 특성이 강하고, 왼쪽은 아시아 음식 전문점 특성

# 선택 이유:

# TF-IDF 단계에서 l2를 먼저 적용해 리뷰 수 영향을 완전히 제거한 후 std로 단어 가중치를 균등화했기 때문에 리뷰 수 편향과 고빈도 단어 독식 문제를 동시에 해결함

#=================================
# CASE 2: Steakhouses 브랜드 선별 + 전체 단어
# 최적 조합: TF-IDF_l2 + std (apply_stdscaler=True, apply_l2=False)
#=================================
if __name__ == '__main__':

    input_data_filtering_conditions = dict(
        input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2',
        remove_brand_w_word_in_name=False,
        brand_categories_slted=['Steakhouses'],
        words_to_delete=[],
        words_to_include_exclusively=[],
    )
    apply_stdscaler, apply_l2 = True, False

    # 1) 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
    print(f"Steakhouses 브랜드 수: {len(data_w_meta_cols)}개")

    # 2) Scree plot
    fig_scree = pca_scree_plot_perBrand(
        data_w_meta_cols, apply_stdscaler, apply_l2, input_data_filtering_conditions)
    fig_scree.show()

    # 3) PCA coeff, score
    pca_score_w_meta_cols, pca_coeff_df = calculate_pca_coeff_score_perBrand(
        data_w_meta_cols, apply_stdscaler, apply_l2, num_comp_to_extract=2)

    # 4) Biplot
    fig_biplot = graph_pca_biplot_perBrand(
        pca_score_w_meta_cols, pca_coeff_df,
        num_words_to_display=30, scale_factor='auto',
        apply_stdscaler=apply_stdscaler, apply_l2=apply_l2,
        input_data_filtering_conditions=input_data_filtering_conditions)
    fig_biplot.show()

    # 5) 주성분별 상위 단어
    top_words_df = get_top_words_per_pc(pca_coeff_df, top_n=10)
    print(top_words_df)

# CASE 2 결과 분석 (Steakhouses 브랜드 선별)
# Scree plot:
# PC1(6.32%) + PC2(5.58%) = 11.9%로 CASE 1(6.0%+3.3%)보다 설명력이 높아짐. 브랜드를 Steakhouses로 좁혔기 때문에 동질적인 브랜드들 간 변동성이 줄어 주성분 설명력이 상대적으로 높아진 것.
# 단어벡터 해석:

# PC1(+) 방향: dinner, appet, start, dine, dessert, entre, wine → 격식 있는 풀코스 파인다이닝 축
# PC1(-) 방향: sushi → 스테이크하우스와 거리가 먼 메뉴 축
# PC2(+) 방향: waitress, quick, burger, breakfast, fast, fri → 캐주얼·패스트 다이닝 축
# PC2(-) 방향: steakhous, filet, steak, lobster, butter, wine → 정통 스테이크하우스 전문성 축

# 브랜드 위치 해석:

# PC1 오른쪽 + PC2 아래: 정통 스테이크 파인다이닝 (filet, steak, wine 중심)
# PC1 왼쪽: sushi 등 비스테이크 메뉴 포함 브랜드 (야드하우스 등 복합 카테고리)
# PC2 위쪽: 캐주얼한 스테이크 브랜드 (아웃백 등)
# 큰 원(리뷰 많은 브랜드)이 PC1 오른쪽에 몰려있어 인기 스테이크하우스들이 파인다이닝 축에 위치함을 확인

#=================================
# CASE 3: Steakhouses 브랜드 선별 + 서비스/경험 & 장소/분위기 키워드 선별
# 최적 조합: TF-IDF_l2 + std (apply_stdscaler=True, apply_l2=False)
#=================================

# HW2에서 정의한 단어 카테고리 기준
service_experience = [
    'bartend', 'call', 'card', 'chang', 'charg', 'chef', 'clean', 'cours',
    'deliveri', 'dine', 'dinner', 'drive', 'event', 'famili', 'fast',
    'free', 'fun', 'groupon', 'guy', 'happi', 'hard', 'help', 'hour',
    'husband', 'kid', 'ladi', 'late', 'line', 'live', 'long', 'manag',
    'mani', 'music', 'night', 'noth', 'offer', 'ok', 'old', 'option',
    'outsid', 'owner', 'parti', 'pay', 'person', 'play', 'pm',
    'point', 'pub', 'reserv', 'room', 'sake', 'sampl', 'seat', 'select',
    'server', 'show', 'song', 'sport', 'start', 'station', 'stay', 'store',
    'tabl', 'town', 'truck', 'view', 'waiter', 'waitress', 'watch',
    'water', 'week',
]

location_atmosphere = [
    'airport', 'aria', 'ayc', 'band', 'bar', 'bellagio', 'cafe', 'casino',
    'citi', 'club', 'cool', 'court', 'danc', 'diner', 'door', 'downtown',
    'express', 'game', 'grill', 'hotel', 'hous', 'insid', 'island', 'king',
    'kitchen', 'lake', 'loung', 'mall', 'market', 'mgm', 'mr', 'palm',
    'palac', 'park', 'pool', 'prime', 'rio', 'roberto', 'rock', 'shop',
    'spot', 'steakhous', 'street', 'strip', 'venetian', 'walk', 'wine',
    'wynn',
]

words_to_include_exclusively = sorted(list(set(service_experience + location_atmosphere)))
print(f"선별 키워드 수: {len(words_to_include_exclusively)}개")

if __name__ == '__main__':

    input_data_filtering_conditions = dict(
        input_file_name='reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm_tfidf_l2',
        remove_brand_w_word_in_name=False,
        brand_categories_slted=['Steakhouses'],
        words_to_delete=[],
        words_to_include_exclusively=words_to_include_exclusively,
    )
    apply_stdscaler, apply_l2 = True, False

    # 1) 데이터 추출
    data_w_meta_cols = lfd.filtering_dtm_at_brand_level(input_data_filtering_conditions)
    print(f"브랜드 수: {len(data_w_meta_cols)}개")
    
    # 실제 사용된 단어 수 확인
    meta_cols_check = [col for col in data_w_meta_cols.columns if col in meta_cols_pool]
    word_cols_check = [col for col in data_w_meta_cols.columns if col not in meta_cols_pool]
    print(f"실제 사용 단어 수: {len(word_cols_check)}개")

    # 2) Scree plot
    fig_scree = pca_scree_plot_perBrand(
        data_w_meta_cols, apply_stdscaler, apply_l2, input_data_filtering_conditions)
    fig_scree.show()

    # 3) PCA coeff, score
    pca_score_w_meta_cols, pca_coeff_df = calculate_pca_coeff_score_perBrand(
        data_w_meta_cols, apply_stdscaler, apply_l2, num_comp_to_extract=2)

    # 4) Biplot
    fig_biplot = graph_pca_biplot_perBrand(
        pca_score_w_meta_cols, pca_coeff_df,
        num_words_to_display=30, scale_factor='auto',
        apply_stdscaler=apply_stdscaler, apply_l2=apply_l2,
        input_data_filtering_conditions=input_data_filtering_conditions)
    fig_biplot.show()

    # 5) 주성분별 상위 단어
    top_words_df = get_top_words_per_pc(pca_coeff_df, top_n=10)
    print(top_words_df)

# CASE 3 결과 분석 (Steakhouses 브랜드 선별 + 서비스/경험·장소/분위기 키워드 선별)
# Scree plot:
# PC1(8.54%) + PC2(5.69%) = 14.23%로 CASE 2(11.9%)보다 설명력이 더 높아짐. 키워드를 서비스/경험·장소/분위기로 좁히면서 스테이크하우스 간 차별화에 핵심적인 단어들만 남아 주성분 설명력이 향상된 것.
# 단어벡터 해석:

# PC1(+) 방향: seat, tabl, dinner, server, manag, reserv, start, parti → 테이블 서비스 중심의 격식 있는 다이닝 경험 축
# PC1(-) 방향: drive, clean, fast, famili, kid → 캐주얼·패밀리 레스토랑 축
# PC2(+) 방향: wine, waiter, steakhous, reserv, cours → 파인다이닝 전문성 축
# PC2(-) 방향: waitress, fast, kid, famili → 캐주얼 서비스 축

# 브랜드 위치 해석:

# PC1 오른쪽: 격식 있는 테이블 서비스 스테이크하우스 (monamigabi 등 파인다이닝 계열)
# PC1 왼쪽: 캐주얼하거나 복합 카테고리 스테이크하우스 (아웃백 등)
# PC2 위쪽: 와인·코스 중심 정통 파인다이닝
# 오른쪽 하단 파란 원(낮은 별점) 브랜드가 눈에 띄는데 서비스 관련 불만이 많은 브랜드로 해석 가능

# CASE 2 vs CASE 3 차이:

# CASE 2(전체 단어)에서는 sushi, burger 등 음식 단어가 주성분에 혼재되어 스테이크하우스 간 서비스 차별화가 희석됨
# CASE 3(서비스·분위기 키워드)에서는 wine, reserv, cours 등 파인다이닝 경험 단어가 뚜렷하게 분리되어 브랜드 포지셔닝이 더 명확하게 드러남

