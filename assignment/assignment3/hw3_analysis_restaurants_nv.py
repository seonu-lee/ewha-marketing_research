import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pprint import pprint

#=================================
# 공통 설정
#=================================
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment2\results"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment3\results"

meta_cols = ['name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories']

#=================================
# 0. 데이터 불러오기
#=================================
# TF vs TF-IDF 비교용 (max_df=0.9)
df_dtm_09 = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm.csv")

# 고유단어/경쟁브랜드 분석용 (max_df=1.0, 공통단어 포함)
df_dtm_10 = pd.read_csv(f"{PATH_to_data}/reviews_restaurants_nv_perBrand_0.1_1.0_0.3_10_dtm.csv")

word_cols_09 = [col for col in df_dtm_09.columns if col not in meta_cols]
word_cols_10 = [col for col in df_dtm_10.columns if col not in meta_cols]

data_tf_09 = df_dtm_09[word_cols_09]  # TF vs TF-IDF 비교용
data_tf_10 = df_dtm_10[word_cols_10]  # 고유단어/경쟁브랜드 분석용

print(f"DTM (0.9) shape: {df_dtm_09.shape}")
print(f"DTM (1.0) shape: {df_dtm_10.shape}")
print(f"\n브랜드 목록 상위 10개:")
print(df_dtm_09['name'].head(10).tolist())

# 0.9 DTM: 330개 단어 (공통단어 제거)
# 1.0 DTM: 419개 단어 (공통단어 포함)
# 상위 브랜드 보면 hashhouseagogo, bacchanalbuffet, wickedspoon 등 라스베이거스 유명 레스토랑들이 보임.

#=================================
# 1. TF-IDF 생성 (0.9 DTM 기반)
#=================================
def create_tfidf(data_tf, apply_l2):
    if apply_l2:
        tfidf = TfidfTransformer(norm='l2')
    else:
        tfidf = TfidfTransformer(norm=None)
    dtm_tfidf = tfidf.fit_transform(data_tf)
    df_tfidf = pd.DataFrame(dtm_tfidf.toarray(), columns=data_tf.columns)
    df_tfidf = df_tfidf.round(5)
    return df_tfidf

data_tfidf_09 = create_tfidf(data_tf_09, apply_l2=False)
data_tfidf_l2_09 = create_tfidf(data_tf_09, apply_l2=True)

print(f"TF-IDF shape: {data_tfidf_09.shape}")
print(f"TF-IDF+L2 shape: {data_tfidf_l2_09.shape}")
print(data_tfidf_09.head(3))

# TF-IDF shape: (3891, 323)
# TF-IDF+L2 shape: (3891, 323)
# TF-IDF 변환 과정에서 모든 값이 0인 열이 제거되어 단어 수가 330에서 323으로 줄어듦. TF-IDF+L2는 TF-IDF와 동일한 shape을 가짐 (L2 정규화는 값의 크기만 조정하기 때문).

#=================================
# 2. TF vs TF-IDF 차이 분석
#=================================

# 공통 단어 컬럼 (TF-IDF 변환 후 남은 컬럼 기준)
common_cols = data_tfidf_09.columns.tolist()
data_tf_09_common = data_tf_09[common_cols]

#---------------------------------
# 2.1 출현 빈도 상위 단어 비교
#---------------------------------
k = 30
top_tf = data_tf_09_common.sum().sort_values(ascending=False).head(k)
top_tfidf = data_tfidf_09.sum().sort_values(ascending=False).head(k)
top_tfidf_l2 = data_tfidf_l2_09.sum().sort_values(ascending=False).head(k)

comparison = pd.DataFrame({
    'rank_tf': top_tf.index,
    'weight_tf': top_tf.values,
    'rank_tfidf': top_tfidf.index,
    'weight_tfidf': top_tfidf.values,
    'rank_tfidf_l2': top_tfidf_l2.index,
    'weight_tfidf_l2': top_tfidf_l2.values,
})
print("===== 상위 단어 순위 비교 =====")
print(comparison.to_string())

#---------------------------------
# 2.2 누적 비중 곡선 비교
#---------------------------------
cum_tf = data_tf_09_common.sum(axis=0).sort_values(ascending=False).cumsum()
cum_tfidf = data_tfidf_09.sum(axis=0).sort_values(ascending=False).cumsum()
cum_tfidf_l2 = data_tfidf_l2_09.sum(axis=0).sort_values(ascending=False).cumsum()

cum_tf = cum_tf / cum_tf.iloc[-1]
cum_tfidf = cum_tfidf / cum_tfidf.iloc[-1]
cum_tfidf_l2 = cum_tfidf_l2 / cum_tfidf_l2.iloc[-1]

ranks = np.arange(1, len(cum_tf) + 1)
reference_prop_point = 0.8

cut_tf = np.searchsorted(cum_tf.values, reference_prop_point) + 1
cut_tfidf = np.searchsorted(cum_tfidf.values, reference_prop_point) + 1
cut_tfidf_l2 = np.searchsorted(cum_tfidf_l2.values, reference_prop_point) + 1

fig = go.Figure()
fig.add_trace(go.Scatter(x=ranks, y=cum_tf, mode='lines', name='TF'))
fig.add_trace(go.Scatter(x=ranks, y=cum_tfidf, mode='lines', line=dict(color='firebrick'), name='TF-IDF'))
fig.add_trace(go.Scatter(x=ranks, y=cum_tfidf_l2, mode='lines', line=dict(color='blue'), name='TF-IDF+L2'))
fig.add_hline(y=reference_prop_point, line_dash="dot", line_color="gray", line_width=2)
fig.add_vline(x=cut_tf, line_dash="dash", line_color="green", line_width=2)
fig.add_vline(x=cut_tfidf, line_dash="dash", line_color="red", line_width=2)
fig.add_vline(x=cut_tfidf_l2, line_dash="dash", line_color="blue", line_width=2)
fig.update_layout(
    width=800, height=450, template="simple_white",
    title=f"누적 비중 비교 (80%) | TF: {cut_tf}개 · TF-IDF: {cut_tfidf}개 · TF-IDF+L2: {cut_tfidf_l2}개",
    xaxis_title='Rank', yaxis_title='Cumulative Share',
)
fig.show()
fig.write_image(f"{PATH_to_save}/cumulative_share_nv.png", scale=2)

#---------------------------------
# 2.3 코사인 유사도 비교
#---------------------------------
sim_tf = cosine_similarity(data_tf_09_common)
sim_tfidf = cosine_similarity(data_tfidf_09)
sim_tfidf_l2 = cosine_similarity(data_tfidf_l2_09)

upper_idx = np.triu_indices_from(sim_tf, 1)

print(f"\n===== 코사인 유사도 비교 =====")
print(f"TF       mean={sim_tf[upper_idx].mean():.3f}, std={sim_tf[upper_idx].std():.3f}")
print(f"TF-IDF   mean={sim_tfidf[upper_idx].mean():.3f}, std={sim_tfidf[upper_idx].std():.3f}")
print(f"TF-IDF+L2 mean={sim_tfidf_l2[upper_idx].mean():.3f}, std={sim_tfidf_l2[upper_idx].std():.3f}")

# [인사이트] TF vs TF-IDF 차이 분석

# 1. 상위 단어 순위 변화

# TF 기준 상위 단어는 `fri`, `tabl`, `sauc`, `dish` 등 레스토랑 리뷰에서 보편적으로 등장하는 단어들이 상위를 차지함. 
# TF-IDF 적용 후 `burger`, `sushi`, `buffet`, `steak`, `sandwich` 등 특정 브랜드에 집중되는 음식명의 순위가 상승하고, `tabl`, `dish`, `seat` 등 보편적 서비스 단어의 순위가 하락함. 
# 이는 TF-IDF가 브랜드 특화 단어를 효과적으로 부각시킴을 보여줌. 특히 NV 특유의 `buffet`, `sushi`가 TF-IDF에서 상위로 올라온 점은 라스베이거스 외식 문화의 특성을 반영함.

# ---

# 2. 누적 비중 비교

# 전체 가중치의 80%를 커버하는 데 필요한 단어 수가 TF 159개 → TF-IDF 186개 → TF-IDF+L2 188개로 증가함. 
# TF-IDF 적용 후 소수 고빈도 단어의 지배력이 완화되고 단어 간 가중치가 더 균등하게 분산됨을 확인할 수 있음.

# ---

# 3. 코사인 유사도 비교

# TF 유사도 평균(0.232)이 TF-IDF(0.159)보다 높음. TF 기준에서 모든 브랜드가 `fri`, `tabl`, `sauc` 등 고빈도 공통어를 공유하여 인위적으로 유사하게 보이기 때문임. 
# TF-IDF 적용 후 공통어 영향이 제거되어 브랜드 간 실질적 차별성이 더 잘 반영됨. TF-IDF와 TF-IDF+L2의 유사도가 동일(0.159)한 것은 L2 정규화가 벡터의 방향은 바꾸지 않고 크기만 조정하기 때문임.


#=================================

# 브랜드 목록 확인
print(df_dtm_10['name'].tolist())


# Buffets 카테고리 브랜드 목록 및 기본 정보 확인
buffet_brands = df_dtm_09[df_dtm_09['categories'].str.contains('Buffet', na=False)][
    ['name', 'review_count', 'avg_stars', 'categories']
].sort_values('review_count', ascending=False)

print(f"Buffets 카테고리 브랜드 수: {len(buffet_brands)}")
print(buffet_brands.to_string())

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 1. 리뷰 수 상위 15개만 추리기
buffet_top = buffet_brands.head(15)
print("=== 리뷰 수 상위 15개 뷔페 브랜드 ===")
print(buffet_top[['name', 'review_count', 'avg_stars']].to_string())

# 2. 해당 브랜드들 간 코사인 유사도 계산
buffet_names = buffet_top['name'].tolist()
X_tfidf_10 = df_dtm_10.set_index('name')[word_cols_10]
buffet_sim = cosine_similarity(X_tfidf_10.loc[buffet_names])
buffet_sim_df = pd.DataFrame(buffet_sim, index=buffet_names, columns=buffet_names)

print("\n=== 브랜드 간 코사인 유사도 ===")
print(buffet_sim_df.round(3).to_string())

#=================================

#=================================
# 3. 고유단어 추출 함수 정의
#=================================
def log_odds_dirichlet(count_row, count_corpus, alpha=0.01):
    cw_A = count_row + alpha
    cw_B = (count_corpus - count_row) + alpha
    delta = np.log(cw_A) - np.log(np.sum(cw_A)) - (np.log(cw_B) - np.log(np.sum(cw_B)))
    var = 1/(cw_A) + 1/(cw_B)
    z = delta / np.sqrt(var)
    return pd.DataFrame({'log_odds': delta, 'z_score': z})

def extracting_unique_words_for_brand(data_tf, df_tf, brand_name):
    brand_idx = df_tf[df_tf['name'] == brand_name].index[0]
    counts_A = data_tf.iloc[brand_idx].values
    counts_C = data_tf.sum(axis=0).values
    df_logodds = log_odds_dirichlet(count_row=counts_A, count_corpus=counts_C, alpha=0.1)
    df_logodds.index = data_tf.columns
    df_logodds = df_logodds.sort_values('z_score', ascending=False)
    return df_logodds

#=================================
# 4. 브랜드별 고유단어 추출
#=================================
brand_A = 'bacchanalbuffet'
brand_B = 'genkoreanbbqhouse'

df_logodds_A = extracting_unique_words_for_brand(data_tf_10, df_dtm_10, brand_A)
df_logodds_B = extracting_unique_words_for_brand(data_tf_10, df_dtm_10, brand_B)

print(f"===== {brand_A} 고유단어 상위 10개 =====")
print(df_logodds_A.head(10))
print(f"\n===== {brand_A} 고유단어 하위 10개 =====")
print(df_logodds_A.tail(10))

print(f"\n===== {brand_B} 고유단어 상위 10개 =====")
print(df_logodds_B.head(10))
print(f"\n===== {brand_B} 고유단어 하위 10개 =====")
print(df_logodds_B.tail(10))

# [인사이트] 브랜드별 고유단어 분석


# bacchanalbuffet 고유단어

# 상위 단어: `buffet`(279.7), `line`(143.9), `leg`(139.4), `station`(137.6), `crab`(134.3), `seafood`(110.3), `dessert`(93.6), `oyster`(78.6)

# 카지노 호텔 대형 뷔페의 특성이 잘 드러남. `station`(음식 스테이션), `line`(줄 서기), `select`(선택) 등 뷔페 운영 방식에 대한 언급이 많고, `crab`, `seafood`, `oyster`, `leg`(킹크랩 다리) 등 고급 해산물 메뉴가 핵심 고유단어로 나타남.

# 하위 단어: `order`(-53.8), `place`(-42.8), `servic`(-34.5), `restaur`(-33.8)

# `order`와 `servic`이 낮다는 건 뷔페 특성상 주문/서비스 개념이 약함을 반영함.

# ---

# genkoreanbbqhouse 고유단어

# 상위 단어: `korean`(85.5), `bbq`(55.8), `server`(46.7), `meat`(45.2), `ayc`(38.6), `belli`(33.5), `grill`(31.2), `brisket`(28.9)

# 한국식 BBQ 특성이 뚜렷하게 나타남. `ayc`(All You Can eat), `belli`(삼겹살), `brisket`(차돌박이) 등 한국 BBQ 메뉴 용어가 고유단어로 부각됨. `server`와 `tabl`이 높은 것은 고기를 직접 구워주는 테이블 서비스 방식을 반영함.

# 하위 단어: `buffet`(-9.5), `strip`(-8.5), `vega`(-8.6), `fri`(-11.3)

# `strip`과 `vega`가 낮다는 건 라스베이거스 스트립 관광객보다 로컬 고객 중심임을 시사함.

#=================================
# 5. 두 브랜드 비교 테이블
#=================================
def compare_unique_words_for_two_brands(data_tf, df_tf, brands_to_compare):

    brandA, brandB = brands_to_compare

    # logodds 계산
    df_logodds_A = extracting_unique_words_for_brand(data_tf, df_tf, brandA).reset_index().rename(columns={'index': 'word'})
    df_logodds_A['brand'] = brandA
    df_logodds_B = extracting_unique_words_for_brand(data_tf, df_tf, brandB).reset_index().rename(columns={'index': 'word'})
    df_logodds_B['brand'] = brandB

    df_pooled = pd.concat([df_logodds_A, df_logodds_B], axis=0)
    pivot_z = df_pooled.pivot(index='word', columns='brand', values='z_score')

    # High/Low 판정
    thr = 1.96
    high = pivot_z > thr
    low = pivot_z < -thr

    # 2x2 테이블
    HH = (high[brandA] & high[brandB]).sum()
    HL = (high[brandA] & low[brandB]).sum()
    LH = (low[brandA] & high[brandB]).sum()
    LL = (low[brandA] & low[brandB]).sum()

    hl_freq_table = pd.DataFrame(
        [[HH, HL], [LH, LL]],
        index=[f'{brandA} High', f'{brandA} Low'],
        columns=[f'{brandB} High', f'{brandB} Low']
    )

    # 전용 키워드 추출
    hl_words_dic = {
        f'{brandA}_only': pivot_z.index[high[brandA] & low[brandB]].tolist(),
        f'{brandB}_only': pivot_z.index[low[brandA] & high[brandB]].tolist(),
        'common_high': pivot_z.index[high[brandA] & high[brandB]].tolist(),
        'common_low': pivot_z.index[low[brandA] & low[brandB]].tolist(),
    }

    return hl_freq_table, hl_words_dic

# 적용
brands_to_compare = [brand_A, brand_B]
hl_freq_table, hl_words_dic = compare_unique_words_for_two_brands(data_tf_10, df_dtm_10, brands_to_compare)

print("===== 2x2 비교 테이블 =====")
print(hl_freq_table)
print("\n===== 단어 분류 =====")
pprint(hl_words_dic, width=80, compact=True)

#=================================
# 6. 경쟁 브랜드 추출
#=================================
def nearest_brands(target_brand, X, topn=10):
    sim = cosine_similarity(X.loc[[target_brand]], X)[0]
    s = pd.Series(sim, index=X.index).sort_values(ascending=False)
    return s.drop(target_brand).head(topn)

# TF-IDF 기반 유사도 (1.0 DTM)
X_tfidf_10 = df_dtm_10.set_index('name')[word_cols_10]

print(f"===== {brand_A} 경쟁 브랜드 =====")
print(nearest_brands(brand_A, X_tfidf_10, topn=10))

print(f"\n===== {brand_B} 경쟁 브랜드 =====")
print(nearest_brands(brand_B, X_tfidf_10, topn=10))

# [인사이트] 경쟁 브랜드 추출


# bacchanalbuffet 경쟁 브랜드

# 상위 10개가 모두 뷔페 브랜드로 구성됨. 
# `thebuffet`(0.957), `thebuffetatbellagio`(0.956), `carnivalworldbuffet`(0.954), `wickedspoon`(0.953) 등 라스베이거스 카지노 호텔 뷔페들이 상위를 차지함. 
# 단어 사용 패턴 기반 경쟁 관계가 실제 시장의 경쟁 구도(카지노 호텔 뷔페 간 경쟁)와 일치함.

# ---

# genkoreanbbqhouse 경쟁 브랜드

# 상위 10개가 모두 한국식 BBQ 브랜드로 구성됨. 
# `hwaro`(0.952), `dohkoreanbbq`(0.936), `leeskoreanbbq`(0.933) 등 한국식 BBQ 전문점이 상위를 차지하며, `gyukakujapanesebbq`(0.915)는 한국식은 아니지만 테이블 BBQ 방식을 공유하는 일본식 BBQ로 언어 패턴이 유사하게 나타남.


#=================================
# 7. 경쟁 브랜드 고유단어 분석
#=================================
def analyze_competitor_brands(target_brand, X, data_tf, df_tf, topn=10):
    
    # 경쟁 브랜드 추출
    competitors = nearest_brands(target_brand, X, topn=topn)
    
    print(f"\n{'='*60}")
    print(f"[{target_brand}] 경쟁 브랜드 고유단어 분석")
    print(f"{'='*60}")
    
    for competitor in competitors.index:
        df_logodds = extracting_unique_words_for_brand(data_tf, df_tf, competitor)
        top_words = df_logodds.head(10).index.tolist()
        print(f"\n--- {competitor} (유사도: {competitors[competitor]:.3f}) ---")
        print(f"고유단어 상위 10개: {top_words}")

# bacchanalbuffet 경쟁 브랜드 분석
analyze_competitor_brands(brand_A, X_tfidf_10, data_tf_10, df_dtm_10, topn=10)

# genkoreanbbqhouse 경쟁 브랜드 분석
analyze_competitor_brands(brand_B, X_tfidf_10, data_tf_10, df_dtm_10, topn=10)

# [인사이트] 경쟁 브랜드 고유단어 분석

# bacchanalbuffet 경쟁 브랜드

# 10개 경쟁 브랜드 모두 `buffet`, `leg`(킹크랩 다리), `crab`, `station`, `line`, `dessert`, `prime`, `select`가 반복적으로 등장함. 
# 라스베이거스 카지노 호텔 뷔페들이 공통적으로 해산물, 스테이션 방식, 디저트를 핵심 경험으로 제공하고 있음을 시사함.

# 브랜드별 차별화 단어를 보면:
# - `thebuffetatbellagio` → `bellagio` : 벨라지오 호텔 브랜드 정체성 강조
# - `carnivalworldbuffet` → `rio` : 리오 호텔 소속
# - `wickedspoon` → `brunch`, `strawberri`, `mac` : 브런치 메뉴 특화
# - `spicemarketbuffet` → `market`, `spice`, `groupon` : 가성비 이미지
# - `thebuffetataria` → `aria`, `indian` : 아리아 호텔 소속, 인도 음식 특화

# 즉 경쟁 브랜드들은 **뷔페라는 공통 포맷 안에서 호텔 브랜드 또는 특정 메뉴로 차별화**하는 전략을 취하고 있음.

# ---

# genkoreanbbqhouse 경쟁 브랜드

# 10개 경쟁 브랜드 모두 `korean`, `bbq`, `meat`, `brisket`(`차돌박이`), `belli`(`삼겹살`), `ayc`(All You Can Eat), `grill`이 반복적으로 등장함. 
# 한국식 BBQ 시장이 메뉴 구성과 운영 방식(AYCE)이 매우 표준화되어 있음을 반영함.

# 브랜드별 차별화 단어를 보면:
# - `hwaro` → `shake` : 음료 메뉴 차별화
# - `leeskoreanbbq` → `servic`, `qualiti` : 서비스 품질 강조
# - `gangnamasianbbqdining` → `combo`, `miso`, `happi` : 아시아 퓨전 메뉴 포함
# - `gyukakujapanesebbq` → `japanes`, `miso`, `garlic` : 일본식 BBQ 스타일
# - `goongkoreanbbqrestaurant` → `side`, `qualiti` : 사이드 메뉴와 품질 강조

# 한국식 BBQ 경쟁 브랜드들은 핵심 메뉴는 동일하지만 서비스 품질, 사이드 메뉴, 퓨전 요소로 미세한 차별화를 시도하고 있음.

