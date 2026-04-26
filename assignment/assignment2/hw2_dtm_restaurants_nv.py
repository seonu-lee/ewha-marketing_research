import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
tqdm.pandas()

#=================================
# 공통 설정
#=================================
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\datasets\yelp_dataset"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment2\results"

#=================================
# 1단계. 리뷰 텍스트 전처리
#=================================
def cleaning_review_text_perBrand(biz_cat_slted, state_slted):

    # 0. 데이터 불러오기
    business_raw = pd.read_csv(f"{PATH_to_data}/yelp_business.csv")
    reviews_raw = pd.read_csv(f"{PATH_to_data}/yelp_review.csv")

    business = business_raw.copy()
    reviews = reviews_raw.copy()

    # 1. 브랜드 이름 클리닝
    business['name_ori'] = business['name']
    business['name'] = business['name'].str.lower()
    business['name'] = business['name'].str.replace('[^a-z0-9]', '', regex=True)

    # 2. 분석 대상 필터링
    business_slted = business.copy()
    business_slted = business_slted[business_slted['state'] == state_slted]
    business_slted = business_slted[business_slted['categories'].str.contains(biz_cat_slted, na=False)]
    business_slted = business_slted[business_slted['review_count'] > 10]

    # 3. 브랜드별 리뷰 aggregation
    reviews_slted = pd.merge(reviews, business_slted[['business_id', 'name']], how='inner', on='business_id')

    brand_reviews = reviews_slted.groupby('name').agg({
        'review_id': 'count',
        'stars': 'mean',
        'text': ' '.join,
        'useful': 'sum',
        'funny': 'sum',
        'cool': 'sum'
    }).reset_index()

    brand_reviews = brand_reviews.rename(columns={
        'review_id': 'review_count',
        'stars': 'avg_stars',
        'text': 'pooled_text',
        'useful': 'useful_count',
        'funny': 'funny_count',
        'cool': 'cool_count'
    })

    brand_reviews = pd.merge(
        brand_reviews,
        business_slted[['name', 'categories']].drop_duplicates(subset=['name'], keep='first'),
        on='name', how='left'
    )
    brand_reviews = (brand_reviews
        .sort_values(by='review_count', ascending=False)
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={'index': 'doc_id'})
    )

    # 4. 텍스트 전처리
    # nltk.download('stopwords')  # 최초 1회만 필요, 이후 주석 처리
    stop_words = set(stopwords.words('english'))
    custom_remove = {"does", "not", "thing"}

    brand_reviews['pooled_text_clean'] = brand_reviews['pooled_text'].str.lower()
    brand_reviews['pooled_text_clean'] = brand_reviews['pooled_text_clean'].str.replace(r'\d+', '', regex=True)
    brand_reviews['pooled_text_clean'] = brand_reviews['pooled_text_clean'].str.replace(r'[^\w\s]', '', regex=True)

    stemmer = PorterStemmer()
    def tokenize_filter_stem(text):
        tokens = text.split()
        tokens = [w for w in tokens if w not in stop_words and w not in custom_remove]
        tokens = [stemmer.stem(w) for w in tokens]
        return ' '.join(tokens)

    brand_reviews['pooled_text_clean'] = brand_reviews['pooled_text_clean'].progress_apply(tokenize_filter_stem)

    # 저장
    brand_reviews = brand_reviews.drop(['pooled_text'], axis=1)
    brand_reviews.to_csv(f"{PATH_to_save}/reviews_{biz_cat_slted.lower()}_{state_slted.lower()}_perBrand.csv", index=False, encoding='utf-8-sig')

    return brand_reviews


if __name__ == '__main__':
    biz_cat_slted = 'Restaurants'
    state_slted = 'NV'
    brand_reviews = cleaning_review_text_perBrand(biz_cat_slted, state_slted)
    print(f"브랜드 수: {len(brand_reviews)}")
    print(brand_reviews.head())

# 브랜드 수: 3,891개 (AZ 5,151개보다 적음)
# 상위 브랜드가 hashhouseagogo, bacchanalbuffet, wickedspoon 등 라스베이거스 유명 레스토랑/뷔페들로 채워져 있어서 NV 선택이 적절했음을 확인할 수 있음
# pooled_text_clean에 la vega, buffet, strip 같은 NV 특유의 단어들이 보임

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

#=================================
# 공통 설정
#=================================
PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment2\results"
PATH_to_save = r"C:\Users\seonu\Documents\ewha-marketing_research\assignment\assignment2\results"

#=================================
# 2단계. DTM 생성
#=================================
def create_dtm_perBrand(input_file_name, min_df_rate, max_df_rate, min_review_prop_in_doc, min_doc_count_finally, manual_stopwords):

    # 0. 데이터 불러오기
    brand_reviews = pd.read_csv(f"{PATH_to_data}/{input_file_name}.csv")

    # 1. DTM 생성
    nltk_stop = stopwords.words('english')
    initial_manual_stopwords = []
    stop_words = sorted(list(set(nltk_stop + initial_manual_stopwords)))

    vectorizer = CountVectorizer(stop_words=stop_words)
    dtm = vectorizer.fit_transform(brand_reviews['pooled_text_clean'])
    terms = vectorizer.get_feature_names_out()
    print(f"초기 DTM shape: {dtm.shape}")  # (브랜드 수 x 전체 단어 수)

    # 2.1 전체 브랜드 기준 희소/공통단어 제거 (비율 기준)
    term_frequencies = np.asarray((dtm > 0).sum(axis=0)).flatten()
    doc_count = dtm.shape[0]
    df_rate = term_frequencies / doc_count

    word_mask = (df_rate >= min_df_rate) & (df_rate < max_df_rate)

    dtm_reduced = dtm[:, word_mask]
    terms_reduced = terms[word_mask]

    dtm_df = pd.DataFrame(dtm_reduced.toarray(), columns=terms_reduced)
    dtm_df['doc_id'] = brand_reviews['doc_id']

    # 제거된 단어 출력 (적절성 검토용)
    sparse_words_deleted = terms[df_rate < min_df_rate]
    common_words_deleted = terms[df_rate >= max_df_rate]
    print(f"\n제거된 희소단어(총 {len(sparse_words_deleted)}개): {sparse_words_deleted}")
    print(f"\n제거된 공통단어(총 {len(common_words_deleted)}개): {common_words_deleted}")

    # 2.2 브랜드별 희소단어 제거
    dtm_df = dtm_df.set_index('doc_id').astype(float)
    review_count = brand_reviews.set_index('doc_id')['review_count']
    dtm_normalized = dtm_df.div(review_count, axis=0)

    high_freq_terms_per_doc = []
    for _, row in dtm_normalized.iterrows():
        high_terms = row[row >= min_review_prop_in_doc].index.tolist()
        high_freq_terms_per_doc.append(high_terms)

    high_freq_terms = sorted(list(set(
        [item for words_batch in high_freq_terms_per_doc for item in words_batch]
    )))

    # 2.3 전체 브랜드 등장 횟수 기준 추가 필터링
    term_doc_counts = {}
    for term in high_freq_terms:
        count = sum(term in term_list for term_list in high_freq_terms_per_doc)
        term_doc_counts[term] = count

    term_doc_df = pd.DataFrame(list(term_doc_counts.items()), columns=['term', 'doc_count'])
    term_doc_df_slted = term_doc_df[term_doc_df['doc_count'] > min_doc_count_finally]
    words_features = term_doc_df_slted['term'].to_list()
    print(f"\n2.1~2.3 필터링 후 남은 단어 수: {len(words_features)}")

    dtm_df = dtm_df[words_features]

    # 3. 수동 불용어 제거
    dtm_df_cleaned = dtm_df.copy()
    to_drop = [w for w in manual_stopwords if w in dtm_df_cleaned.columns]
    dtm_df_cleaned = dtm_df_cleaned.drop(to_drop, axis=1)
    print(f"수동 불용어 제거 후 남은 단어 수: {len(dtm_df_cleaned.columns)}")

    # 최종 정리 및 저장
    dtm_df_cleaned = dtm_df_cleaned.loc[~(dtm_df_cleaned == 0).all(axis=1)]
    dtm_df_cleaned = dtm_df_cleaned.loc[:, ~(dtm_df_cleaned == 0).all(axis=0)]

    brand_reviews_dtm = pd.merge(
        brand_reviews.drop(columns=['pooled_text_clean']),
        dtm_df_cleaned.reset_index(),
        on='doc_id', how='inner'
    )
    brand_reviews_dtm = brand_reviews_dtm.set_index('doc_id')
    brand_reviews_dtm.to_csv(
        f"{PATH_to_save}/{input_file_name}_{min_df_rate}_{max_df_rate}_{min_review_prop_in_doc}_{min_doc_count_finally}_dtm.csv",
        index=False, encoding='utf-8-sig'
    )
    print(f"\n최종 DTM shape: {brand_reviews_dtm.shape}")

    return brand_reviews_dtm


if __name__ == "__main__":

    manual_stopwords = [
        'al', "also", "alway", 'anoth', 'area', 'around', 'ask',
        'back', 'bite', 'box',
        'come', 'could', 'came',
        'dont', 'day', 'de', 'didnt',
        'even', 'ever', 'el',
        'get', 'give', 'got',
        'im', 'ive',
        'let', 'la', 'last',
        'make', 'made', 'mayb',
        'name',
        'one',
        'round',
        'someth', 'still', 'seem', 'sinc', 'sub', 'said',
        'told', 'that', 'think', 'two', 'though', 'thought', 'took',
        'us',
        'want', 'way', 'went', 'would', 'wasnt',
        'your', 'year',
    ]

    input_file_name = 'reviews_restaurants_nv_perBrand'
    min_df_rate = 0.1
    max_df_rate = 0.9
    min_review_prop_in_doc = 0.3
    min_doc_count_finally = 10

    brand_reviews_dtm = create_dtm_perBrand(
        input_file_name, min_df_rate, max_df_rate,
        min_review_prop_in_doc, min_doc_count_finally, manual_stopwords
    )

# 초기 DTM: (3,891 × 374,895)
# 3,891개 브랜드, 374,895개 단어로 시작. AZ(305,676개)보다 단어 수가 많은데, NV 특유의 다양한 언어(일본어 등 관광객 리뷰)가 포함된 영향으로 보임.

# 제거된 희소단어 (371,181개)
# 전체의 99%가 희소단어로 제거됨. 'ﾉﾉ', 'ﾊﾟﾁｸﾘ' 같은 일본어/특수문자가 포함되어 있어, 라스베이거스 관광지 특성상 외국인 리뷰가 섞여 있음을 확인할 수 있음. 제거가 적절함.

# 제거된 공통단어 (131개)
# AZ(112개)보다 많음. 'vega', 'chicken', 'buffet' 계열 단어들이 NV 특유의 공통어로 추가 제거된 점이 눈에 띔. 90% 이상 브랜드에서 등장하니 변별력이 없어 제거가 적절함.

# 최종 DTM: (3,891 × 332)
# 수동 불용어 제거 후 332개 단어가 최종 분석 단어로 확정됨.

# 최종단어목록 
print(brand_reviews_dtm.columns.to_list())

#=================================
# 3단계. 단어 의미 기준 분류
#=================================

word_categories = {

    '음식/메뉴': [
        'appet', 'asada', 'bacon', 'bagel', 'ball', 'bbq', 'bean', 'beef',
        'belli', 'bowl', 'bread', 'breakfast', 'brisket', 'broth', 'brunch',
        'buffet', 'bun', 'burger', 'burrito', 'butter', 'caesar', 'cake',
        'carn', 'cart', 'cevich', 'chees', 'cheesesteak', 'chili', 'chip',
        'chocol', 'chow', 'clam', 'combo', 'cooki', 'corn', 'crab', 'cream',
        'crepe', 'crust', 'cup', 'deli', 'dessert', 'dim', 'dip', 'dish',
        'dog', 'donut', 'duck', 'dumpl', 'egg', 'enchilada', 'entre',
        'filet', 'finger', 'fish', 'french', 'fri', 'fruit', 'garlic',
        'green', 'hash', 'hummu', 'ice', 'ingredi', 'item', 'juic', 'lamb',
        'latt', 'leg', 'lobster', 'lunch', 'mac', 'margarita', 'meat',
        'meatbal', 'milk', 'miso', 'mushroom', 'nacho', 'noodl', 'onion',
        'oyster', 'pad', 'pancak', 'pasta', 'pastri', 'pepper', 'pho',
        'pie', 'pita', 'pizza', 'plate', 'poke', 'pork', 'portion', 'pot',
        'potato', 'pull', 'ramen', 'rib', 'rice', 'roast', 'roll', 'salad',
        'salmon', 'salsa', 'salt', 'sandwich', 'sangria', 'sashimi', 'sauc',
        'sausag', 'seafood', 'shake', 'shrimp', 'slice', 'slider', 'smoothi',
        'soup', 'spring', 'steak', 'strawberri', 'sum', 'sushi', 'taco',
        'tapa', 'tea', 'tempura', 'tender', 'teriyaki', 'toast', 'tofu',
        'tomato', 'tortilla', 'truffl', 'tuna', 'turkey', 'waffl', 'wing',
        'wonton', 'wrap', 'york',
    ],

    '음식 특성': [
        'authent', 'bake', 'cold', 'cook', 'crispi', 'cut', 'decent',
        'deep', 'dri', 'excel', 'fill', 'fresh', 'healthi', 'home', 'hot',
        'huge', 'juic', 'large', 'light', 'perfect', 'qualiti', 'quick',
        'red', 'season', 'size', 'skewer', 'smoke', 'sour', 'spice',
        'spici', 'style', 'sweet', 'white',
    ],

    '음식 종류/cuisine': [
        'asian', 'blue', 'chicago', 'china', 'chines', 'curri', 'filipino',
        'greek', 'hawaiian', 'indian', 'italian', 'japanes', 'korean',
        'mexican', 'philli', 'thai', 'vietnames', 'vegan', 'vegetarian',
        'veggi',
    ],

    '서비스/경험': [
        'bartend', 'call', 'card', 'chang', 'charg', 'chef', 'clean',
        'cours', 'deliveri', 'dine', 'dinner', 'drive', 'event', 'famili',
        'fast', 'free', 'fun', 'groupon', 'guy', 'happi', 'hard', 'help',
        'hour', 'husband', 'kid', 'ladi', 'late', 'line', 'live', 'long',
        'manag', 'mani', 'music', 'night', 'noth', 'offer', 'ok', 'old',
        'option', 'outsid', 'owner', 'parti', 'pay', 'perfect', 'person',
        'play', 'pm', 'point', 'pub', 'reserv', 'room', 'sake', 'sampl',
        'seat', 'select', 'server', 'show', 'song', 'sport', 'start',
        'station', 'stay', 'store', 'tabl', 'town', 'truck', 'view',
        'waiter', 'waitress', 'watch', 'water', 'week',
    ],

    '장소/분위기': [
        'airport', 'aria', 'ayc', 'band', 'bar', 'bellagio', 'cafe',
        'casino', 'citi', 'club', 'cool', 'court', 'danc', 'diner',
        'door', 'downtown', 'express', 'game', 'grill', 'hotel', 'hous',
        'insid', 'island', 'king', 'kitchen', 'lake', 'loung', 'mall',
        'market', 'mgm', 'mr', 'palm', 'palac', 'park', 'pool', 'prime',
        'rio', 'roberto', 'rock', 'spot', 'steakhous', 'street', 'strip',
        'venetian', 'walk', 'wine', 'wynn',
    ],

    '가격/가치': [
        'cheap', 'deal', 'favorit', 'half', 'local', 'pepper', 'piec',
        'price', 'prime', 'publ', 'short', 'special', 'steakhous', 'top',
        'value', 'worth',
    ],
}

# 분류 결과 출력
print("===== 단어 의미 기준 분류 결과 =====\n")
all_classified = []
for cat, words in word_categories.items():
    print(f"[{cat}] ({len(words)}개)")
    print(words)
    print()
    all_classified.extend(words)

# 분류되지 않은 단어 확인
dtm_words = [col for col in brand_reviews_dtm.columns if col not in [
    'name', 'review_count', 'avg_stars', 'useful_count',
    'funny_count', 'cool_count', 'categories'
]]
unclassified = [w for w in dtm_words if w not in all_classified]
print(f"[미분류 단어] ({len(unclassified)}개)")
print(unclassified)

# 미분류 단어 10개 재분류 필요

# 불용어 추가
if __name__ == "__main__":

    manual_stopwords = [
        'al', "also", "alway", 'anoth', 'area', 'around', 'ask',
        'back', 'bite', 'box',
        'come', 'could', 'came',
        'dont', 'day', 'de', 'didnt',
        'even', 'ever', 'el',
        'get', 'give', 'got',
        'im', 'ive',
        'let', 'la', 'last',
        'make', 'made', 'mayb',
        'name',
        'one',
        'round',
        'someth', 'still', 'seem', 'sinc', 'sub', 'said',
        'told', 'that', 'think', 'two', 'though', 'thought', 'took',
        'us',
        'want', 'way', 'went', 'would', 'wasnt',
        'your', 'year',
        'ga', 'mi',  # 의미 불명확한 노이즈 단어 추가
    ]

    # 기존 0.9 조건 (주석 처리)
    # input_file_name = 'reviews_restaurants_nv_perBrand'
    # min_df_rate = 0.1
    # max_df_rate = 0.9
    # min_review_prop_in_doc = 0.3
    # min_doc_count_finally = 10

    # 1.0 조건 추가 # HW3에서 max_df_rate 1.0으로 변경하여 공통단어 제거 없이 분석 진행 예정. 비교를 위해 기존 조건도 유지.
    input_file_name = 'reviews_restaurants_nv_perBrand'
    min_df_rate = 0.1
    max_df_rate = 1.0  # 여기만 변경
    min_review_prop_in_doc = 0.3
    min_doc_count_finally = 10

    brand_reviews_dtm = create_dtm_perBrand(
        input_file_name, min_df_rate, max_df_rate,
        min_review_prop_in_doc, min_doc_count_finally, manual_stopwords
    )

# 이전: 332개 → 이번: 330개 (ga, mi 2개 제거 확인)
# 최종 DTM shape: (3,891 × 330)

word_categories = {

    '음식/메뉴': [
        'appet', 'asada', 'bacon', 'bagel', 'bakeri', 'ball', 'bbq', 'bean', 
        'beef', 'beer', 'belli', 'bowl', 'bread', 'breakfast', 'brisket', 
        'broth', 'brunch', 'buffet', 'bun', 'burger', 'burrito', 'butter', 
        'caesar', 'cake', 'carn', 'cart', 'cevich', 'chees', 'cheesesteak', 
        'chili', 'chip', 'chocol', 'chow', 'clam', 'cocktail', 'coffe', 
        'combo', 'cooki', 'corn', 'crab', 'cream', 'crepe', 'crust', 'cup', 
        'deli', 'dessert', 'dim', 'dip', 'dish', 'dog', 'donut', 'duck', 
        'dumpl', 'egg', 'enchilada', 'entre', 'filet', 'finger', 'fish', 
        'french', 'fri', 'fruit', 'garlic', 'green', 'hash', 'hummu', 'ice', 
        'ingredi', 'item', 'juic', 'lamb', 'latt', 'leg', 'lobster', 'lunch', 
        'mac', 'margarita', 'meat', 'meatbal', 'milk', 'miso', 'mushroom', 
        'nacho', 'noodl', 'onion', 'oyster', 'pad', 'pancak', 'pasta', 
        'pastri', 'pepper', 'pho', 'pie', 'pita', 'pizza', 'plate', 'poke', 
        'pork', 'portion', 'pot', 'potato', 'pull', 'ramen', 'rib', 'rice', 
        'roast', 'roll', 'salad', 'salmon', 'salsa', 'salt', 'sandwich', 
        'sangria', 'sashimi', 'sauc', 'sausag', 'seafood', 'shake', 'shave',
        'shrimp', 'slice', 'slider', 'smoothi', 'soup', 'spring', 'steak', 
        'strawberri', 'sum', 'sushi', 'taco', 'tapa', 'tea', 'tempura', 
        'tender', 'teriyaki', 'toast', 'tofu', 'tomato', 'tortilla', 'truffl', 
        'tuna', 'turkey', 'waffl', 'wing', 'wonton', 'wrap', 'york',
    ],

    '음식 특성': [
        'authent', 'bake', 'cold', 'cook', 'crispi', 'cut', 'decent', 'deep', 
        'differ', 'dri', 'excel', 'fill', 'fresh', 'healthi', 'home', 'hot', 
        'huge', 'juic', 'larg', 'large', 'light', 'perfect', 'qualiti', 
        'quick', 'red', 'season', 'size', 'skewer', 'smoke', 'sour', 'spice', 
        'spici', 'style', 'sweet', 'white',
    ],

    '음식 종류/cuisine': [
        'asian', 'blue', 'chicago', 'china', 'chines', 'curri', 'filipino', 
        'greek', 'hawaiian', 'indian', 'italian', 'japanes', 'korean', 
        'mexican', 'philli', 'thai', 'vietnames', 'vegan', 'vegetarian', 'veggi',
    ],

    '서비스/경험': [
        'bartend', 'call', 'card', 'chang', 'charg', 'chef', 'clean', 'cours', 
        'deliveri', 'dine', 'dinner', 'drive', 'event', 'famili', 'fast', 
        'free', 'fun', 'groupon', 'guy', 'happi', 'hard', 'help', 'hour', 
        'husband', 'kid', 'ladi', 'late', 'line', 'live', 'long', 'manag', 
        'mani', 'music', 'night', 'noth', 'offer', 'ok', 'old', 'option', 
        'outsid', 'owner', 'parti', 'pay', 'perfect', 'person', 'play', 'pm', 
        'point', 'pub', 'reserv', 'room', 'sake', 'sampl', 'seat', 'select', 
        'server', 'show', 'song', 'sport', 'start', 'station', 'stay', 'store', 
        'tabl', 'town', 'truck', 'view', 'waiter', 'waitress', 'watch', 
        'water', 'week',
    ],

    '장소/분위기': [
        'airport', 'aria', 'ayc', 'band', 'bar', 'bellagio', 'cafe', 'casino', 
        'citi', 'club', 'cool', 'court', 'danc', 'diner', 'door', 'downtown', 
        'express', 'game', 'grill', 'hotel', 'hous', 'insid', 'island', 'king', 
        'kitchen', 'lake', 'loung', 'mall', 'market', 'mgm', 'mr', 'palm', 
        'palac', 'park', 'pool', 'prime', 'rio', 'roberto', 'rock', 'shop',
        'spot', 'steakhous', 'street', 'strip', 'venetian', 'walk', 'wine', 
        'wynn',
    ],

    '가격/가치': [
        'cheap', 'deal', 'favorit', 'half', 'local', 'pepper', 'piec', 'price', 
        'prime', 'short', 'special', 'steakhous', 'top', 'worth',
    ],
}

# 분류 결과 출력
print("===== 단어 의미 기준 분류 결과 =====\n")
all_classified = []
for cat, words in word_categories.items():
    print(f"[{cat}] ({len(words)}개)")
    print(words)
    print()
    all_classified.extend(words)

# 미분류 단어 확인
dtm_words = [col for col in brand_reviews_dtm.columns if col not in [
    'name', 'review_count', 'avg_stars', 'useful_count',
    'funny_count', 'cool_count', 'categories'
]]
unclassified = [w for w in dtm_words if w not in all_classified]
print(f"[미분류 단어] ({len(unclassified)}개)")
print(unclassified)

# 미분류 0개 확인, 최종 분류 완성
# 하지만 합계가 336개인데 최종 DTM 단어는 330개임. 
# prime, steakhous, juic, large, pepper 같이 두 카테고리에 중복 분류된 단어들이 있음
# 중복 단어는 의미가 불명확해질 수 있으니 제거하는 것이 적절하다고 판단됨.

# 중복제거
word_categories = {

    '음식/메뉴': [
        'appet', 'asada', 'bacon', 'bagel', 'bakeri', 'ball', 'bbq', 'bean',
        'beef', 'beer', 'belli', 'bowl', 'bread', 'breakfast', 'brisket',
        'broth', 'brunch', 'buffet', 'bun', 'burger', 'burrito', 'butter',
        'caesar', 'cake', 'carn', 'cart', 'cevich', 'chees', 'cheesesteak',
        'chili', 'chip', 'chocol', 'chow', 'clam', 'cocktail', 'coffe',
        'combo', 'cooki', 'corn', 'crab', 'cream', 'crepe', 'crust', 'cup',
        'deli', 'dessert', 'dim', 'dip', 'dish', 'dog', 'donut', 'duck',
        'dumpl', 'egg', 'enchilada', 'entre', 'filet', 'finger', 'fish',
        'french', 'fri', 'fruit', 'garlic', 'green', 'hash', 'hummu', 'ice',
        'ingredi', 'item', 'juic', 'lamb', 'latt', 'leg', 'lobster', 'lunch',
        'mac', 'margarita', 'meat', 'meatbal', 'milk', 'miso', 'mushroom',
        'nacho', 'noodl', 'onion', 'oyster', 'pad', 'pancak', 'pasta',
        'pastri', 'pepper', 'pho', 'pie', 'pita', 'pizza', 'plate', 'poke',
        'pork', 'portion', 'pot', 'potato', 'pull', 'ramen', 'rib', 'rice',
        'roast', 'roll', 'salad', 'salmon', 'salsa', 'salt', 'sandwich',
        'sangria', 'sashimi', 'sauc', 'sausag', 'seafood', 'shake', 'shave',
        'shrimp', 'slice', 'slider', 'smoothi', 'soup', 'spring', 'steak',
        'strawberri', 'sum', 'sushi', 'taco', 'tapa', 'tea', 'tempura',
        'tender', 'teriyaki', 'toast', 'tofu', 'tomato', 'tortilla', 'truffl',
        'tuna', 'turkey', 'waffl', 'wing', 'wonton', 'wrap', 'york',
    ],

    '음식 특성': [
        'authent', 'bake', 'cold', 'cook', 'crispi', 'cut', 'decent', 'deep',
        'differ', 'dri', 'excel', 'fill', 'fresh', 'healthi', 'home', 'hot',
        'huge', 'larg', 'light', 'perfect', 'qualiti', 'quick', 'red',
        'season', 'size', 'skewer', 'smoke', 'sour', 'spice', 'spici',
        'style', 'sweet', 'white',
    ],

    '음식 종류/cuisine': [
        'asian', 'blue', 'chicago', 'china', 'chines', 'curri', 'filipino',
        'greek', 'hawaiian', 'indian', 'italian', 'japanes', 'korean',
        'mexican', 'philli', 'thai', 'vietnames', 'vegan', 'vegetarian', 'veggi',
    ],

    '서비스/경험': [
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
    ],

    '장소/분위기': [
        'airport', 'aria', 'ayc', 'band', 'bar', 'bellagio', 'cafe', 'casino',
        'citi', 'club', 'cool', 'court', 'danc', 'diner', 'door', 'downtown',
        'express', 'game', 'grill', 'hotel', 'hous', 'insid', 'island', 'king',
        'kitchen', 'lake', 'loung', 'mall', 'market', 'mgm', 'mr', 'palm',
        'palac', 'park', 'pool', 'prime', 'rio', 'roberto', 'rock', 'shop',
        'spot', 'steakhous', 'street', 'strip', 'venetian', 'walk', 'wine',
        'wynn',
    ],

    '가격/가치': [
        'cheap', 'deal', 'favorit', 'half', 'local', 'piec', 'price',
        'short', 'special', 'top', 'worth',
    ],
}

# 분류 결과 출력
print("===== 단어 의미 기준 분류 결과 =====\n")
all_classified = []
for cat, words in word_categories.items():
    print(f"[{cat}] ({len(words)}개)")
    print(words)
    print()
    all_classified.extend(words)

# 중복 확인
from collections import Counter
duplicates = [w for w, cnt in Counter(all_classified).items() if cnt > 1]
print(f"[중복 단어] ({len(duplicates)}개): {duplicates}")

# 미분류 단어 확인
dtm_words = [col for col in brand_reviews_dtm.columns if col not in [
    'name', 'review_count', 'avg_stars', 'useful_count',
    'funny_count', 'cool_count', 'categories'
]]
unclassified = [w for w in dtm_words if w not in all_classified]
print(f"\n[미분류 단어] ({len(unclassified)}개): {unclassified}")

# 중복제거 완료 
# 합계 330개로 최종 DTM 단어 수와 일치함. 분류 완성

# 분류 결과 저장
word_category_df = pd.DataFrame([
    {'term': word, 'category': cat}
    for cat, words in word_categories.items()
    for word in words
])
word_category_df = word_category_df.sort_values(by=['category', 'term']).reset_index(drop=True)
word_category_df.to_csv(f"{PATH_to_save}/word_categories_restaurants_nv.csv", index=False, encoding='utf-8-sig')
print(f"저장 완료: word_categories_restaurants_nv.csv")
print(word_category_df.groupby('category')['term'].count())

##########=================================
# 4단계. TF-IDF 계산 및 저장
from sklearn.feature_extraction.text import TfidfTransformer

#=================================
# TF-IDF 저장
#=================================
def create_and_save_tfidf_perBrand(dtm_file_name, apply_l2):
    df = pd.read_csv(f"{PATH_to_save}/{dtm_file_name}.csv")
    meta_cols_pool = ['name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories']
    meta_cols = [col for col in df.columns if col in meta_cols_pool]
    dtm_cols = [col for col in df.columns if col not in meta_cols]
    df_meta = df[meta_cols]
    df_dtm = df[dtm_cols]
    tfidf = TfidfTransformer(norm='l2' if apply_l2 else None)
    dtm_tfidf = tfidf.fit_transform(df_dtm)
    df_tfidf = pd.DataFrame(dtm_tfidf.toarray(), columns=dtm_cols).round(5)
    df_tfidf = pd.concat([df_meta, df_tfidf], axis=1)
    suffix = '_tfidf_l2' if apply_l2 else '_tfidf'
    df_tfidf.to_csv(f"{PATH_to_save}/{dtm_file_name}{suffix}.csv", index=False, encoding='utf-8-sig')
    print(f"저장 완료: {dtm_file_name}{suffix}.csv")
    return df_tfidf

if __name__ == "__main__":
    dtm_file_name = 'reviews_restaurants_nv_perBrand_0.1_0.9_0.3_10_dtm'
    create_and_save_tfidf_perBrand(dtm_file_name, apply_l2=False)
    create_and_save_tfidf_perBrand(dtm_file_name, apply_l2=True)