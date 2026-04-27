
import pandas as pd


### 조건 설정
PATH_to_data = PATH_to_data = r"C:\Users\seonu\Documents\ewha-marketing_research\session4_dtm\results"
meta_cols_pool = ['user_id', 'name', 'review_count', 'avg_stars', 'useful_count', 'funny_count', 'cool_count', 'categories'] # meta col으로 사용될 수있는 것들은 모두 포함 

# ────────────────────────────────────────────────────────────
### 브랜드 레벨, 조건에 맞는 dtm 데이터 추출
def filtering_dtm_at_brand_level(input_data_filtering_conditions):

    '''
    Parameters
    ----------
    데이터 필터링 조건 dictionary
    - input_file_name: 사용할 dtm 데이터 파일 이름
    - remove_brand_w_word_in_name: 브랜드 이름 자체에 워드가 들어있는 샘플 제거 여부
    - brand_categories_slted: 카테고리 기준 브랜드 필터링
    - words_to_delete: 제거할 워드
    - words_to_include_exclusively: 사용할 워드 지정 (여기에 포함된 단어만 사용)

    Returns
    -------
    필터링된 dtm 데이터
    '''

    # input data 필터링 조건
    input_file_name = input_data_filtering_conditions['input_file_name']
    remove_brand_w_word_in_name = input_data_filtering_conditions['remove_brand_w_word_in_name']
    brand_categories_slted = input_data_filtering_conditions['brand_categories_slted']
    words_to_delete = input_data_filtering_conditions['words_to_delete']
    words_to_include_exclusively = input_data_filtering_conditions['words_to_include_exclusively']

    #=================================
    # 0. 조건설정 및 데이터 불러오기 
    #=================================
    ### 데이터 불러오기
    df = pd.read_csv(f"{PATH_to_data}/{input_file_name}.csv") # dtm 데이터 불러오기

    # meta columns
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.

    #=================================
    # 전처리 - 브랜드, 워드 필터링
    #=================================
    ### 전체 브랜드 + 전체 words
    # df_slted = df.set_index('name').copy() # 전체 브랜드 데이터
    df_slted = df.copy() # 전체 브랜드 데이터
    words_slted = [word for word in df_slted.columns if word not in meta_cols] # 메타컬럼을 제외한 전체 words
    # print('**full_word_list**\n', words_slted) # 참고용


    #--------------------------
    ### 워드 필터링
    # 제거할 워드 필터링 
    if len(words_to_delete) > 0:
        words_slted = [word for word in words_slted if word not in words_to_delete]

    # 사용할 워드 지정
    if len(words_to_include_exclusively) > 0:
        words_slted = [word for word in words_slted if word in words_to_include_exclusively] 


    #--------------------------
    ### 브랜드 필터링
    # 카테고리 기준 브랜드 필터링
    if len(brand_categories_slted) > 0:
        df_slted = df_slted[df_slted['categories'].str.contains('|'.join(brand_categories_slted))]

    # 브랜드 이름 자체에 워드가 들어있는 샘플 제거 
    if remove_brand_w_word_in_name == True:
        df_slted = df_slted[~df_slted['name'].str.contains('|'.join(words_slted), regex=True)] 


    #--------------------------
    ### 최종 데이터
    word_cols = [word for word in words_slted if word in df_slted.columns] # 이미 기존 컬럼에 있는 데이터만 추출해서 필요없지만 한번더   
    cols_to_include = meta_cols + word_cols # 분리해두었던 meta col과 합체
    data_w_meta_cols = df_slted[cols_to_include].reset_index(drop=True)

    return data_w_meta_cols


# ────────────────────────────────────────────────────────────
### user-brand 레벨, 조건에 맞는 dtm 데이터 추출
def filtering_dtm_at_user_brand_level(input_data_filtering_conditions):

    # input data 필터링 조건
    input_file_name = input_data_filtering_conditions['input_file_name']
    words_to_delete = input_data_filtering_conditions['words_to_delete']
    words_to_include_exclusively = input_data_filtering_conditions['words_to_include_exclusively']

    df = pd.read_csv(f"{PATH_to_data}/{input_file_name}.csv")

    # meta columns
    meta_cols = [col for col in df.columns if col in meta_cols_pool] # 데이터의 컬럼들중 meta col pool 에 있는 것들을 meta col로 설정. 데이터에 meta col이 다를 수 있기때문에 이렇게함.


    #=================================
    # 전처리 - 브랜드, 키워드 필터링
    #=================================
    ### 전체 브랜드 + 전체 words
    # df_slted = df.set_index('name').copy() # 전체 브랜드 데이터
    df_slted = df.copy() # 전체 브랜드 데이터
    words_slted = [word for word in df_slted.columns if word not in meta_cols] # 메타컬럼을 제외한 전체 words
    # print('**full_word_list**\n', words_slted) # 참고용


    #--------------------------
    ### 키워드 필터링
    # 제거할 키워드 필터링 
    if len(words_to_delete) > 0:
        words_slted = [word for word in words_slted if word not in words_to_delete]

    # 사용할 키워드 지정
    if len(words_to_include_exclusively) > 0:
        words_slted = [word for word in words_slted if word in words_to_include_exclusively] 


    #--------------------------
    ### 최종 데이터
    word_cols = [word for word in words_slted if word in df_slted.columns] # 이미 기존 컬럼에 있는 데이터만 추출해서 필요없지만 한번더   
    cols_to_include = meta_cols + word_cols # 분리해두었던 meta col과 합체
    data_w_meta_cols = df_slted[cols_to_include].reset_index(drop=True)

    return data_w_meta_cols
