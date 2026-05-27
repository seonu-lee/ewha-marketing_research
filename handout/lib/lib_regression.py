import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#-------------------------------------
# vif 계산
#-------------------------------------
def calc_vif(X):
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] # 변수(word)별 vif 계산. variance_inflation_factor(X.values, i) - i번재 변수에 대해 vif 계산

    vif_df = pd.DataFrame({
        "variable" : X.columns,
        "VIF" : vif
    })
    vif_df = vif_df[vif_df["variable"] != "const"] # 상수항(const) 행 제거 (해석 대상 아님)
    vif_df = vif_df.sort_values(by="VIF", ascending=False).reset_index(drop=True) # VIF 내림차순 정렬
    return vif_df

#-------------------------------------
# 회귀 분석
#-------------------------------------    
def reg_analysis(y, X_scaled, w=""):

    ### 회귀 분석
    # weight 벡터가 주어지면 WLS 적용하고, 주어지지 않으면 OLS 적용함
    if len(w)==0: 
        reg_result = sm.OLS(y, X_scaled).fit(cov_type="HC3") # OLS, robust std error 적용
    else:
        reg_result = sm.WLS(y, X_scaled, weights=w).fit(cov_type="HC3") # WLS, robust standard error 적용

    ### 계수, 표준 오차, p-value 추출
    reg_result_df = pd.DataFrame({
        "coef": reg_result.params,
        "std_err": reg_result.bse,
        "p_value": reg_result.pvalues
    }).round(3) 
    
    return reg_result, reg_result_df   

