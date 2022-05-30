from util import *


## Feature engineering
@T
def feature_engineering(data_, emp_title_thr_cumsum=0.5):
    data = data_.copy()
    data['int_rate']    = data['int_rate'].map(lambda v: np.log(max(v, 0.1)))
    data['installment'] = data['installment'].map(lambda v: np.log(max(v, 0.1)))
    data['emp_title']   = data['emp_title'].map(lambda s: s.lower().strip())
    data['emp_title']   = data['emp_title'].str.replace('[-_&.,]', '')  # time-consuming
    vals = (data['emp_title'].value_counts().cumsum() / len(data)) <= emp_title_thr_cumsum
    data['emp_title'] = data['emp_title'].map(lambda s: s if s in vals[vals].index else 'other')
    print("major emp_title:", data['emp_title'].value_counts().index.values[:10])
    data = data.join(get_embedding(data['emp_title'], PATH.embedding, embedding_length=50, max_len=4))
    data.drop(columns=['emp_title'], inplace=True)
    data['emp_length'][data['emp_length'] == '10+ years'] = '91 years'
    data['emp_length'][data['emp_length'] == '< 1 year'] = '0 year'
    data['annual_inc'] = data['annual_inc'].map(lambda v: np.log(max(v, 0.1)))
    data['annual_inc_zero_x_annual_inc'] = data['annual_inc_zero'] * data['annual_inc']
    data['dti'] = data['dti'].map(lambda v: np.log(max(v, 0.1)))
    data['dti_nan_x_dti'] = data['dti_nan'] * data['dti']
    data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line']).map(lambda d: int(d.year) + float(d.month) / 12)
    data['revol_bal'] = data['revol_bal'].map(lambda v: np.log(max(v, 0.1)))
    data['revol_bal_zero_x_revol_bal'] = data['revol_bal_zero'] * data['revol_bal']
    data['revol_util_zero_x_revol_util'] = data['revol_util_zero'] * data['revol_util']
    data['revol_util_nan_x_revol_util']  = data['revol_util_nan'] * data['revol_util']
    data['total_acc'] = data['total_acc'].map(lambda v: np.log(max(v, 0.1)))
    data['mort_acc_nan_x_mort_acc'] = data['mort_acc_nan'] * data['mort_acc']
    data['pub_rec_bankruptcies_nan_x_pub_rec_bankruptcies'] = data['pub_rec_bankruptcies_nan'] * data['pub_rec_bankruptcies']
    return data
def get_ord_nom_num_cols():
    ord_cols = ('term', 'grade', 'sub_grade', 'emp_length', 'initial_list_status', 'application_type')
    nom_cols = ('home_ownership', 'verification_status', 'purpose', 'addr_state')
    num_cols = ('loan_amnt', 'int_rate', 'installment', 'earliest_cr_line', 'annual_inc', 'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies')
    return ord_cols, nom_cols, num_cols


## Pipeline
def get_imputer():
    from sklearn.preprocessing import FunctionTransformer
    return FunctionTransformer(impute_data)
def get_fe_processor():  # feature engineering
    from sklearn.pipeline import make_pipeline
    return make_pipeline(get_static_fe_processor(), get_dynamic_fe_processor())
def get_static_fe_processor():
    from sklearn.preprocessing import FunctionTransformer
    return FunctionTransformer(feature_engineering)
def get_dynamic_fe_processor():
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
    ord_cols, nom_cols, num_cols = get_ord_nom_num_cols()
    return make_column_transformer((OrdinalEncoder(), ord_cols),
                                   (OneHotEncoder(), nom_cols),
                                   (StandardScaler(), num_cols),
                                   remainder='passthrough', n_jobs=-1)
def get_preprocessor():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    return make_pipeline(get_imputer(),
                         get_fe_processor(),
                         FunctionTransformer(lambda arr: arr.astype(np.float32)))
