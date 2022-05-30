import sys
sys.path.append('analysis-tools')

from analysis_tools.common import *
from analysis_tools.eda import *
from analysis_tools.preprocessing import *


class PATH:
    root   = abspath(dirname(__file__))
    input  = join(root, 'input')
    output = join(root, 'output')
    cache  = join(root, 'cache')
    embedding = join(root, 'embedding')

    for path in (root, input, output, cache, embedding):
        os.makedirs(path, exist_ok=True)


# Load dataset
def load_data():
    return pd.read_csv(join(PATH.input, 'accepted_2007_to_2018Q4.csv'), index_col=0)
def select_data(data, target):
    cols = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
            'home_ownership', 'annual_inc', 'verification_status', 'loan_status', 'purpose', 'title', 'zip_code',
            'addr_state', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            'initial_list_status', 'application_type', 'mort_acc', 'pub_rec_bankruptcies']
    data = data[cols]
    return data[data[target].isin(['Fully Paid', 'Current', 'Charged Off'])]
def split_data(data, target, test_size):
    from sklearn.model_selection import train_test_split

    idxs_notnull = data.dropna().index
    data_notnull = data[data.index.isin(idxs_notnull)]
    data_isnull  = data[~data.index.isin(idxs_notnull)]
    train_full_data_notnull, test_data = train_test_split(data_notnull, stratify=data_notnull[target], test_size=int(len(data)*test_size), random_state=RANDOM_STATE)
    train_full_data_notnull, val_data  = train_test_split(train_full_data_notnull, stratify=train_full_data_notnull[target], test_size=int(len(data)*test_size), random_state=RANDOM_STATE)
    train_data = pd.concat([train_full_data_notnull, data_isnull])
    train_full_data = pd.concat([train_data, val_data])
    datas = {'train': train_data, 'val': val_data, 'test': test_data, 'train_full': train_full_data}
    print("Data size:", {key: len(val) for key, val in datas.items()})
    return datas


# Preprocessing
## Imputing
@T
def impute_data(data_):
    data = data_.copy()
    for col in ('emp_title', 'emp_length', 'title', 'zip_code', 'dti', 'annual_inc', 'revol_util', 'revol_bal', 'mort_acc', 'pub_rec_bankruptcies'):
        eval(f"impute_{col}")(data)
    return data
def impute_emp_title(data):
    data['emp_title'].fillna('other', inplace=True)
def impute_emp_length(data):
    data['emp_length'].fillna('other', inplace=True)
def impute_title(data):
    data.drop(columns=['title'], inplace=True)
def impute_zip_code(data):
    data.drop(columns=['zip_code'], inplace=True)
def impute_dti(data):
    col = 'dti'
    data[f'{col}_nan'] = data[col].map(lambda v: 1 if pd.isnull(v) else 0)
    data[col].fillna(0.1, inplace=True)
def impute_annual_inc(data):
    data['annual_inc_zero'] = data['annual_inc'].map(lambda v: 1 if v == 0 else 0)
def impute_revol_util(data):
    col = 'revol_util'
    data[f'{col}_zero'] = data[col].map(lambda v: 1 if v == 0 else 0)
    data[f'{col}_nan']  = data[col].map(lambda v: 1 if pd.isnull(v) else 0)
    data[col].fillna(0, inplace=True)
def impute_revol_bal(data):
    data['revol_bal_zero'] = data['revol_bal'].map(lambda v: 1 if v == 0 else 0)
def impute_mort_acc(data):
    col = 'mort_acc'
    data[f'{col}_nan'] = data[col].map(lambda v: 1 if pd.isnull(v) else 0)
    data[col].fillna(0.1, inplace=True)
def impute_pub_rec_bankruptcies(data):
    col = 'pub_rec_bankruptcies'
    data[f'{col}_nan'] = data[col].map(lambda v: 1 if pd.isnull(v) else 0)
    data[col].fillna(0.1, inplace=True)
