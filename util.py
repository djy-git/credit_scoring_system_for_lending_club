import sys
sys.path.append('analysis-tools')

from analysis_tools.common import *
from analysis_tools.eda import *
from analysis_tools.preprocessing import *


class PATH:
    root   = abspath(dirname(__file__))
    input  = join(root, 'input')
    output = join(root, 'output')


# Load dataset
def load_data(target):
    data = pd.read_csv(join(PATH.input, 'accepted_2007_to_2018Q4.csv'), index_col=0)
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
    print(len(train_data), len(val_data), len(test_data))
    return train_data, val_data, test_data, train_full_data


# Imputing
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


# Preprocessing
@T
def feature_engineering(data_, emp_title_thr_cumsum=0.5):
    data = data_.copy()
    data['int_rate']    = data['int_rate'].map(lambda v: np.log(max(v, 0.1)))
    data['installment'] = data['installment'].map(lambda v: np.log(max(v, 0.1)))
    data['emp_title']   = data['emp_title'].map(lambda s: s.lower().strip())
    data['emp_title']   = data['emp_title'].str.replace('[-_&.,]', '')  # time-consuming
    vals = (data['emp_title'].value_counts().cumsum() / len(data)) <= emp_title_thr_cumsum
    data['emp_title'] = data['emp_title'].map(lambda s: s if s in vals[vals].index else 'other')
    print(data['emp_title'].value_counts().index)
    data = data.join(get_embedding(data['emp_title'], embedding_length=50, max_len=4))
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
@T
def get_embedding(texts, embedding_length=50, max_len=None):
    assert embedding_length in [50, 100, 200, 300], f'embedding_length({embedding_length}) should be in [50, 100, 200, 300]'
    embedding_dict_path = f'glove.6B.{embedding_length}d.txt'
    if not exists(embedding_dict_path):
        from urllib.request import urlretrieve
        import zipfile
        filename = "glove.6B.zip"
        urlretrieve(f"http://nlp.stanford.edu/data/{filename}", filename=filename)
        with zipfile.ZipFile(filename) as zf:
            zf.extractall()
    embedding_dict = {}
    with open(embedding_dict_path, encoding='utf8') as f:
        for line in f:
            word, *emb = line.split()
            embedding_dict[word] = np.asarray(emb, dtype=np.float32)
    print('len(embedding_dict):', len(embedding_dict))

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences  = tokenizer.texts_to_sequences(texts)
    vocab_size = len(tokenizer.word_index) + 1  # +1: padding
    max_len    = max_len if max_len else max(len(l) for l in sequences)
    print('max_len:', max_len, '| vocab_size:', vocab_size)

    from tensorflow.keras.layers import Embedding, Flatten
    from tensorflow.keras import Sequential
    embedding_matrix = np.zeros((vocab_size, embedding_length))
    for word, index in tokenizer.word_index.items():
        if (vector_value := embedding_dict.get(word)) is not None:
            embedding_matrix[index] = vector_value
    model = Sequential([Embedding(*embedding_matrix.shape, weights=[embedding_matrix], input_length=max_len, trainable=False), Flatten()])
    X = pad_sequences(sequences, maxlen=max_len, padding='post')
    emb = model.predict(X)
    return pd.DataFrame(emb, columns=[f"{texts.name}_emb_{i}" for i in range(emb.shape[1])], index=texts.index)
def get_preprocessor(ord_cols, nom_cols, num_cols):
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer

    return make_pipeline(
        FunctionTransformer(impute_data),
        FunctionTransformer(feature_engineering),

        make_column_transformer(
            (OrdinalEncoder(), ord_cols),
            (OneHotEncoder(), nom_cols),
            (StandardScaler(), num_cols),
            remainder='passthrough', n_jobs=-1),

        dtype_converter(np.float32)
    )
