import sys
sys.path.append('analysis-tools')

# !pip install tabulate switch parse missingno
# !pip install openpyxl
from analysis_tools.common import *


class PATH:
    root   = abspath(dirname(__file__))
    input  = join(root, 'input')
    output = join(root, 'output')


def convert_dtypes(data):
    cat_features = data.select_dtypes('object').columns
    num_features = data.columns.drop(cat_features)
    data[cat_features] = data[cat_features].astype('category')
    data[num_features] = data[num_features].astype(np.float32)
