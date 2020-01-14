from moon_code import *
from xgb_modelfit import *


raw_data = load_raw_data('raw_data_20181008.pickle')

total_stock_list = list(raw_data.keys())

nan_li = ['BGF', 'BGF리테일', 'HDC현대산업개발', 'LIG넥스원',
          'SK케미칼', '넷마블', '두산밥캣', '만도', '삼성물산',
          '삼성바이오로직스', '삼성에스디에스', '오리온',
          '이노션', '종근당', '코스맥스', '쿠쿠홀딩스',
          '현대로템', '현대중공업지주', '효성중공업', '동아에스티']

for stock in nan_li:
    del raw_data[stock]

data = {}
for stock, df in raw_data.items():
    dff = df.copy()
    dff.columns = pd.MultiIndex.from_arrays([[stock]*6, df.columns])
    data[stock] = dff

# with open('raw_data_20181023.pickle', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# data = load_raw_data('raw_data_20181023.pickle')

# Get Panel Data
p = pd.DataFrame()
for stock, df in data.items():
    p = pd.concat((p, df), axis=1)

q = p.T.swaplevel().sortlevel().T

def get_multi_index(col_name):
    return pd.MultiIndex.from_product([[col_name],list(raw_data.keys())])

def add_feature(q, col_name, expr):
    ind = get_multi_index(col_name)
    try:
        df = pd.DataFrame(expr.values, columns=ind, index=q.index)
    except AttributeError:
        df = pd.DataFrame(expr, columns=ind, index=q.index)
    q = pd.concat((q, df), axis=1)
    return q

# Add Feature
q = add_feature(q, 'pct_change', q.Close.pct_change())
for i in range(3, 31, 2):
    q = add_feature(q, "close_ma_{}".format(i), q.Close.rolling(window=i).mean())
    q = add_feature(q, "volume_ma_{}".format(i), q.Volume.rolling(window=i).mean())
for i in [7, 21, 30, 60, 90, 91, 100, 120]:
    q = add_feature(q, "his_vol_{}".format(i), q['pct_change'].rolling(window=i).std()*(252**0.5))
for i in [1, 3, 5, 7, 10, 12, 14]:
    q = add_feature(q, 'returnsClosePrevRaw{}'.format(i), np.log(q.Open / q.Close.shift(i)))
    q = add_feature(q, 'returnsOpenPrevRaw{}'.format(i), np.log(q.Open / q.Open.shift(i)))
for i in [1, 3, 5, 7, 10]:
    q = add_feature(q, 'Y3_{}'.format(i),
                    np.where(q.Close.rolling(i).mean() - q.Close.shift(1).rolling(i).mean() >=0, 1, 0))
    for shift in [1, 2, 3, 4, 5]:
        q = add_feature(q, 'sh{}_Y3_{}'.format(shift, i),
                    q['Y3_{}'.format(i)].shift(-1*shift))

# Adjust Time Period
q2 = q.loc['2013-07-01':'2018-07-01'].copy()

# Generate Column and Std_Scaler
total_col = list(q2.columns.get_level_values(0).unique())
target_li = [col for col in total_col if col.find('Y') != -1]
X_li = [col for col in total_col if col.find('Y') == -1]
scl_q = q2.copy()
scl_col = pd.MultiIndex.from_product([['scl_'+i for i in list(scl_q.columns.get_level_values(0).unique())],
                            list(scl_q.columns.get_level_values(1).unique())])
scl_q.columns = scl_col
scl_X_li = list(scl_q.columns.get_level_values(0).unique())
scl_li = scl_X_li.copy()
del_li = []
for ix, i in enumerate(scl_X_li):
    if (i.find('returns') != -1) or (i.find('his') != -1) or (i.find('pct') != -1):
        scl_li[ix] = scl_X_li[ix][4:]
    elif (i.find('Y') != -1):
        del_li.append(i)
scl_X_li = sorted(list(set(scl_X_li) - set(del_li)))
scl_li = sorted(list(set(scl_li) - set(del_li)))
origin_X_li = [i for i in X_li if i.find('returns') != 0]
origin_scl_X_li = [i for i in scl_li if i.find('returns') != 0]

scl_q = scl_q[scl_X_li]

# Train Test Split
train1, test1 = train_test_split(q2, 0.8, dtrain=True)
train2, test2 = train_test_split(scl_q, 0.8, dtrain=True)

def apply_standard_scale(train, test, predictors):
    std_scaler = StandardScaler()
    std_scaler.fit(train[predictors].values)
    sub_train = std_scaler.transform(train[predictors].values)
    sub_test = std_scaler.transform(test[predictors].values)
    sub_train = pd.DataFrame(sub_train, columns = train.columns, index = train.index)
    sub_test = pd.DataFrame(sub_test, columns = test.columns, index = test.index)
    return sub_train, sub_test

train2, test2 = apply_standard_scale(train2, test2, scl_X_li)

train, test = pd.concat((train1, train2), axis=1), pd.concat((test1, test2), axis=1)

train_df = train.T.swaplevel().sortlevel().T
test_df = test.T.swaplevel().sortlevel().T

# store = (train, test, target_li, X_li, scl_X_li, scl_li)
#
# with open("store_학습준비파일_20181023.pickle", 'wb') as handle:
#     pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)
