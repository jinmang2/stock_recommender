import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import pickle
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

def load_raw_data(filename):
    """
    Description
    raw_data 호출함수

    Use Library 'pickle'

    Argument
    filename : string, such as './data/raw_data0824.pickle'

    Return
    raw_data : Your file data
    """
    with open(filename, 'rb') as handle:
        raw_data = pickle.load(handle)
    return raw_data

def load_code_name_data(filename, return_dict=False, encoding='cp949'):
    """
    Description
    kospi200 파일 호출 함수
    return_dict = True로 설정하여 name_dict와 code_dict를 return할 수 있다.

    Use Library 'Pandas'

    Argument
    filename : string, such as './data/raw_data0824.pickle'
    return_dict : Boolean, if True then function returns name_dict and code_dict
    encoding : string, default value is cp949

    Return
    kospi200 : pd.DataFrame object
    (if return_dict:)
    name_dict : dict, key : stock_name & value : stock_code
    code_dict : dict, key : stock_code & value : stock_name
    """
    kospi200 = pd.read_csv(filename, encoding=encoding)
    kospi200 = kospi200[kospi200.columns[:2]]
    kospi200["종목코드"] = kospi200["종목코드"].map('{:06d}'.format)
    if return_dict:
        name_dict = { i : j for i, j in zip(kospi200["종목명"], kospi200["종목코드"]) }
        code_dict = { j : i for i, j in zip(kospi200["종목명"], kospi200["종목코드"]) }
        return kospi200, name_dict, code_dict
    return kospi200

def load_party_data(filename):
    """
    Description
    income과 sector별 주식 분류 데이터 로드 함수

    Use Library 'Pandas'

    Argument
    filename : string, such as './data/raw_data0824.pickle'

    Return
    party_df : pd.DataFrame object
    """
    party_df = pd.read_csv(filename)
    party_df["종목코드"] = party_df["종목코드"].map('{:06d}'.format)
    return party_df

def add_feature():
    pass

def shift_dataset_make_new(raw_data, feature_list, target, shift):
    """
    Description
    특정 데이터셋을 지정한 feature를 일정 기간만큼 shift시키는 함수
    본래 target을 list로 받아 shift하지 않을 column을 지정할 예정이었으나
      현재 구현된 바로는 target은 string으로 받는다.

    Argument
    raw_data : dict which has pd.DataFrame as element
    feature_list : sequence data type, set of feature
    target : string, target values
    shift : int, adjusting shift DataFrame

    Return
    data : dict which has pd.DataFrame as element
    """
    data = {}
    for code in raw_data.keys():
        data[code] = pd.DataFrame()
        for feature in feature_list:
            data[code]["sh{}_{}".format(shift, feature)] = \
                    raw_data[code][feature].shift(shift)
        data[code][target] = raw_data[code][target]
        data[code] = data[code].dropna(axis=0)
    return data

def adjust_window(data, start, end):
    """
    Description
    데이터프레임 날짜 조정 함수

    Argument
    data : dict which has pd.DataFrame as element
    start : string, such as '2012-01-01'
    end : string, such as '2018-07-31'

    Return
    df : dict which has pd.DataFrame as element
    """
    df = {}
    for code in data.keys():
        df[code] = data[code].loc[start:end].copy()
        df[code] = df[code].dropna(axis=0)
    return df

def train_test_split(df, train_ratio, X=None, y=None, random = False, dtrain = False):
    """
    Description
    한 주식에 대한 train과 test set을 분리하는 함수
    dtrain = True로 설정할 경우 _idx 데이터를 return안하게 만들 수 있다.
    random = True로 설정할 경우 임의로 index를 설정, train과 test를 섞는다.

    Argument
    df : DataFrame object
    train_ratio : float, in range(0,1)
    X : sequence data, such as list, tuple (Train Features)
    y : str (target value)
    random : boolean
    dtrain : boolean

    Return
    if dtrain == False:
        train : 학습시킬 Feature data
        train_idx : 지도학습의 Y value
        test : test 검증할 Feature data
        test_idx : test set의 Y value
    else:
        train : 학습시킬 data (both X and y)
        test : test 검증할 data (both X and y)
    """
    train_size = int(len(df)*train_ratio)
    if random:
        shuffle_indicies = np.random.permutation(len(df))
        train_indicies = shuffle_indicies[:train_size]
        test_indicies = shuffle_indicies[train_size:]
    else:
        normal_indicies = np.arange(len(df))
        train_indicies = normal_indicies[:train_size]
        test_indicies = normal_indicies[train_size:]
    if dtrain:
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]
        return train, test
    else:
        train = df.iloc[:train_size][X]
        train_idx = df.iloc[:train_size][y]
        test = df.iloc[train_size:][X]
        test_idx = df.iloc[train_size:][y]
        return train, train_idx, test, test_idx

def multi_train_test_split(data, train_ratio, X=None, y=None, random = False, dtrain = True):
    """
    Description
    다수의 주식에 대하여 train과 test set을 나누는 함수
    dtrain = True로 설정할 경우 _idx 데이터를 return안하게 만들 수 있다.
    random = True로 설정할 경우 임의로 index를 설정, train과 test를 섞는다.

    Argument
    df : DataFrame object
    train_ratio : float, in range(0,1)
    X : sequence data, such as list, tuple (Train Features)
    y : str (target value)
    random : boolean
    dtrain : boolean

    Return
    if dtrain == False:
        train : 학습시킬 Feature data
        train_idx : 지도학습의 Y value
        test : test 검증할 Feature data
        test_idx : test set의 Y value
    else:
        train : 학습시킬 data (both X and y)
        test : test 검증할 data (both X and y)
    """
    train = {}
    train_idx = {}
    test = {}
    test_idx = {}
    if dtrain:
        for code, df in data.items():
            sub_train, sub_test = train_test_split(df, train_ratio, X, y, random, dtrain)
            train[code] = sub_train
            test[code] = sub_test
        return train, test
    else:
        for code, df in data.items():
            X_train, y_train, X_test, y_test = train_test_split(df, train_ratio, X, y, random, dtrain)
            train[code] = X_train
            train_idx[code] = y_train
            test[code] = X_test
            test_idx[code] = y_test
        return train, train_idx, test, test_idx

def apply_standard_scale(std_scaler, train, test, predictors):
    """
    Description
    sklearn의 StandardScaler로 scale을 조절하는 함수.
    변환하고 싶은 feature list를 predictors로 넣어 스케일을 조절

    Argument
    std_scaler : StandardScaler object of scikit_learn
    train : pd.DataFrame
    test : pd.DataFrame
    predictors : list

    Return
    sub_train : 한 개의 dataframe object
    sub_test : 한 개의 dataframe object
    """
    std_scaler.fit(train[predictors].values)
    sub_train = std_scaler.transform(train[predictors].values)
    sub_test = std_scaler.transform(test[predictors].values)
    return sub_train, sub_test

def multi_apply_standard_scale(train, test, predictors):
    """
    Description
    apply_standard_scale 함수를 다수의 주식에 적용하는 함수

    Argument
    std_scaler : StandardScaler object of scikit_learn
    train : dict
    test : dict
    predictors : list
    """
    std_scaler = StandardScaler()
    for code in train.keys():
        sub_train, sub_test = apply_standard_scale(std_scaler, train[code], test[code], predictors)
        train[code][predictors] = sub_train
        test[code][predictors] = sub_test

def modelfit_with_score(X_train, y_train, X_test, y_test, models, model_name):
    score = {} #주식별로 모델별 Test Score를 저장할 dict 생성
    confusion = {} #주식별로 모델별 Confusion Matrix를 저장할 dict 생성
    for clf, name in zip(models, model_name):
        model_score = [] #각 모델별 score가 저장될 list
        confusion_mat = [] #각 모델별 confusion matrix가 저장될 list
        clf.fit(X_train, y_train) #fitting
        y_pred = clf.predict(X_test) #predict
        # if str(clf)[:3] == 'XGB':
        #     print(clf.n_estimators)
        try:
            y_prob = clf.predict_proba(X_test)
        except NotFittedError:
            clf.probability = True
            y_prob = clf.predict_proba(X_test)
        # 모델 별 score 및 confusion matrix를 list에 저장
        model_score.append(metrics.accuracy_score(y_test, y_pred))
        model_score.append(metrics.precision_score(y_test, y_pred))
        model_score.append(metrics.recall_score(y_test, y_pred))
        model_score.append(metrics.f1_score(y_test, y_pred))
        model_score.append(metrics.roc_auc_score(y_test, y_prob[:,1]))
        confusion_mat.append(metrics.confusion_matrix(y_test, y_pred))
        # 모델별 score및 confusion matrix를 모델별로 저장
        score[name] = model_score
        confusion[name] = confusion_mat
    return score, confusion

def multi_modelfit_with_score(train, test, models, model_name, X=None, y=None,\
        train_idx=None, test_idx=None, dtrain=True):
    result = {} #Test 결과를 저장할 Dictionary 자료구조 생성
    conf_mx = {} #Confusion Matrix를 저장한 Dictionary 자료구조 생성
    for i in train.keys():
        if dtrain:
            X_train = train[i][X].values
            y_train = train[i][y].values
            X_test = test[i][X].values
            y_test = test[i][y].values
        else:
            X_train = train[i].values
            y_train = train_idx[i].values
            X_test = test[i].values
            y_test = test_idx[i].values
        score, confusion = modelfit_with_score(X_train, y_train, X_test, y_test, models, model_name)
        # 주식별 score와 confusion matrix를 저장
        result[i] = score
        conf_mx[i] = confusion
    # 결과값을 return으로 받음
    return result, conf_mx




"""굳이 사용안해도 되는 함수"""
def is_period_less_than_2_year(raw_data):
    del_li = [key for key, df in raw_data.items() if len(df) < 252*2]
    return del_li

def is_duplicated_more_than_20(raw_data):
    del_li = [key for key, df in raw_data.items() if df.loc['2012-01-01':'2018-07-31',['Close','Volume','Open']].duplicated().sum() > 20]
    return del_li
