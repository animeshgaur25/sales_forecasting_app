import psycopg2
import pred
import flask
import random
import sqlalchemy
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from warnings import filterwarnings
from warnings import catch_warnings
from joblib import delayed
from joblib import Parallel
from multiprocessing import cpu_count
from math import sqrt
import time
from sklearn.model_selection import train_test_split
import math
import json
from os import stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()

app = flask.Flask(__name__, template_folder='templates')

def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')


# Calculating Sales per Month 
def data_preprocessing(versa_sales1, item_id):
    versa_sales1['transaction_date'] = pd.to_datetime(versa_sales1['transaction_date'], errors='coerce')
    versa_sales1['transaction_date'] = versa_sales1['transaction_date'].dropna()
    versa_sales1 = versa_sales1.drop(columns=['firm_id'])
    versa_sales1["transaction_date"] = versa_sales1.transaction_date.to_numpy().astype('datetime64[M]')
    versa_sales2 = versa_sales1.groupby(versa_sales1["transaction_date"], as_index=False).agg({'delta': np.sum})
    flag = 0
    current_time = datetime.now()
    versa_maxyear = (versa_sales2.transaction_date.max()).year
    # if current_time.year-versa_maxyear > 1:
    #     flag = -1
    #     print(flag)
    #     return -1
    r = pd.date_range(start=versa_sales2.transaction_date.min(), end=versa_sales2.transaction_date.max(), freq='MS')
    #r = pd.date_range(start=versa_sales2.transaction_date.min(), end=datetime.now())

    versa_sales3 = versa_sales2.set_index('transaction_date').reindex(r , fill_value=0).rename_axis('transaction_date').reset_index()
    versa_sales_monthly = versa_sales3.groupby(versa_sales3.transaction_date.dt.to_period("M")).agg({'delta': np.sum})
    # print(versa_sales3)
    versa_sales_monthly["date"] = versa_sales_monthly.index
    versa_sales_monthly2 = versa_sales_monthly.reset_index(inplace=True)
    versa_sales_monthly = versa_sales_monthly.drop('date', axis=1)

    versa_sales_monthly.transaction_date = versa_sales_monthly.transaction_date.map(
        str)
    versa_sales_monthly['transaction_date'] = pd.to_datetime(
        versa_sales_monthly['transaction_date'])
    versa_sm = versa_sales_monthly.set_index('transaction_date')
    print(versa_sm)
    return versa_sm


# def difference(dataset):
#     diff = list()
#     for i in range(1, len(dataset)):
#         value = dataset[i] - dataset[i - 1]
#         #print(value)
#         diff.append(value)
#     return pd.Series(diff)

def data_pts_chk(num):
    if(num < 15):
        return False
    else:
        return True


# Ho: It is non stationary
# H1: It is stationary
def adfuller_test(sales):
    try:
        result = adfuller(sales)
    except ValueError:
        return -1


    labels = ['ADF Test Statistic', 'p-value',
              '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
        return 1
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        return 0


def sarima_configs(seasonal=[0], d_params = 0):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    q_params = [0, 1, 2]
    P_params = [0, 1]
    D_params = [0, 1]
    Q_params = [0, 1]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for q in q_params:
            for P in P_params:
                for D in D_params:
                    for Q in Q_params:
                        for m in m_params:
                            cfg = [(p, d_params, q), (P, D, Q, m)]
                            models.append(cfg)
    models = random.sample(models, 200)
    return models


def sarima_forecast(history, config):
    order, sorder = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder,
                    enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def walk_forward_validation(train_data, test_data, cfg):
    predictions = list()
    # split dataset
    train, test = train_data["log_sales"], test_data["log_sales"]
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


def score_model(train_data, test_data, cfg, debug=False):
    result = None
    # convert config to a key
    key = cfg
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(train_data, test_data, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(train_data, test_data, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


def grid_search(train_data, test_data, cfg_list, parallel=False):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(train_data, test_data, cfg)
                 for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(train_data, test_data, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


def user_inp_grid_search(item_id, firm_id, versa_sm):
    # time.sleep(10)
    print("Hi")
    # return 0
    item_id = int(item_id)
    firm_id = int(firm_id)
    versa_sm1 = versa_sm
    versa_sm1['log_sales'] = np.log(versa_sm1['delta']).dropna()
    versa_sm.replace([np.inf, -np.inf],0 , inplace=True)

    # Calculating d to make sales data stationary.
    d = 0
    while True:
        stationary = adfuller_test(versa_sm1['log_sales'])
        if stationary==-1:
            print("More Sales Data is required for this item")
            break

        elif stationary == 0:
            d= d + 1
            versa_sm1 = versa_sm1.diff().dropna()

        else:
            break 

    # Splitting the data in training and test sets
    train_data, test_data = train_test_split(versa_sm1, shuffle=False, test_size=0.2)

    cfg_list = sarima_configs(seasonal=[0, 2, 3, 4, 6, 12], d_params = d)
    # grid search
    scores = grid_search(train_data, test_data, cfg_list)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
    minpara = scores[0][0]
    print(minpara)
    print(type(minpara))
    p = minpara[0][0]
    # d=cfg[0][1]
    q = minpara[0][2]
    seasonal_p = minpara[1][0]
    seasonal_d = minpara[1][1]
    seasonal_q = minpara[1][2]
    s = minpara[1][3]

    #Updating the results in DB table
    conn = psycopg2.connect(
        database="versa_db",
        user="postgres",
        password="bits123",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    #cur.execute("DELETE from forecasting_parameters WHERE inventory_item_id")
    cur.execute("UPDATE forecasting_parameters SET p = %s ,d = %s ,q = %s,seasonal_p = %s,seasonal_d = %s,seasonal_q = %s,s = %s,flag = %s WHERE inventory_item_id = %s AND firm_id = %s",
                (p, d, q, seasonal_p, seasonal_d, seasonal_q, s, 1, item_id, firm_id))
    # %{'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 's': s, 'item_id': item_id, 'firm_id': firm_id})
    conn.commit()
    cur.close()
    conn.close()
    res = pred.sales_forecast(item_id, firm_id, versa_sm)
    return "Parameter search completed!, click predict"
    # return scores[0]
    # scores[0] needs to be saved in the db.
