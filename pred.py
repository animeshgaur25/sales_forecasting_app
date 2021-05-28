
from sklearn.metrics import mean_squared_error
from warnings import filterwarnings
from warnings import catch_warnings
from joblib import delayed
from joblib import Parallel
from multiprocessing import cpu_count
from math import sqrt
from time import time
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from pandas.plotting import register_matplotlib_converters
from datetime import timedelta
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import json
import sqlalchemy
import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
register_matplotlib_converters()


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

def inverse_diff(initial_val, predictions):
    res = np.r_[initial_val, predictions].cumsum()
    res1 = pd.Series(res[1:], index= predictions.index, name="predicted_mean")
    return res1

def sales_forecast(item_id, firm_id, versa_sm):
    item_id = int(item_id)
    firm_id = int(firm_id)
    versa_sm['log_sales'] = np.log(versa_sm['delta']).dropna()
    versa_sm.replace([np.inf, -np.inf],0 , inplace=True)
    #versa_sales = pd.read_csv(r"C:\Users\prasa\Documents\programs\demo_sales_fc\data_updated22-09.csv", parse_dates=[4], index_col=0, squeeze=True, date_parser=parser)
    engine = sqlalchemy.create_engine(
        'postgresql://root:root@localhost:5432/development_master')
    query = '''
    SELECT *
    FROM forecasting_parameters
    WHERE flag = 1 AND item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' ''' % {'item_id': item_id, 'firm_id': firm_id}
    # SQL injection
    parameters = pd.read_sql_query(query, engine)
    if parameters.empty: 
        return parameters
    para = parameters[(parameters["item_id"] == item_id)]
    p = para.loc[para['item_id'] == item_id, 'p'].iloc[0]
    d = para.loc[para['item_id'] == item_id, 'd'].iloc[0]
    q = para.loc[para['item_id'] == item_id, 'q'].iloc[0]
    P = para.loc[para['item_id'] == item_id, 'seasonal_p'].iloc[0]
    Q = para.loc[para['item_id'] == item_id, 'seasonal_q'].iloc[0]
    D = para.loc[para['item_id'] == item_id, 'seasonal_d'].iloc[0]
    s = para.loc[para['item_id'] == item_id, 's'].iloc[0]
    my_order = (p, d, q)
    my_seasonal_order = (P, D, Q, s)
    if d == 0:
        model = SARIMAX(versa_sm['log_sales'], order=my_order, seasonal_order=my_seasonal_order, enforce_stationarity=False, enforce_invertibility=False )
        start = versa_sm.first_valid_index()
        end = versa_sm.last_valid_index()

    elif d==1:
        first_diff = versa_sm.diff().dropna()
        model = SARIMAX(first_diff['log_sales'], order=my_order, seasonal_order=my_seasonal_order, enforce_stationarity=False, enforce_invertibility=False )
        start = first_diff.first_valid_index()
        end = first_diff.last_valid_index()
        print(first_diff)
       
    else:
        second_diff = first_diff.diff().diff().dropna()
        model = SARIMAX(second_diff['log_sales'], order=my_order, seasonal_order=my_seasonal_order, enforce_stationarity=False, enforce_invertibility=False )
        start = second_diff.first_valid_index()
        end = second_diff.last_valid_index()
        print(second_diff)


# till here
    
    model_fit = model.fit()
    test_pred = model_fit.predict(start, end)
    pred = model_fit.forecast(24)
    predictions = pd.concat([test_pred, pred])
    months = list(predictions.index.astype(str))
    print(predictions)
    if d == 0:
        predictions[predictions < 0 ] = 0
        result = pd.concat([predictions, versa_sm], axis=1)


    else:
        predictions = inverse_diff(versa_sm['log_sales'][0], predictions)
        print(predictions)
        predictions[predictions < 0 ] = 0
        result = pd.concat([predictions, versa_sm], axis=1)

    # else:
    #     first_diff = versa_sm.diff().dropna()
    #     pred_diff= inverse_diff(first_diff['log_sales'][0], predictions)
    #     predictions= inverse_diff(versa_sm['log_sales'][0], pred_diff)
    #     print(predictions)
    #     predictions[predictions < 0 ] = 0
    #     result = pd.concat([predictions, versa_sm], axis=1)

    print("*****************************\n", result)
    result['predicted_mean'] = np.exp(result['predicted_mean'])
    result['forecast_error'] = (result['delta'] - result['predicted_mean'])/result['delta'] * 100
    result.replace([np.inf, -np.inf],0 , inplace=True)
    result = result.fillna(0)
    result = result.round(2)
    resp = (result.to_json(orient='table'))
    versa_sales_data= (versa_sm.to_json(orient='table'))

    conn = psycopg2.connect(
        database="development_master",
        user="root",
        password="root",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    for index, row in result.iterrows():
        query = '''
        SELECT *
        FROM sales_predictions
        WHERE item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' AND months = '%(index)s' ''' % {'item_id': item_id, 'firm_id': firm_id, 'index': index}
        # SQL injection
        pred_table = pd.read_sql_query(query, engine)
        if pred_table.empty: 
            cur.execute('INSERT INTO sales_predictions (months, forecast_sales, actual_sales , forecast_error, item_id, firm_id) values (%s,%s,%s,%s,%s,%s)',
                        (str(index), int(row['predicted_mean']), int(row['delta']), float(row['forecast_error']),  item_id, firm_id))
            conn.commit()
        else:
            cur.execute('UPDATE sales_predictions SET forecast_sales = %s , actual_sales = %s , forecast_error = %s WHERE item_id = %s AND firm_id = %s AND months = %s',
                        (int(row['predicted_mean']), int(row['delta']), float(row['forecast_error']),  item_id, firm_id, str(index)))
            conn.commit()

    cur.close()
    conn.close()
    return (resp)