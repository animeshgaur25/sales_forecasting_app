
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


def sales_forecast(item_id, firm_id, versa_sm):
    item_id = int(item_id)
    firm_id = int(firm_id)
    #versa_sales = pd.read_csv(r"C:\Users\prasa\Documents\programs\demo_sales_fc\data_updated22-09.csv", parse_dates=[4], index_col=0, squeeze=True, date_parser=parser)
    engine = sqlalchemy.create_engine(
        'postgresql://postgres:bits123@localhost:5432/versa_db')
    query = '''
    SELECT *
    FROM forecasting_parameters
    WHERE flag = 1 AND inventory_item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' ''' % {'item_id': item_id, 'firm_id': firm_id}
    # SQL injection
    parameters = pd.read_sql_query(query, engine)

    para = parameters[(parameters["inventory_item_id"] == item_id)]
    p = para.loc[para['inventory_item_id'] == item_id, 'p'].iloc[0]
    d = para.loc[para['inventory_item_id'] == item_id, 'd'].iloc[0]
    q = para.loc[para['inventory_item_id'] == item_id, 'q'].iloc[0]
    P = para.loc[para['inventory_item_id'] == item_id, 'seasonal_p'].iloc[0]
    Q = para.loc[para['inventory_item_id'] == item_id, 'seasonal_q'].iloc[0]
    D = para.loc[para['inventory_item_id'] == item_id, 'seasonal_d'].iloc[0]
    s = para.loc[para['inventory_item_id'] == item_id, 's'].iloc[0]
    if d == 1:
        train_data = versa_sm.diff()[1:]
        # first_diff
    else:
        train_data = versa_sm

    my_order = (p, d, q)
    my_seasonal_order = (P, D, Q, s)

# till here
    model = SARIMAX(train_data["delta"], order=my_order,
                    seasonal_order=my_seasonal_order)
    #start = time()
    model_fit = model.fit()
    #end = time()
    predictions = model_fit.forecast(12)
    # return predictions
    print("*****************************\n", predictions)
    if d == 0:
        predictions.columns = ["values"]
        resp = predictions.to_json(orient='table')

    else:
        res = pd.Series()
        initial_val = versa_sm['delta'][-1]
        for i in range(len(predictions)):
            res = res.append(
                pd.Series((initial_val+predictions[i]), name="predicted_mean", index=[predictions.index[i]]))
            res.name = "predicted_mean"
            initial_val = initial_val+predictions[i]
        resp = res.to_json(orient='table')



    conn = psycopg2.connect(
        database="versa_db",
        user="postgres",
        password="bits123",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    cur.execute("UPDATE sales_predictions SET predictions = %s WHERE item_id = %s AND firm_id = %s",
                (resp, item_id, firm_id))
    
    conn.commit()
    cur.close()
    conn.close()

    return (resp)