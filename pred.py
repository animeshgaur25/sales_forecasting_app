# from flask import flask
# app = Flask(__name__)
# @app.route('/')
# def hello_world():
#     return 'hello init1'
# if __name__== '__main__':
#     app.run()

import math
import json
import sqlalchemy
import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error

def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

def data_pts_chk(num):
    if(num<12):
        return False
    else:
        return True

def sales_forecast(item_id,firm_id,versa_sm):
    item_id=int(item_id)
    firm_id=int(firm_id)
    #versa_sales = pd.read_csv(r"C:\Users\prasa\Documents\programs\demo_sales_fc\data_updated22-09.csv", parse_dates=[4], index_col=0, squeeze=True, date_parser=parser)
    engine=sqlalchemy.create_engine('postgresql://postgres:1997@localhost:5432/versa_db_2')
    query='''
    SELECT *
    FROM forecasting_parameters
    WHERE inventory_item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' ''' %{'item_id': item_id, 'firm_id': firm_id}
    #SQL injection
    parameters= pd.read_sql_query(query, engine)
    #parameters=pd.read_csv(r"C:\Users\prasa\Documents\programs\demo_sales_fc\parameters.csv")
    #versa_sales1 = versa_sales[versa_sales["delta"]<0]
    #if item_id!=-1:
    #    versa_sales1 = versa_sales1[(versa_sales1["inventory_item_id"]==item_id)]
    #if firm_id!=-1:
    #    versa_sales1 = versa_sales1[(versa_sales1["firm_id"]==firm_id)]
    #print(versa_sales1)
    #cols=[2,3,4]
    #versa_sales1=versa_sales1[versa_sales1.columns[cols]]
    #versa_sales1["delta"]=versa_sales1["delta"].abs()
    #versa_sales2=versa_sales1.groupby(versa_sales1["transaction_date"], as_index=False).agg({'delta': np.sum})
    #if versa_sales2["id"].count()<12:
    #    return "error"
    #r = pd.date_range(start=versa_sales2.transaction_date.min(), end=versa_sales2.transaction_date.max(), freq='M')
    #versa_sales3=versa_sales2.set_index('transaction_date').reindex(r).fillna(0.0).rename_axis('transaction_date').reset_index()
    
    #versa_sales_monthly = versa_sales3.groupby(versa_sales3.transaction_date.dt.to_period("M")).agg({'delta': np.sum})
    #versa_sales_monthly["date"]=versa_sales_monthly.index
    #versa_sales_monthly2=versa_sales_monthly.reset_index(inplace = True)
    #versa_sales_monthly=versa_sales_monthly.drop('date',axis=1)
    
    #versa_sales_monthly.transaction_date = versa_sales_monthly.transaction_date.map(str)
    #versa_sales_monthly['transaction_date']=pd.to_datetime(versa_sales_monthly['transaction_date'])
    #versa_sm=versa_sales_monthly.set_index('transaction_date')
    para=parameters[(parameters["inventory_item_id"]==item_id)]
    p=para.loc[para['inventory_item_id'] == item_id, 'p'].iloc[0]
    d=para.loc[para['inventory_item_id'] == item_id, 'd'].iloc[0]
    q=para.loc[para['inventory_item_id'] == item_id, 'q'].iloc[0]
    P=para.loc[para['inventory_item_id'] == item_id, 'seasonal_p'].iloc[0]
    Q=para.loc[para['inventory_item_id'] == item_id, 'seasonal_q'].iloc[0]
    D=para.loc[para['inventory_item_id'] == item_id, 'seasonal_d'].iloc[0]
    s=para.loc[para['inventory_item_id'] == item_id, 's'].iloc[0]
    if d==1:
        train_data = versa_sm.diff()[1:]
        #first_diff
    else:
        train_data=versa_sm
    if data_pts_chk(train_data["delta"].count())==False:
        return "error"
    my_order = (p,d,q)
    my_seasonal_order = (P, D, Q, s)
    model = SARIMAX(train_data["delta"], order=my_order, seasonal_order=my_seasonal_order)
    #start = time()
    model_fit = model.fit()
    #end = time()
    predictions = model_fit.forecast(3)
    #return predictions
    
    if d==0:
        resp1=predictions.to_json(orient='table')
        return str(resp1)
    else:
        res=pd.Series()
        initial_val=versa_sm['delta'][-1]
        for i in range(len(predictions)):
            res=res.append(pd.Series((initial_val+predictions[i]),index=[predictions.index[i]]))
            initial_val=initial_val+predictions[i]
        resp2=res.to_json(orient='table')
        return str(resp2)