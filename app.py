import grid
import pred
import flask
import flask_restful
import sqlalchemy
import psycopg2
# from flask import Flask, request
# from flask_restful import Resource, Api
import math
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
import time
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from threading import Thread


app = flask.Flask(__name__, template_folder='templates')
# api = Api(app)

def data_pts_check(num):
	if(num<15):
		return False
	else:
		return True

@app.route('/')
def hello_world():
    return 'hello init1'
#if __name__== '__main__':
#    app.run()

@app.route('/grid', methods=['GET','POST'])
def main2():
	if (flask.request.method == 'GET'):
		return (flask.render_template('main_2.html'))
	if (flask.request.method == 'POST'):
		item_id=flask.request.form['item_id']
		firm_id=flask.request.form['firm_id']
		item_id=int(item_id)
		firm_id=int(firm_id)
		engine= sqlalchemy.create_engine('postgresql://postgres:1997@localhost:5432/versa_db_2')
		query1='''
		SELECT *
		FROM inventory_transaction_details
		INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id WHERE inventory_items.id = '%(item_id)d'
		ORDER BY inventory_item_id;
		'''%{'item_id': item_id}

		query2='''
		SELECT *
		FROM inventory_transaction_details
		INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id WHERE inventory_items.firm_id = '%(firm_id)d'
		ORDER BY inventory_item_id;
		'''%{'firm_id': firm_id}

		if item_id!=-1:
			versa_sales = pd.read_sql_query(query1, engine)
		else:
			versa_sales = pd.read_sql_query(query2, engine)
		versa_sales1 = versa_sales[versa_sales["delta"]<0]
		print(versa_sales1)
		#if item_id!=-1:
		#	versa_sales1 = versa_sales1[(versa_sales1["inventory_item_id"]==item_id)]
		#if firm_id!=-1:
		#	versa_sales1 = versa_sales1[(versa_sales1["firm_id"]==firm_id)]
		cols=[2,3,4]
		versa_sales1=versa_sales1[versa_sales1.columns[cols]]
		versa_sales1["delta"]=versa_sales1["delta"].abs()
		versa_sales2=versa_sales1.groupby(versa_sales1["transaction_date"], as_index=False).agg({'delta': np.sum})
		print(versa_sales2)
		# cols=[2,3,4]
		# versa_sales1=versa_sales1[versa_sales1.columns[cols]]
		# versa_sales1["delta"]=versa_sales1["delta"].abs()
		# versa_sales2=versa_sales1.groupby(versa_sales1["transaction_date"], as_index=False).agg({'delta': np.sum})
		flag = 0
		current_time = datetime.now()
		versa_maxyear = (versa_sales2.transaction_date.max()).year
		if current_time.year-versa_maxyear > 1:
			flag = -1
			print(flag)
			return 0
		r = pd.date_range(start=versa_sales2.transaction_date.min(), end=versa_sales2.transaction_date.max(),freq='MS')
		#r = pd.date_range(start=versa_sales2.transaction_date.min(), end=datetime.now())
		versa_sales3=versa_sales2.set_index('transaction_date').reindex(r).fillna(0.0).rename_axis('transaction_date').reset_index()

		versa_sales_monthly = versa_sales3.groupby(versa_sales3.transaction_date.dt.to_period("M")).agg({'delta': np.sum})
		versa_sales_monthly["date"]=versa_sales_monthly.index
		versa_sales_monthly2=versa_sales_monthly.reset_index(inplace = True)
		versa_sales_monthly=versa_sales_monthly.drop('date',axis=1)

		versa_sales_monthly.transaction_date = versa_sales_monthly.transaction_date.map(str)
		versa_sales_monthly['transaction_date']=pd.to_datetime(versa_sales_monthly['transaction_date'])
		versa_sm=versa_sales_monthly.set_index('transaction_date')
		if data_pts_check(versa_sm["delta"].count())==False:
			print(-1)
			return 0   	
		engine=sqlalchemy.create_engine('postgresql://postgres:1997@localhost:5432/versa_db_2')
		query='''
		SELECT *
		FROM forecasting_parameters
		WHERE inventory_item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' ''' %{'item_id': item_id, 'firm_id': firm_id}
		#SQL injection
		para_table= pd.read_sql_query(query, engine)
		if len(para_table.index)==1:
			flag=para_table.loc[para_table['inventory_item_id'] == item_id, 'flag'].iloc[0]
			if flag==0:
				return "Prediction engine is searching for the best parameters"
			elif flag==1:
				return "Parameter search already done, click predict"
			#return 0
		conn = psycopg2.connect(
			database="versa_db_2",
			user="postgres",
			password="1997",
			host="localhost",
			port="5432"
			)
		cur= conn.cursor()

		cur.execute("INSERT into forecasting_parameters (inventory_item_id, firm_id,p,d,q,seasonal_p,seasonal_d,seasonal_q,s,flag) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(item_id,firm_id,-1,-1,-1,-1,-1,-1,-1,0))
		conn.commit()
		cur.close()
		conn.close()
		t = Thread(target = grid.user_inp_grid_search, args=(item_id,firm_id,versa_sm),)
		t.start()

		return flask.render_template('main_2.html',original_input={'item_id':item_id,'firm_id':firm_id},result="200! Request received",)

@app.route('/predict', methods=['GET','POST'])
def main():
	if (flask.request.method == 'GET'):
		return (flask.render_template('main.html'))
	if (flask.request.method == 'POST'):
		item_id=flask.request.form['item_id']
		firm_id=flask.request.form['firm_id']

		item_id=int(item_id)
		firm_id=int(firm_id)
		engine= sqlalchemy.create_engine('postgresql://postgres:1997@localhost:5432/versa_db_2')

		query1='''
		SELECT *
		FROM inventory_transaction_details
		INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id WHERE inventory_items.id = '%(item_id)d'
		ORDER BY inventory_item_id;
		'''%{'item_id': item_id}

		query2='''
		SELECT *
		FROM inventory_transaction_details
		INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id WHERE inventory_items.firm_id = '%(firm_id)d'
		ORDER BY inventory_item_id;
		'''%{'firm_id': firm_id}

		if item_id!=-1:
			versa_sales = pd.read_sql_query(query1, engine)
		else:
			versa_sales = pd.read_sql_query(query2, engine)

		# query='''
		# SELECT *
		# FROM inventory_transaction_details
		# INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id
		# ORDER BY inventory_item_id;
		# '''

		# versa_sales = pd.read_sql_query(query, engine)
		versa_sales1 = versa_sales[versa_sales["delta"]<0]
		print(versa_sales1)
		# if item_id!=-1:
		# 	versa_sales1 = versa_sales1[(versa_sales1["inventory_item_id"]==item_id)]
		# if firm_id!=-1:
		# 	versa_sales1 = versa_sales1[(versa_sales1["firm_id"]==firm_id)]
		cols=[2,3,4]
		versa_sales1=versa_sales1[versa_sales1.columns[cols]]
		versa_sales1["delta"]=versa_sales1["delta"].abs()
		versa_sales2=versa_sales1.groupby(versa_sales1["transaction_date"], as_index=False).agg({'delta': np.sum})
		print(versa_sales2)
		# cols=[2,3,4]
		# versa_sales1=versa_sales1[versa_sales1.columns[cols]]
		# versa_sales1["delta"]=versa_sales1["delta"].abs()
		# versa_sales2=versa_sales1.groupby(versa_sales1["transaction_date"], as_index=False).agg({'delta': np.sum})
		flag = 0
		current_time = datetime.now()
		versa_maxyear = (versa_sales2.transaction_date.max()).year
		if current_time.year-versa_maxyear > 1:
			flag = -1
			print(flag)
			return 0
		r = pd.date_range(start=versa_sales2.transaction_date.min(), end=versa_sales2.transaction_date.max(),freq='MS')
		#r = pd.date_range(start=versa_sales2.transaction_date.min(), end=datetime.now())
		versa_sales3=versa_sales2.set_index('transaction_date').reindex(r).fillna(0.0).rename_axis('transaction_date').reset_index()

		versa_sales_monthly = versa_sales3.groupby(versa_sales3.transaction_date.dt.to_period("M")).agg({'delta': np.sum})
		versa_sales_monthly["date"]=versa_sales_monthly.index
		versa_sales_monthly2=versa_sales_monthly.reset_index(inplace = True)
		versa_sales_monthly=versa_sales_monthly.drop('date',axis=1)

		versa_sales_monthly.transaction_date = versa_sales_monthly.transaction_date.map(str)
		versa_sales_monthly['transaction_date']=pd.to_datetime(versa_sales_monthly['transaction_date'])
		versa_sm=versa_sales_monthly.set_index('transaction_date')
		if data_pts_check(versa_sm["delta"].count())==False:
			print(-1)
			return 0   	
		# engine=sqlalchemy.create_engine('postgresql://postgres:1997@localhost:5432/versa_db_2')
		# query='''
		# SELECT *
		# FROM forecasting_parameters
		# WHERE inventory_item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' ''' %{'item_id': item_id, 'firm_id': firm_id}
		# #SQL injection
		# para_table= pd.read_sql_query(query, engine)
		# if len(para_table.index)==1:
		# 	flag=para_table.loc[para_table['inventory_item_id'] == item_id, 'flag'].iloc[0]
		# 	if flag==0:
		# 		print("Prediction engine is searching for the best parameters")
		# 	elif flag==1:
		# 		print("Parameter search already done, click predict")
		# 	return 0;
		# conn = psycopg2.connect(
		# 	database="versa_db_2",
		# 	user="postgres",
		# 	password="1997",
		# 	host="localhost",
		# 	port="5432"
		# 	)
		# cur= conn.cursor()

		# cur.execute("INSERT into forecasting_parameters (inventory_item_id, firm_id,p,d,q,seasonal_p,seasonal_d,seasonal_q,s,flag) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(item_id,firm_id,-1,-1,-1,-1,-1,-1,-1,0))
		# conn.commit()
		# cur.close()
		# conn.close()

		res=pred.sales_forecast(item_id,firm_id,versa_sm)
		return flask.render_template('main.html',original_input={'item_id':item_id,'firm_id':firm_id},result=res,)

# class fc(Resource):
#     def get(self, first_number, second_number):
#         return {'data' :pred.sales_forecast(first_number,second_number)}
    
# api.add_resource(fc, '/4cast/<first_number>/<second_number>')



if __name__=='__main__':
    app.run()