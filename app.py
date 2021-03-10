from threading import Thread
from sklearn.metrics import mean_squared_error
from warnings import filterwarnings
from warnings import catch_warnings
from joblib import delayed
from joblib import Parallel
from multiprocessing import cpu_count
from math import sqrt
import time
import grid
import pred
import abc
import flask
import flask_restful
import sqlalchemy
import psycopg2
from flask import jsonify
import json
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


app = flask.Flask(__name__, template_folder='templates')
# api = Api(app)


def data_pts_check(num):
    if(num < 15):
        return False
    else:
        return True


@app.route('/')
def hello_world():
    return 'hello init1'
# if __name__== '__main__':
#    app.run()


@app.route('/grid', methods=['GET', 'POST'])
def main2():
    if (flask.request.method == 'GET'):
        return (flask.render_template('main_2.html'))
    if (flask.request.method == 'POST'):
        item_id = flask.request.form['item_id']
        firm_id = flask.request.form['firm_id']
        item_id = int(item_id)
        firm_id = int(firm_id)
        engine = sqlalchemy.create_engine(
            'postgresql://postgres:bits123@localhost:5432/versa_db')
        query1 = '''
		SELECT inventory_items.id, inventory_transaction_details.delta,	inventory_transaction_details.transaction_date
		FROM inventory_transaction_details
		INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id WHERE inventory_items.id = '%(item_id)d'
		ORDER BY inventory_item_id;
		''' % {'item_id': item_id}

        query2 = '''
		SELECT inventory_items.id, inventory_transaction_details.delta,	inventory_transaction_details.transaction_date
		FROM inventory_transaction_details
		INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id WHERE inventory_items.firm_id = '%(firm_id)d'
		ORDER BY inventory_item_id;
		''' % {'firm_id': firm_id}

        if item_id != -1:
            versa_sales = pd.read_sql_query(query1, engine)
        else:
            versa_sales = pd.read_sql_query(query2, engine)
        versa_sales1 = versa_sales[versa_sales["delta"] < 0]
        print(versa_sales1)
        # if item_id!=-1:
        #	versa_sales1 = versa_sales1[(versa_sales1["inventory_item_id"]==item_id)]
        # if firm_id!=-1:
        #	versa_sales1 = versa_sales1[(versa_sales1["firm_id"]==firm_id)]
        # cols = [2, 3, 4]
        # versa_sales1 = versa_sales1[versa_sales1.columns[cols]]
        versa_sales1["delta"] = versa_sales1["delta"].abs()
        versa_sales2 = versa_sales1.groupby(
            versa_sales1["transaction_date"], as_index=False).agg({'delta': np.sum})
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
            # return 0
        r = pd.date_range(start=versa_sales2.transaction_date.min(
        ), end=versa_sales2.transaction_date.max(), freq='MS')
        #r = pd.date_range(start=versa_sales2.transaction_date.min(), end=datetime.now())
        versa_sales3 = versa_sales2.set_index('transaction_date').reindex(
            r).fillna(0.0).rename_axis('transaction_date').reset_index()

        versa_sales_monthly = versa_sales3.groupby(
            versa_sales3.transaction_date.dt.to_period("M")).agg({'delta': np.sum})
        versa_sales_monthly["date"] = versa_sales_monthly.index
        versa_sales_monthly2 = versa_sales_monthly.reset_index(inplace=True)
        versa_sales_monthly = versa_sales_monthly.drop('date', axis=1)

        versa_sales_monthly.transaction_date = versa_sales_monthly.transaction_date.map(
            str)
        versa_sales_monthly['transaction_date'] = pd.to_datetime(
            versa_sales_monthly['transaction_date'])
        versa_sm = versa_sales_monthly.set_index('transaction_date')
        if (len(versa_sm.index)) < 12:
            print(-1)
            return flask.render_template('main_2.html', original_input={'item_id': item_id, 'firm_id': firm_id}, result="More Historical Sales Data required ",)
        engine = sqlalchemy.create_engine(
            'postgresql://postgres:bits123@localhost:5432/versa_db')
        query = '''
		SELECT *
		FROM forecasting_parameters
		WHERE inventory_item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' ''' % {'item_id': item_id, 'firm_id': firm_id}
        # SQL injection
        para_table = pd.read_sql_query(query, engine)
        if len(para_table.index) == 1:
            flag = para_table.loc[para_table['inventory_item_id']
                                  == item_id, 'flag'].iloc[0]
            if flag == 0:
                return "Prediction engine is searching for the best parameters"
            elif flag == 1:
                return "Parameter search already done, click predict"

        conn = psycopg2.connect(
            database="versa_db",
            user="postgres",
            password="bits123",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        cur.execute("INSERT into forecasting_parameters (inventory_item_id, firm_id,p,d,q,seasonal_p,seasonal_d,seasonal_q,s,flag) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (item_id, firm_id, -1, -1, -1, -1, -1, -1, -1, 0))
        conn.commit()
        cur.close()
        conn.close()
        t = Thread(target=grid.user_inp_grid_search,
                   args=(item_id, firm_id, versa_sm),)
        print("Hi")
        t.start()

        return flask.render_template('main_2.html', original_input={'item_id': item_id, 'firm_id': firm_id}, result="200! Request received",)


@app.route('/predict', methods=['GET', 'POST'])
def main():
    if (flask.request.method == 'GET'):
        return (flask.render_template('main.html'))
    if (flask.request.method == 'POST'):
        item_id = flask.request.form['item_id']
        firm_id = flask.request.form['firm_id']

        item_id = int(item_id)
        firm_id = int(firm_id)
        engine = sqlalchemy.create_engine(
            'postgresql://postgres:bits123@localhost:5432/versa_db')

        query1 = '''
		SELECT *
		FROM inventory_transaction_details
		INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id WHERE inventory_items.id = '%(item_id)d'
		ORDER BY inventory_item_id;
		''' % {'item_id': item_id}

        query2 = '''
		SELECT *
		FROM inventory_transaction_details
		INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id WHERE inventory_items.firm_id = '%(firm_id)d'
		ORDER BY inventory_item_id;
		''' % {'firm_id': firm_id}

        if item_id != -1:
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
        versa_sales1 = versa_sales[versa_sales["delta"] < 0]
        print(versa_sales1)
        # if item_id!=-1:
        # 	versa_sales1 = versa_sales1[(versa_sales1["inventory_item_id"]==item_id)]
        # if firm_id!=-1:
        # 	versa_sales1 = versa_sales1[(versa_sales1["firm_id"]==firm_id)]
        cols = [2, 3, 4]
        versa_sales1 = versa_sales1[versa_sales1.columns[cols]]
        versa_sales1["delta"] = versa_sales1["delta"].abs()
        versa_sales2 = versa_sales1.groupby(
            versa_sales1["transaction_date"], as_index=False).agg({'delta': np.sum})
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
            # return 0
        r = pd.date_range(start=versa_sales2.transaction_date.min(
        ), end=versa_sales2.transaction_date.max(), freq='MS')
        #r = pd.date_range(start=versa_sales2.transaction_date.min(), end=datetime.now())
        versa_sales3 = versa_sales2.set_index('transaction_date').reindex(
            r).fillna(0.0).rename_axis('transaction_date').reset_index()

        versa_sales_monthly = versa_sales3.groupby(
            versa_sales3.transaction_date.dt.to_period("M")).agg({'delta': np.sum})
        versa_sales_monthly["date"] = versa_sales_monthly.index
        versa_sales_monthly2 = versa_sales_monthly.reset_index(inplace=True)
        versa_sales_monthly = versa_sales_monthly.drop('date', axis=1)

        versa_sales_monthly.transaction_date = versa_sales_monthly.transaction_date.map(
            str)
        versa_sales_monthly['transaction_date'] = pd.to_datetime(
            versa_sales_monthly['transaction_date'])
        versa_sm = versa_sales_monthly.set_index('transaction_date')
        if data_pts_check(versa_sm["delta"].count()) == False:
            print(-1)
            # return 0
        # engine=sqlalchemy.create_engine('postgresql://postgres:bits123@18.216.156.245:5432/versa_db')
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
        # 	database="versa_db",
        # 	user="postgres",
        # 	password="bits123",
        # 	host="18.216.156.245",
        # 	port="5432"
        # 	)
        # cur= conn.cursor()

        # cur.execute("INSERT into forecasting_parameters (inventory_item_id, firm_id,p,d,q,seasonal_p,seasonal_d,seasonal_q,s,flag) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(item_id,firm_id,-1,-1,-1,-1,-1,-1,-1,0))
        # conn.commit()
        # cur.close()
        # conn.close()
        res = pred.sales_forecast(item_id, firm_id, versa_sm)
        results = json.loads(res)
        print(res)
        return flask.render_template('main.html', original_input={'item_id': item_id, 'firm_id': firm_id}, result=results['data'],)

# class fc(Resource):
#     def get(self, first_number, second_number):
#         return {'data' :pred.sales_forecast(first_number,second_number)}

# api.add_resource(fc, '/4cast/<first_number>/<second_number>')


@app.route('/abc', methods=['GET', 'POST'])
def main3():
    if (flask.request.method == 'GET'):
        return (flask.render_template('main3.html'))
    if (flask.request.method == 'POST'):
        firm_id = flask.request.form['firm_id']
        firm_id = int(firm_id)
        engine = sqlalchemy.create_engine(
            'postgresql://postgres:bits123@localhost:5432/versa_db')

        query = '''
		SELECT products.name, parts.part_number, price_components.price,  inventory_items.id, inventory_items.firm_id,  inventory_transaction_details.delta, inventory_transaction_details.transaction_date,  price_components.currency_id, currencies.name as currency_name
		FROM inventory_transaction_details
		INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id INNER JOIN parts on parts.id = inventory_items.part_id INNER JOIN products on parts.id=products.part_id INNER JOIN price_components on price_components.product_id=products.id and price_components.measurement_unit_id = products.measurement_unit_id INNER JOIN currencies on currencies.id = price_components.currency_id WHERE price_components.break_quantity = 1 and inventory_items.firm_id = '%(firm_id)d'
		ORDER BY inventory_item_id;
		''' % {'firm_id': firm_id}
        versa_sales = pd.read_sql_query(query, engine)
        versa_sales['transaction_date'] = pd.to_datetime(
            versa_sales['transaction_date'], errors='coerce')
        versa_sales = versa_sales[versa_sales["delta"] < 0]
        versa_sales["delta"] = versa_sales["delta"].abs()
        versa_sales = versa_sales.drop(['firm_id'], axis=1)
        versa_sales['transaction_date'] = versa_sales.transaction_date.dt.to_period(
            "M")
        versa_sales = versa_sales.groupby(
            ['id', 'part_number', 'price', 'currency_id', 'currency_name', 'transaction_date'], as_index=False).sum()
        # obsolete_items = versa_sales[versa_sales['transaction_date']<'2018-01-01']
        today = pd.to_datetime("today")
        obsolete_time = (today - pd.DateOffset(years=2)).to_period("M")
        versa_sales = versa_sales[versa_sales['transaction_date']
                                  > obsolete_time]

        # non-stocked items
        # versa_sales = versa_sales[versa_sales['transaction_date']>'2019-01-01']

        no_of_items = len(versa_sales)

        # abc analysis
        versa_sales['revenue'] = versa_sales.price * versa_sales.delta
        versa_sales_abc = versa_sales.sort_values(
            by='revenue', ascending=False)
        A, B, C = np.split(versa_sales_abc, [
                           int(.2*no_of_items), int(.5*no_of_items)])

        # HML analysis

        versa_sales_hml = versa_sales.sort_values(by='delta', ascending=False)
        high, medium, low = np.split(
            versa_sales_hml, [int(.2*no_of_items), int(.5*no_of_items)])
        # tables
        ha = pd.merge(high, A, how='inner')
        ha_len = len(ha)
        ha_value = ha['revenue'].sum().astype(int)
        hb = pd.merge(high, B, how='inner')
        hb_len = len(hb)
        hb_value = hb['revenue'].sum().astype(int)
        hc = pd.merge(high, C, how='inner')
        hc_len = len(hc)
        hc_value = hc['revenue'].sum().astype(int)
        ma = pd.merge(medium, A, how='inner')
        ma_len = len(ma)
        ma_value = ma['revenue'].sum().astype(int)
        mb = pd.merge(medium, B, how='inner')
        mb_len = len(mb)
        mb_value = mb['revenue'].sum().astype(int)
        mc = pd.merge(medium, C, how='inner')
        mc_len = len(mc)
        mc_value = mc['revenue'].sum().astype(int)
        la = pd.merge(low, A, how='inner')
        la_len = len(la)
        la_value = la['revenue'].sum().astype(int)
        lb = pd.merge(low, B, how='inner')
        lb_len = len(lb)
        lb_value = lb['revenue'].sum().astype(int)
        lc = pd.merge(low, C, how='inner')
        lc_len = len(lc)
        lc_value = lc['revenue'].sum().astype(int)

        items = {
            'ha_len': ha_len,
            'hb_len': hb_len,
            'hc_len': hc_len,
            'ma_len': ma_len,
            'mb_len': mb_len,
            'mc_len': mc_len,
            'la_len': la_len,
            'lb_len': lb_len,
            'lc_len': lc_len,

        }

        values = {
            'ha_value': ha_value,
            'hb_value': hb_value,
            'hc_value': hc_value,
            'ma_value': ma_value,
            'mb_value': mb_value,
            'mc_value': mc_value,
            'la_value': la_value,
            'lb_value': lb_value,
            'lc_value': lc_value,

        }
        return flask.render_template('main3.html', original_input={'firm_id': firm_id}, items=items, values=values, tables=[la.to_html(classes='data', header="true"), ma.to_html(classes='data', header="true"), ha.to_html(classes='data', header="true"), lb.to_html(classes='data', header="true"), mb.to_html(classes='data', header="true"), hb.to_html(classes='data', header="true"), lc.to_html(classes='data', header="true"), mc.to_html(classes='data', header="true"), hc.to_html(classes='data', header="true")], titles=['na', 'Low and A', 'Medium and A', 'High and A', 'Low and B', 'Medium and B', 'High and B', 'Low and C', 'Medium and C', 'High and C'],)


if __name__ == '__main__':
    app.run()
