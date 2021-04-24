from threading import Thread, Timer
from sklearn.metrics import mean_squared_error
from warnings import filterwarnings
from warnings import catch_warnings
from joblib import delayed
from joblib import Parallel
from multiprocessing import cpu_count
from math import sqrt
from sklearn.model_selection import train_test_split
import math
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


def data_pts_check(versa_sm):
    if (len(versa_sm.index)) < 12:
        print("More Sales Data is required")
        return -1
def extract_sales_data(firm_id):
    engine = sqlalchemy.create_engine(
            'postgresql://postgres:bits123@localhost:5432/versa_db')

    query1 = '''
    SELECT inventory_items.id , inventory_items.firm_id, inventory_transaction_details.delta,	inventory_transaction_details.transaction_date
    FROM inventory_transaction_details
    INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id WHERE inventory_items.firm_id = '%(firm_id)d'
    ORDER BY inventory_item_id;
    '''% {'firm_id': firm_id}

    versa_sales = pd.read_sql_query(query1, engine)
    versa_sales1 = versa_sales[versa_sales["delta"] < 0]
    versa_sales1["delta"] = versa_sales1["delta"].abs()

    versa_sales1 = versa_sales1.groupby(['firm_id', 'id', 'transaction_date'], as_index=False).sum()
    return versa_sales1


def call_grid_search(versa_sales1, item_id, engine, conn, cur):
    firm_id = versa_sales1.loc[versa_sales1['id']== item_id, 'firm_id'].iloc[0]
    versa_sales1 = versa_sales1[versa_sales1.id ==item_id]
    print("***************************Item Id:", item_id, "********************************")
    versa_sm = grid.data_preprocessing(versa_sales1, item_id)
    
    if versa_sm is None:
        return flask.render_template('main_2.html', result = "More Sales Data is required for this item")
    else:
        query = '''
        SELECT *
        FROM forecasting_parameters
        WHERE inventory_item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' ''' % {'item_id': int(item_id), 'firm_id': int(firm_id)}
    
        # SQL injection
        para_table = pd.read_sql_query(query, engine)

        if para_table.empty:
            cur.execute("INSERT into forecasting_parameters (inventory_item_id, firm_id,p,d,q,seasonal_p,seasonal_d,seasonal_q,s,flag) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (int(item_id), int(firm_id), -1, -1, -1, -1, -1, -1, -1, 0))
            conn.commit()

            t = Thread(target=grid.user_inp_grid_search,
                        args=(item_id, firm_id, versa_sm),)
            t.start()
        else:
            t = Thread(target=grid.user_inp_grid_search,
                    args=(item_id, firm_id, versa_sm),)
            t.start()


    return


def extract_high_items(firm_id):
    engine = sqlalchemy.create_engine(
    'postgresql://postgres:bits123@localhost:5432/versa_db')
    query = '''
    SELECT products.name, parts.part_number, price_components.price,  inventory_items.id, inventory_items.firm_id,  inventory_transaction_details.delta, inventory_transaction_details.transaction_date,  price_components.currency_id, currencies.name as currency_name
    FROM inventory_transaction_details
    INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id INNER JOIN parts on parts.id = inventory_items.part_id INNER JOIN products on parts.id=products.part_id INNER JOIN price_components on price_components.product_id=products.id and price_components.measurement_unit_id = products.measurement_unit_id INNER JOIN currencies on currencies.id = price_components.currency_id WHERE price_components.break_quantity = 1 and inventory_items.firm_id = '%(firm_id)d'
    ORDER BY inventory_item_id;
    ''' % {'firm_id': firm_id}
    data_abc = pd.read_sql_query(query, engine)
    print(data_abc)
    if data_abc.empty:
        return data_abc

    else:
        data_abc['transaction_date'] = pd.to_datetime(data_abc['transaction_date'], errors='coerce')
        data_abc['transaction_date'] = data_abc['transaction_date'].dropna()
        data_abc = data_abc[data_abc["delta"] < 0]
        data_abc["delta"] = data_abc["delta"].abs()
        data_abc = data_abc.drop(['firm_id'], axis=1)
        data_abc['transaction_date'] = data_abc.transaction_date.dt.to_period(
            "M")
        # data_abc = data_abc[data_abc==firm_id]
        data_abc = data_abc.groupby(
            ['id', 'part_number', 'price', 'currency_id', 'currency_name', 'transaction_date'], as_index=False).sum()
        # obsolete_items = data_abc[data_abc['transaction_date']<'2018-01-01']
        today = pd.to_datetime("today")
        print(data_abc)
        obsolete_time = (today - pd.DateOffset(years=2)).to_period("M")
        data_abc = data_abc[data_abc['transaction_date'] > obsolete_time]

        # non-stocked items
        # data_abc = data_abc[data_abc['transaction_date']>'2019-01-01']

        no_of_items = len(data_abc)

        # abc analysis
        data_abc['revenue'] = data_abc.price * data_abc.delta
        versa_sales_abc = data_abc.sort_values(
            by='revenue', ascending=False)
        A, B, C = np.split(versa_sales_abc, [
                            int(.2*no_of_items), int(.5*no_of_items)])

        # HML analysis

        versa_sales_hml = data_abc.sort_values(by='delta', ascending=False)
        high, medium, low = np.split(
            versa_sales_hml, [int(.2*no_of_items), int(.5*no_of_items)])
        # tables
        ha = pd.merge(high, A, how='inner')
        high_item_ids = high['id'].unique()
        return high_item_ids

def weekly_update_parameters():
    engine = sqlalchemy.create_engine(
    'postgresql://postgres:bits123@localhost:5432/versa_db')
    conn = psycopg2.connect(
        database="versa_db",
        user="postgres",
        password="bits123",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    query = '''
        SELECT firms.id
        FROM firms'''

    firms = pd.read_sql_query(query, engine)
    firm_ids = firms['id']
    print(firm_ids)
    for firms_id in firm_ids[1:]:
        print("****************************", firms_id)
        firm_id = int(firms_id)
        item_ids = extract_high_items(firm_id)
        if item_ids is not None:
            versa_sales1 = extract_sales_data(firm_id)
            firm_ids = versa_sales1['firm_id']
            print(item_ids)
            thread_list = []

            for item_id in item_ids:
                thread =  Thread(target = call_grid_search, args=(versa_sales1, item_id,engine, conn, cur),)
                thread_list.append(thread)
            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()

    return

@app.route('/')
def hello_world():
    return 'hello init1'
# if __name__== '__main__':
#    app.run()


@app.route('/grid', methods=['GET', 'POST'])
def main2():
    if (flask.request.method == 'GET'):
        return flask.render_template('main_2.html')

    if (flask.request.method == 'POST'):
        item_id = flask.request.form['item_id']
        firm_id = flask.request.form['firm_id']
        item_id = int(item_id)
        firm_id = int(firm_id)
       
        engine = sqlalchemy.create_engine(
        'postgresql://postgres:bits123@localhost:5432/versa_db')
        conn = psycopg2.connect(
            database="versa_db",
            user="postgres",
            password="bits123",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        # query = '''
        # SELECT forecasting_parameters.inventory_item_id
        # FROM forecasting_parameters
        # WHERE forecasting_parameters.firm_id = %(firm_id)d and forecasting_parameters.flag = 1
        # ''' % {'firm_id': firm_id}

        item_ids = extract_high_items(firm_id)

        if item_ids is None:
            return flask.render_template('main_2.html', result = "More Sales Data is required for this item",items = item_ids)
        
        print(item_ids)

        versa_sales = extract_sales_data(firm_id)
        if item_id == -1:
            thread_list = []
            
            for item_id in item_ids:
                print(item_id)
                check = data_pts_check(versa_sales)
                if check ==-1:
                    continue

                thread =  Thread(target=call_grid_search, args=(versa_sales, item_id, engine, conn, cur),)
                thread_list.append(thread)

            for thread in thread_list:
                thread.start()

            for thread in thread_list:
                thread.join()


        else:
            print(versa_sales)
            versa_sales = versa_sales[versa_sales.id ==item_id]
            check = data_pts_check(versa_sales)
            if check ==-1:
                return flask.render_template('main_2.html', result = "More Sales Data is required for this item",items = item_ids)
            else:
                thread =  Thread(target=call_grid_search, args=(versa_sales, item_id, engine, conn, cur),)
                thread.start()
    
        return flask.render_template('main_2.html',result = "Request Received! Running different possible model configurations on Historical Sales Data to get best sales forecast.",items = item_ids)


@app.route('/predict', methods=['GET', 'POST'])
def main():


    if (flask.request.method == 'GET'):
        engine = sqlalchemy.create_engine(
            'postgresql://postgres:bits123@localhost:5432/versa_db')

        query = '''
        SELECT forecasting_parameters.inventory_item_id
        FROM forecasting_parameters
        WHERE forecasting_parameters.firm_id = 568 and forecasting_parameters.flag = 1
        '''

        item_ids = pd.read_sql_query(query, engine)
        return flask.render_template('main.html', items = item_ids)


    if (flask.request.method == 'POST'):
        item_id = flask.request.form['item_id']
        firm_id = flask.request.form['firm_id']
        item_id = int(item_id)
        firm_id = int(firm_id)
  
        versa_sales = extract_sales_data(firm_id)
        versa_sales = versa_sales[versa_sales.id ==item_id]
        versa_sm = grid.data_preprocessing(versa_sales, item_id)

        engine = sqlalchemy.create_engine(
            'postgresql://postgres:bits123@localhost:5432/versa_db')
        conn = psycopg2.connect(
            database="versa_db",
            user="postgres",
            password="bits123",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        query1 = '''
        SELECT forecasting_parameters.inventory_item_id
        FROM forecasting_parameters
        WHERE firm_id = 568 and forecasting_parameters.flag = 1
        '''

        item_ids = pd.read_sql_query(query1, engine)
        query = '''
        SELECT *
        FROM sales_predictions
        WHERE item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' ''' % {'item_id': int(item_id), 'firm_id': int(firm_id)}
    
        # SQL injection
        para_table = pd.read_sql_query(query, engine)

        if para_table.empty:
            cur.execute("INSERT into sales_predictions (item_id, firm_id, predictions) values (%s,%s,%s)",
                        (int(item_id), int(firm_id), -1))
        
        res = pred.sales_forecast(item_id, firm_id, versa_sm)
        results = json.loads(res)
        print(res)
        return flask.render_template('main.html', original_input={'item_id': item_id, 'firm_id': firm_id},items=item_ids, result=results['data'],)

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
        data_abc = pd.read_sql_query(query, engine)
        data_abc['transaction_date'] = pd.to_datetime(
            data_abc['transaction_date'], errors='coerce')
        data_abc = data_abc[data_abc["delta"] < 0]
        data_abc["delta"] = data_abc["delta"].abs()
        data_abc = data_abc.drop(['firm_id'], axis=1)
        data_abc['transaction_date'] = data_abc.transaction_date.dt.to_period(
            "M")
        # data_abc = data_abc[data_abc==firm_id]
        data_abc = data_abc.groupby(
            ['id', 'part_number', 'price', 'currency_id', 'currency_name', 'transaction_date'], as_index=False).sum()
        # obsolete_items = data_abc[data_abc['transaction_date']<'2018-01-01']
        today = pd.to_datetime("today")
        obsolete_time = (today - pd.DateOffset(years=2)).to_period("M")
        data_abc = data_abc[data_abc['transaction_date']
                                  > obsolete_time]

        # non-stocked items
        # data_abc = data_abc[data_abc['transaction_date']>'2019-01-01']

        no_of_items = len(data_abc)

        # abc analysis
        print(data_abc)
        data_abc['revenue'] = data_abc.price * data_abc.delta
        versa_sales_abc = data_abc.sort_values(
            by='revenue', ascending=False)
        A, B, C = np.split(versa_sales_abc, [
                           int(.2*no_of_items), int(.5*no_of_items)])

        # HML analysis

        versa_sales_hml = data_abc.sort_values(by='delta', ascending=False)
        high, medium, low = np.split(
            versa_sales_hml, [int(.2*no_of_items), int(.5*no_of_items)])
        # tables
        ha = pd.merge(high, A, how='inner')
        ha_len = len(ha)
        ha_value = int(ha['revenue'].sum())
        ha_demand =int(ha['delta'].mean())
        hb = pd.merge(high, B, how='inner')
        hb_len = len(hb)
        hb_value = int(hb['revenue'].sum())
        hb_demand =int(hb['delta'].mean())
        hc = pd.merge(high, C, how='inner')
        hc_len = len(hc)
        hc_value = int(hc['revenue'].sum())
        hc_demand =int(hc['delta'].mean())
        ma = pd.merge(medium, A, how='inner')
        ma_len = len(ma)
        ma_value = int(ma['revenue'].sum())
        ma_demand =int(ma['delta'].mean())
        mb = pd.merge(medium, B, how='inner')
        mb_len = len(mb)
        mb_value = int(mb['revenue'].sum())
        mb_demand =int(mb['delta'].mean())
        mc = pd.merge(medium, C, how='inner')
        mc_len = len(mc)
        mc_value = int(mc['revenue'].sum())
        mc_demand =int(mc['delta'].mean())
        la = pd.merge(low, A, how='inner')

        if not la.empty:
            la_len = len(la)
            la_value = int(la['revenue'].sum())
            la_demand =int(la['delta'].mean())
        else:
            la_demand = 0 
            la_len= 0
            la_value = 0

        lb = pd.merge(low, B, how='inner')
        lb_len = len(lb)
        lb_value = int(lb['revenue'].sum())
        lb_demand =int(lb['delta'].mean())
        lc = pd.merge(low, C, how='inner')
        lc_len = len(lc)
        lc_value = int(lc['revenue'].sum())
        lc_demand =int(lc['delta'].mean())

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

        demand = {
            'ha_demand': ha_demand,
            'hb_demand': hb_demand,
            'hc_demand': hc_demand,
            'ma_demand': ma_demand,
            'mb_demand': mb_demand,
            'mc_demand': mc_demand,
            'la_demand': la_demand,
            'lb_demand': lb_demand,
            'lc_demand': lc_demand,
        }
        return flask.render_template('main3.html', original_input={'firm_id': firm_id}, items=items, values=values, demand=demand, tables=[la.to_html(classes='data', header="true"), ma.to_html(classes='data', header="true"), ha.to_html(classes='data', header="true"), lb.to_html(classes='data', header="true"), mb.to_html(classes='data', header="true"), hb.to_html(classes='data', header="true"), lc.to_html(classes='data', header="true"), mc.to_html(classes='data', header="true"), hc.to_html(classes='data', header="true")], titles=['na', 'Low and A', 'Medium and A', 'High and A', 'Low and B', 'Medium and B', 'High and B', 'Low and C', 'Medium and C', 'High and C'],)




if __name__ == '__main__':
    app.run()
