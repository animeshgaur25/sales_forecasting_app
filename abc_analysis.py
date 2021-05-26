import numpy as np 
import pandas as pd
import datetime
from datetime import datetime, date
from datetime import timedelta
from time import time
from warnings import catch_warnings
from warnings import filterwarnings
from joblib import delayed
from joblib import Parallel
from multiprocessing import cpu_count
from math import sqrt
from time import time
import json
import sqlalchemy
import psycopg2
import warnings
warnings.filterwarnings("ignore")


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')


def abc_analysis(firm_id, data_abc):
    firm_id = int(firm_id)

#Extracting sales data
    data_abc = data_abc[data_abc["delta"] < 0]
    data_abc["delta"] = data_abc["delta"].abs()

# obsolete_items = data_abc[data_abc['transaction_date']<'2018-01-01']
    today = pd.to_datetime("today")
    obsolete_time = (today - pd.DateOffset(years=2)).to_period("M")
    data_abc = data_abc[data_abc['transaction_date']
                                > obsolete_time]

# non-stocked items
    # data_abc = data_abc[data_abc['transaction_date']>'2019-01-01']

#calculating revenue
    no_of_items = len(data_abc)
    data_abc['revenue'] = data_abc.price * data_abc.delta

# ABC analysis
    versa_sales_abc = data_abc.sort_values(
        by='revenue', ascending=False)
    A, B, C = np.split(versa_sales_abc, [
                        int(.2*no_of_items), int(.5*no_of_items)])

# HML analysis
    versa_sales_hml = data_abc.sort_values(by='delta', ascending=False)
    high, medium, low = np.split(
        versa_sales_hml, [int(.2*no_of_items), int(.5*no_of_items)])

# Calculating items for ABC matrix table
    ha = pd.merge(high, A, how='inner')
    if not ha.empty:
        ha_len = len(ha)
        ha_value = int(ha['revenue'].sum())
        ha_demand =int(ha['delta'].sum())
    else:
        ha_demand = 0 
        ha_len= 0
        ha_value = 0

    hb = pd.merge(high, B, how='inner')
    if not hb.empty:
        hb_len = len(hb)
        hb_value = int(hb['revenue'].sum())
        hb_demand =int(hb['delta'].sum())
    else:
        hb_demand = 0 
        hb_len= 0
        hb_value = 0

    hc = pd.merge(high, C, how='inner')
    if not hc.empty:
        hc_len = len(hc)
        hc_value = int(hc['revenue'].sum())
        hc_demand =int(hc['delta'].sum())
    else:
        hc_demand = 0 
        hc_len= 0
        hc_value = 0

    ma = pd.merge(medium, A, how='inner')
    if not ma.empty:
        ma_len = len(ma)
        ma_value = int(ma['revenue'].sum())
        ma_demand =int(ma['delta'].sum())
    else:
        ma_demand = 0 
        ma_len= 0
        ma_value = 0

    mb = pd.merge(medium, B, how='inner')
    if not mb.empty:
        mb_len = len(mb)
        mb_value = int(mb['revenue'].sum())
        mb_demand =int(mb['delta'].sum())
    else:
        mb_demand = 0 
        mb_len= 0
        mb_value = 0

    mc = pd.merge(medium, C, how='inner')
    if not mb.empty:
        mc_len = len(mc)
        mc_value = int(mc['revenue'].sum())
        mc_demand =int(mc['delta'].sum())
    else:
        mc_demand = 0 
        mc_len= 0
        mc_value = 0
        
    la = pd.merge(low, A, how='inner')
    if not la.empty:
        la_len = len(la)
        la_value = int(la['revenue'].sum())
        la_demand =int(la['delta'].sum())
    else:
        la_demand = 0 
        la_len= 0
        la_value = 0

    lb = pd.merge(low, B, how='inner')
    if not lb.empty:
        lb_len = len(lb)
        lb_value = int(lb['revenue'].sum())
        lb_demand =int(lb['delta'].sum())
    else:
        lb_demand = 0 
        lb_len= 0
        lb_value = 0

    lc = pd.merge(low, C, how='inner')
    if not lc.empty:
        lc_len = len(lc)
        lc_value = int(lc['revenue'].sum())
        lc_demand =int(lc['delta'].sum())
    else:
        lc_demand = 0 
        lc_len= 0
        lc_value = 0

    items = {
        'ha': ha.part_number.tolist(),
        'hb': hb.part_number.tolist(),
        'hc': hc.part_number.tolist(),
        'ma': ma.part_number.tolist(),
        'mb': mb.part_number.tolist(),
        'mc': mc.part_number.tolist(),
        'la': la.part_number.tolist(),
        'lb': lb.part_number.tolist(),
        'lc': lc.part_number.tolist(),

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
    data = {
        'items': items,
        'values': values,
        'demand': demand
    }

    conn = psycopg2.connect(
        database="versa_db",
        user="postgres",
        password="bits123",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    query = '''
    SELECT *
    FROM sales_abc_analyses
    WHERE firm_id = '%(firm_id)d' ''' % {'firm_id': firm_id}
    engine = sqlalchemy.create_engine('postgresql://postgres:bits123@localhost:5432/versa_db')
    # SQL injection
    abc_table = pd.read_sql_query(query, engine)
    if abc_table.empty:
        cur.execute('INSERT INTO sales_abc_analyses (firm_id, abc_response_json) values (%s, %s)',
        (firm_id, str(data))) 
        conn.commit()
    else:
        cur.execute('UPDATE sales_abc_analyses SET abc_response_json = %s WHERE firm_id = %s',
        (str(data), firm_id))
        conn.commit()
    cur.close()
    conn.close()
    return  (data)