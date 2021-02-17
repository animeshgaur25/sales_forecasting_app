import numpy as np 
import pandas as pd
import datetime
from datetime import datetime, date
from datetime import timedelta
from time import time
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
import sqlalchemy
import psycopg2

def abc_analysis(firm_id, versa_sales):

    firm_id=int(firm_id)
    # obsolete items

	# obsolete_items = versa_sales[versa_sales['transaction_date']<'2018-01-01']
	versa_sales = versa_sales[versa_sales['transaction_date']>'2018-01-01']

	# non-stocked items
	versa_sales = versa_sales[versa_sales['transaction_date']>'2019-01-01']

	# abc analysis
    total_sales = versa_sales[versa_sales['delta'] < 0]
    
	# HML analysis
    total_sales =total_sales.groupby(['inventory_item_id']).sum()
    total_sales = total_sales.sort_values(by='delta')
    no_of_items = len(total_sales)
    high, medium, low = np.split(total_sales, [int(.2*no_of_items), int(.5*no_of_items)])
    return high, medium, low