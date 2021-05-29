#!/usr/bin/python3
import threading
import time
import sys
import argparse
import requests as reqs
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
from urllib import response
register_matplotlib_converters()

def main():
    # Parser arguments
    parser = argparse.ArgumentParser(description='Versa Inv POC')
    parser.add_argument(
        'thread', help='Input the number of concurrent thread.', type=int)
    parser.add_argument('count',
                        help='Input the number count value for the thread. The number in count will decrement until it reaches 0.',
                        type=int)
    parser.add_argument('-s', '--synchronize', help='Run the thread with synchronize. Default is false',
                        action="store_true")
    parser.add_argument(
        '-d', '--delay', help='Give a delay to threads in seconds. Default is 2', default=2, type=int)
    args = parser.parse_args()

    # Initialize the arguments
    threadNumber = args.thread
    count = args.count
    synchronize = args.synchronize
    delay = args.delay

    # Test the arguments, to avoid problems
    if threadNumber <= 0 or count <= 0 or delay <= 0:
        print("[!] Argument cannot be zero or less than zero. Exiting")
        sys.exit(2)

    # Print out current settings
    print('[i] Please specify', threadNumber, 'number of thread(s).')
    print('[i] Every thread will count', count,
          'concurrently until it reaches 0.')
    print('[i] Synchronize is set to', synchronize)
    print('[i] Delay value is', delay, 'seconds\n')

    # Initialize threadNumber in a list
    threadCount = []
    while threadNumber > 0:
        threadCount.append(threadNumber)
        threadNumber -= 1
    threadCount.reverse()

    # Initialize threads into a list threads
    threads = []
    # Create new threads
    for i in threadCount:
        i = VersaInvThread(i, "Thread-" + str(i), count, synchronize, delay)
        threads.append(i)

    # Start new Threads
    for i in threads:
        i.start()

    # Wait for all threads to complete
    for i in threads:
        i.join()

    print("Done. Exiting Main Thread")


class VersaInvThread(threading.Thread):
    def __init__(self, threadID, name, counter, synchronize, delay):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.synchronize = synchronize
        self.delay = delay
        self.session = reqs.session()

    def run(self):
        print("[+] Starting " + self.name)
        threadLock = threading.Lock()

        # Get lock to synchronize threads
        if self.synchronize:
            threadLock.acquire()

        # Animesh please add any code to call in thread in this function
        thread_count(self.name, self.delay, self.counter, self.session)

        # Free lock to release next thread
        if self.synchronize:
            threadLock.release()

# Animesh please add any code to call in thread in this function


def data_pts_check(versa_sm):
    if (len(versa_sm.index)) < 12:
        print(-1)
    return flask.render_template('main_2.html', result="More Historical Sales Data required ",)

def thread_count(threadName, delay, counter, session):
    while counter:
        time.sleep(delay)
        ptime = time.ctime(time.time())
        print("%s: %s" % (threadName, response))
        #print("%s: %s" % (threadName, counter))
        
    counter -= 1

def extract_data():
    engine = sqlalchemy.create_engine(
            'postgresql://root:root@localhost:5432/development_master')

    query1 = '''
    SELECT inventory_items.id , inventory_items.firm_id, inventory_transaction_details.delta,	inventory_transaction_details.transaction_date
    FROM inventory_transaction_details
    INNER JOIN inventory_items ON inventory_transaction_details.inventory_item_id=inventory_items.id 
    ORDER BY inventory_item_id;
    '''

    versa_sales = pd.red_sql_query(query1, engine)
    versa_sales1 = versa_sales[versa_sales["delta"] < 0]
    versa_sales1["delta"] = versa_sales1["delta"].abs()

    versa_sales1 = versa_sales1.groupby(['firm_id', 'id', 'transaction_date'], as_index=False).sum()
    return versa_sales1

def call_grid_search(versa_sales1, item_id):
    firm_id = versa_sales1.loc[versa_sales1['inventory_item_id']== item_id, 'firm_id'].iloc[0]
    print(id)
    engine = sqlalchemy.create_engine(
        'postgresql://root:root@localhost:5432/development_master')
    query = '''
    SELECT *
    FROM forecasting_parameters
    WHERE inventory_item_id = '%(item_id)d' AND firm_id = '%(firm_id)d' ''' % {'item_id': item_id, 'firm_id': firm_id}
    # SQL injection
    para_table = pd.read_sql_query(query, engine)
    if len(para_table.index)==1:
        print("para table:", para_table)
        flag = para_table.loc[para_table['inventory_item_id']== item_id, 'flag'].iloc[0]
        if flag == 0:
            return "Prediction engine is searching for best parameters"
        elif flag == 1:
            return 
            

    versa_sm = grid.data_preprocessing(versa_sales1, item_id)
    if versa_sm is not None:
        data_pts_check(versa_sm)
        conn = psycopg2.connect(
            database="development_master",
            user="root",
            password="root",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        cur.execute("INSERT into forecasting_parameters (inventory_item, firm_id,p,d,q,seasonal_p,seasonal_d,seasonal_q,s,flag) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (item_id, firm_id, -1, -1, -1, -1, -1, -1, -1, 0))
        conn.commit()
        cur.close()
        conn.close()
        t = Thread(target=grid.user_inp_grid_search,
                    args=(item_id, firm_id, versa_sm),)
        t.start()
