# Sales Forecasting<a name="TOP"></a>
Forecasting the demand for a particular product can be very beneficial for a company. It reduces the likelihood of backorders and also provides data that can help with prioritizing sales of various products to maximize profit. A time-series forecasting method called SARIMA was chosen for forecasting sales.

# ABC analysis #
A report to be added to the inventory management system is ABC analysis. ABC analysis provides the client with a visual representation of the “important products” (around 20% of total products make up this tier) which account for the majority of their revenue, while also isolating the products that do not provide significant revenue and/or are sold at low frequencies. This classification is based on the Pareto Principle that states that 80% of the sales come from just 20% of the products.

# Flask App #
For the model to be used efficiently by the organization it had to be converted into an API which can be called to generate predictions based on the firm and item id. Sqlalchemy and psycopg2 were used to dynamically query the data from Versa’s PostgreSQL database. SQL queries were performed to retrieve and write the data on Versa’s database and are secure of SQL Injection attacks. The three APIs developed were routed with different function calls to take in the firm_id and item_id and on post request, the first API(/grid) performs the grid search and stores them in forecasting_parameters table while the second one(/predict) takes these parameters to generate the prediction and save them in sales_predictions table and the third one(/abc) takes the firm_id and gives a JSON response for the ABC matrix.


- - - - 
# Installation Guide #

    


## 1. Install Python 3.8 ## 
## 2. Install PIP3 ##
## 3. Install python3-venv ##

    sudo apt install python3-venv
  
## 4. Create a virtual environment ##

    python3 -m venv versa-env
  
## 5. Activate the virtual environment ##

    source versa-env/bin/activate
## 6. Install Required Packages ##

    pip install -r requirements.txt
    
## 7. Run Flask app ##

    flask run --host=0.0.0.0
