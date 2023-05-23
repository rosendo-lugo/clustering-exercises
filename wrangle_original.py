import pandas as pd
import numpy as np
from env import get_db_url
import os


from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------
def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df

    
# ----------------------------------------------------------------------------------
def get_zillow_data():
    # How to import a database from MySQL
    url = get_db_url('zillow')
    query = '''
    select *
    from properties_2017 p
        join predictions_2017 p2 using (parcelid)
    where p.propertylandusetypeid = 261 and 279
            '''
    filename = 'zillow.csv'
    df = check_file_exists(filename, query, url)
    
    # filter to just 2017 transactions
    df = df[df['transactiondate'].str.startswith("2017", na=False)]
    
    # split transaction date to year, month, and day
    df_split = df['transactiondate'].str.split(pat='-', expand=True).add_prefix('transaction_')
    df = pd.concat([df.iloc[:, :40], df_split, df.iloc[:, 40:]], axis=1)
    
    # Drop duplicate rows in column: 'parcelid', keeping max transaction date
    df = df.drop_duplicates(subset=['parcelid'])
    
    # rename columns
    df.columns
    df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms',
                            'calculatedfinishedsquarefeet':'area','taxvaluedollarcnt':'property_value',
                            'fips':'county','transaction_0':'transaction_year',
                            'transaction_1':'transaction_month','transaction_2':'transaction_day'})
    
    # total outliers removed are 6029 out of 52442
    # # Look at properties less than 1.5 and over 5.5 bedrooms (Outliers were removed)
    # df = df[~(df['bedrooms'] < 1.5) & ~(df['bedrooms'] > 5.5)]

    # Look at properties less than .5 and over 4.5 bathrooms (Outliers were removed)
    df = df[~(df['bathrooms'] < .5) & ~(df['bathrooms'] > 4.5)]

    # Look at properties less than 1906.5 and over 2022.5 years (Outliers were removed)
    df = df[~(df['yearbuilt'] < 1906.5) & ~(df['yearbuilt'] > 2022.5)]

    # Look at properties less than -289.0 and over 3863.0 area (Outliers were removed)
    df = df[~(df['area'] < -289.0) & ~(df['area'] > 3863.0)]

    # Look at properties less than -444576.5 and over 1257627.5 property value (Outliers were removed)
    df = df[~(df['property_value'] < -444576.5) &  ~(df['property_value'] > 1257627.5)]
    
    # replace missing values with "0"
    df = df.fillna({'bedrooms':0,'bathrooms':0,'area':0,'property_value':0,'county':0})
    
    # drop any nulls in the dataset
    df = df.dropna()
    
    # drop all duplicates
    df = df.drop_duplicates(subset=['parcelid'])
    
    # change the dtype from float to int  
    df[['bedrooms','area','property_value','yearbuilt','transaction_month','transaction_day']] = df[['bedrooms','area','property_value','yearbuilt','transaction_month','transaction_day']].astype(int)
    
    # rename the county codes inside county
    df['county'] = df['county'].map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    
    # get dummies and concat to the dataframe
    dummy_tips = pd.get_dummies(df[['county']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_tips], axis=1)
    
    # dropping these columns for right now until I find a use for them
    df = df.drop(columns =['parcelid','transactiondate','transaction_year','transaction_month','transaction_day'])
    
    # Define the desired column order
    new_column_order = ['bedrooms','bathrooms','area','yearbuilt','county','county_Orange','county_Ventura','property_value',]

    # Reindex the DataFrame with the new column order
    df = df.reindex(columns=new_column_order)

    # write the results to a CSV file
    df.to_csv('df_prep.csv', index=False)

    # read the CSV file into a Pandas dataframe
    prep_df = pd.read_csv('df_prep.csv')
    
    return df.set_index('customer_id'), prep_df

# ----------------------------------------------------------------------------------
def get_split(df):
    '''
    train=tr
    validate=val
    test=ts
    test size = .2 and .25
    random state = 123
    '''  
    # split your dataset
    # 80/20
    train_validate, ts = train_test_split(df, test_size=.2, random_state=123)
    # 75/25
    tr, val = train_test_split(train_validate, test_size=.25, random_state=123)
    
    return tr, val, ts

# ----------------------------------------------------------------------------------
# def scale_data(train,validate,test,to_scale):
#     """
#     to_scale = ['column1','column2','column3','column4','column5']
#     """
    
#     #make copies for scaling
#     train_scaled = train.copy()
#     validate_scaled = validate.copy()
#     test_scaled = test.copy()

#     #scale them!
#     #make the thing
#     scaler = MinMaxScaler()

#     #fit the thing
#     scaler.fit(train[to_scale])

#     #use the thing
#     train_scaled[to_scale] = scaler.transform(train[to_scale])
#     validate_scaled[to_scale] = scaler.transform(validate[to_scale])
#     test_scaled[to_scale] = scaler.transform(test[to_scale])
    
#     return train_scaled, validate_scaled, test_scaled


# ----------------------------------------------------------------------------------
def get_Xs_ys_to_scale_baseline(tr_m, val_m, ts_m, target):
    '''
    tr = train
    val = validate
    ts = test
    target = target value
    '''

    # Separate the features (X) and target variable (y) for the training set
    X_tr, y_tr = tr_m.drop(columns=[target,'gender']), tr_m[target]
    
    # Separate the features (X) and target variable (y) for the validation set
    X_val, y_val = val_m.drop(columns=[target,'gender']), val_m[target]
    
    # Separate the features (X) and target variable (y) for the test set
    X_ts, y_ts = ts_m.drop(columns=[target,'gender']), ts_m[target]
    
    # Get the list of columns to be scaled
    to_scale = X_tr.columns.tolist()
    
    # Calculate the baseline (mean) of the target variable in the training set
    baseline = y_tr.mean()
    
    # Return the separated features and target variables, columns to scale, and baseline
    return X_tr, X_val, X_ts, y_tr, y_val, y_ts, to_scale, baseline
# ----------------------------------------------------------------------------------

def scale_data(X,Xv,Xts,to_scale):
    '''
    X = X_train
    Xv = X_validate
    Xts = X_test
    to_scale, is found in the get_Xs_ys_to_scale_baseline
    '''
    
    #make copies for scaling
    X_tr_sc = X.copy()
    X_val_sc = Xv.copy()
    X_ts_sc = Xts.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(X[to_scale])

    #use the thing
    X_tr_sc[to_scale] = scaler.transform(X[to_scale])
    X_val_sc[to_scale] = scaler.transform(Xv[to_scale])
    X_ts_sc[to_scale] = scaler.transform(Xts[to_scale])
    
    return X_tr_sc, X_val_sc, X_ts_sc
# ----------------------------------------------------------------------------------
# remove all outliers put each feature one at a time
def outlier(df, feature, m=1.5):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = m
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound

# upper_bound, lower_bound = outlier(df, 'bedroomcnt')

# ----------------------------------------------------------------------------------

def get_mallcustomer_data():
    """
    This function will:
        - read the given `sql_query`
        - from the connection made to the `mall_customers` DB
            - using the `get_connection_url` from my wrangle module.
            
        - output a df with the mall `customer_id` set as it's index
                - `customer_id` is the table's PK. 
                    This id is an attribute of the table but will not be used as a feature to investigate.
    """
    
    sql_query = 'SELECT * FROM customers;'
    url = get_db_url('mall_customers')
    
    df = pd.read_sql(sql_query, url)
    
    return df.set_index('customer_id')

# ----------------------------------------------------------------------------------
def get_iris_data():
    # How to import a database from MySQL
    url = get_db_url('iris_db')
    query = '''
    SELECT *
    FROM species
    '''
    filename = 'iris.csv'
    df = check_file_exists(filename, query, url)
    
#   # Define the desired column order
#     new_column_order = ['bedrooms','bathrooms','area','yearbuilt','county','county_Orange','county_Ventura','property_value',]

#     # Reindex the DataFrame with the new column order
#     df = df.reindex(columns=new_column_order)

#     # write the results to a CSV file
#     df.to_csv('df_prep.csv', index=False)

#     # read the CSV file into a Pandas dataframe
#     prep_df = pd.read_csv('df_prep.csv')
    
    return df



