import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
def algo_run(ind):
    filename = 'fake.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred_rf=loaded_model.predict(ind)

    return y_pred_rf


def msg_process():
    fraud_data= pd.read_csv('inputdata/inputdata.csv')
    ip_country=pd.read_csv('Data/IpAddress_to_Country.csv')

    # Convert to date time type
    fraud_data['purchase_time']=pd.to_datetime(fraud_data['purchase_time'], errors='coerce')
    fraud_data['signup_time']=pd.to_datetime(fraud_data['signup_time'], errors='coerce')
    fraud_data['difference']=fraud_data['purchase_time']-fraud_data['signup_time']
    fraud_data['difference']=fraud_data['difference'].astype('timedelta64[m]')

    # Only get the hours from singup time and purchase time
    fraud_data['signup_hour']=fraud_data['signup_time'].dt.hour
    fraud_data['purchase_hour']=fraud_data['purchase_time'].dt.hour

    # Get the day of week
    fraud_data['purchase_dayofweek']=fraud_data['purchase_time'].dt.strftime('%A')

    #Encode the categorical values into numerical values
    fraud_data['purchase_dayofweek'], key_dayofweek=fraud_data['purchase_dayofweek'].factorize(sort=True)

    # Endcode this fearture
    fraud_data['device_id'], key_device=fraud_data['device_id'].factorize(sort=True)

    #Define two new columns
    u, indices, counts=np.unique(fraud_data['device_id'], return_inverse=True, return_counts=True)
    fraud_data['usage_device_count']= counts[indices]
    u, indices, counts=np.unique(fraud_data['ip_address'], return_inverse=True, return_counts=True)
    fraud_data['usage_ip_count']= counts[indices]

    #Encode this fearture
    fraud_data['sex'], key_sex=fraud_data['sex'].factorize(sort=True)
    #Encode this fearture
    fraud_data['source'], key_source=fraud_data['source'].factorize(sort=True)
    #Encode this fearture
    fraud_data['browser'], key_browser=fraud_data['browser'].factorize(sort=True)

    # Function to map
    def map_ip_address(i):
        result=ip_country[(ip_country['lower_bound_ip_address'].le(i))&
                        (ip_country['upper_bound_ip_address'].ge(i))]['country'].values
        if result.shape[0] >0:
            return result[0]
        else:
            return np.NaN

    # Mapping
    fraud_data['country']=fraud_data['ip_address'].map(map_ip_address)
    #Encode
    fraud_data['country'], key=fraud_data['country'].factorize(sort=True)
    fraud_data.to_csv('Data/data_after_process.csv')

    X=fraud_data  
    X['signup_time'] = pd.to_datetime(X['signup_time'],infer_datetime_format=True)
    X['signup_time']=X['signup_time'].apply(lambda x: x.toordinal())
    X['purchase_time'] = pd.to_datetime(X['purchase_time'],infer_datetime_format=True)
    X['purchase_time']=X['purchase_time'].apply(lambda x: x.toordinal())
    print(X)
    ypred=algo_run(X)

    return ypred    
