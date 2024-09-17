from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
import os
import numpy as np
import pickle
import time
import graphviz
import pandas as pd
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from class_algo import * 
from sklearn import linear_model
app = Flask(__name__, static_folder='static')

@app.route("/")
def index():
    return render_template('home.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    USER = str(request.form['USER'])
    signuptime = str(request.form['signuptime'])
    purchase_time = str(request.form['purchase_time'])
    purchase_value = str(request.form['purchase_value'])
    device_id = str(request.form['device_id'])
    source = str(request.form['source'])
    browser = str(request.form['browser'])
    sex = str(request.form['sex'])
    age = str(request.form['age'])
    ip_address = str(request.form['ip_address'])
    col_names =  ['user_id','signup_time','purchase_time','purchase_value','device_id','source','browser','sex','age','ip_address']
    attendance = pd.DataFrame(columns = col_names)
    attendance.loc[len(attendance)] = [USER,signuptime,purchase_time,purchase_value,device_id,source,browser,sex,age,ip_address]
    fileName="inputdata\inputdata.csv"
    attendance.to_csv(fileName,index=False)
    data=msg_process()
    if data == 1:
        first="Fraud Activity Predicted in Online Shopping"
    else:
        first=" Genuine Transaction"
     
    
    
    return render_template('Result.htm', first=str(first))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
