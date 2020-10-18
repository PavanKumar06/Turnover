from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import numpy as np
import pandas as pd
import keras
import pickle


# Create View Here

@api_view(["GET"])
def Train(self):
    try:
        dataset = pd.read_csv('EmpTurnover.csv')
        X = dataset.iloc[:, 0:26].values 
        y = dataset.iloc[:, 26].values 
        
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        labelencoder = LabelEncoder() 
        X[:, 5] = labelencoder.fit_transform(X[:, 5]) 
        X[:, 20] = labelencoder.fit_transform(X[:, 20]) 
        y = labelencoder.fit_transform(y) 
        ct = ColumnTransformer([("Reform", OneHotEncoder(), [1, 3, 6, 13, 14])], remainder = 'passthrough')
        X = np.array(ct.fit_transform(X), dtype=np.float) 
        X = np.delete(X, [0, 3, 9, 12, 15], axis=1)
        
        from sklearn.preprocessing import StandardScaler 
        sc = StandardScaler()
        X = sc.fit_transform(X)
        
        binary_file = open('ObjStore.bin', mode='wb')
        pickle.dump(sc, binary_file)
        binary_file.close()
        
        
        from keras.models import Sequential 
        from keras.layers import Dense 
        from keras.layers import Dropout 
        
        classifier = Sequential() 
               
        classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 40))
        classifier.add(Dropout(p = 0.1))
            
        classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.1))
         
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        classifier.fit(X, y, batch_size = 10, epochs = 100)
        
        classifier.save('ObjectState')
        
        return JsonResponse("Model Trained", safe=False)
        
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)



@api_view(["GET"])    
def Predict(self):
    try:
        import sqlite3
        from datetime import datetime
        from keras import backend as K
        import codecs, json 

        
        K.clear_session()
        model = keras.models.load_model('ObjectState')
        
        with open('ObjStore.bin', 'rb') as pickle_file:
            sc1 = pickle.load(pickle_file)
        
        con = sqlite3.connect('TurnoverPredictor.db')
        
        df = pd.read_sql_query("SELECT DateOfBirth, Department, DistanceFromHome, EducationField, Education, Gender, MaritalStatus, NumCompaniesWorked, TotalWorkingYears, DateOfJoining, LastRoleUpdate, LastPromotionUpdate, LastManagerUpdate, BusinessTravel, JobRole, AnnualIncome, StockOptionLevel, TrainingTimesLastYear, JobInvolvement, JobLevel, OverTime, PercentSalaryHike, PerformanceRating, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance FROM ((Users INNER JOIN UserPerformances ON Users.Id = UserPerformances.UserId) INNER JOIN UserFeedbacks ON Users.Id = UserFeedbacks.UserId)", con)
        
        for col in ['DateOfBirth', 'Department', 'EducationField', 'Education', 'Gender', 'MaritalStatus', 'DateOfJoining', 'LastRoleUpdate', 'LastPromotionUpdate', 'LastManagerUpdate', 'BusinessTravel', 'JobRole', 'StockOptionLevel', 'TrainingTimesLastYear', 'JobInvolvement', 'OverTime', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        for col in ['DistanceFromHome', 'NumCompaniesWorked', 'TotalWorkingYears', 'AnnualIncome', 'PercentSalaryHike', 'PerformanceRating']:
            df[col] = df[col].fillna(df[col].mean())
        
        X_pred = df.iloc[:, 0:26].values 
        
        col = [0, 9, 10, 11, 12]
        for j in col:
            for i in range(len(X_pred[:, j])):
                if (X_pred[:, j][i].find(".")):
                    X_pred[:, j][i] = X_pred[:, j][i][0: X_pred[:, j][i].find(".")] 
                X_pred[:, j][i] = datetime.strptime(X_pred[:, j][i], '%Y-%m-%d %H:%M:%S') 
                X_pred[:, j][i] = int((datetime.today() - X_pred[:, j][i]).days/365)
                
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        labelencoder = LabelEncoder() 
        X_pred[:, 5] = labelencoder.fit_transform(X_pred[:, 5]) 
        X_pred[:, 20] = labelencoder.fit_transform(X_pred[:, 20]) 
        ct = ColumnTransformer([("Reform", OneHotEncoder(), [1, 3, 6, 13, 14])], remainder = 'passthrough')
        X_pred = np.array(ct.fit_transform(X_pred), dtype=np.float) 
        X_pred = np.delete(X_pred, [0, 3, 9, 12, 15], axis=1)
        
        X_pred = sc1.transform(X_pred)
        
        y_pred = model.predict(X_pred)
        
        count = 0
        for i in y_pred:
            if(i > 0.2):
                count = count + 1                  
        
        turnover = round((count/len(y_pred))*100, 2)
        
        y_pred = np.append(turnover, y_pred)
        
        y_pred = y_pred.tolist()
        file_path = "Prediction.json"
        json.dump(y_pred, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        
        K.clear_session()

        return JsonResponse(y_pred, safe=False)
        

    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)


