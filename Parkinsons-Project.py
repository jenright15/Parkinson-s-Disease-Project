
#### This is a project that attempts to create a model that can predict Parkinsons Disease

## Needed Libraries
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Read the data
df = pd.read_csv('C:\\Users\\johne\\Desktop\\Python Projects\\parkinsons.data')
df.head()

## Observe features and data labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values

## Counts for the status column (0-1)
print(labels[labels==1].shape[0], labels[labels==0].shape[0])

## Normalizing the features
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

## Test and Train
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

## Training our model
model=XGBClassifier()
model.fit(x_train,y_train)

## Model Accuracy
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100) #94.87%
