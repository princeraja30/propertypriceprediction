import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# IMPORTING DATASET
dataset=pd.read_csv("/content/HousePrices.csv")
print(dataset.head())
dataset.info()
#TAKING CARE OF NULL VALUES
dataset.isna().sum()
dataset.drop(["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"],axis=1,inplace=True)
dataset.info()
dataset["LotFrontage"]=dataset["LotFrontage"].fillna(dataset["LotFrontage"].mean())

dataset=dataset.fillna(dataset.bfill())

dataset["GarageYrBlt"]=dataset["GarageYrBlt"].fillna(dataset["GarageYrBlt"].mean())
dataset["MasVnrArea"]=dataset["MasVnrArea"].fillna(dataset["MasVnrArea"].mean())
dataset.info()
dataset.isnull().sum()
sns.boxplot(dataset["Property_Sale_Price"])
sns.boxplot(dataset["LotFrontage"])
# TAKING CARE OF OUTLIERS

q1=dataset["Property_Sale_Price"].quantile(0.25)
q3=dataset["Property_Sale_Price"].quantile(0.75)
iqr=q3-q1
lower_lmt=q1-1.5*iqr
upper_lmt=q3+1.5*iqr
dataset=dataset[((dataset["Property_Sale_Price"]>lower_lmt) & (dataset["Property_Sale_Price"]<upper_lmt))]
 
q1=dataset["LotFrontage"].quantile(0.25)
q3=dataset["LotFrontage"].quantile(0.75)
iqr=q3-q1
print(q1,q3)
lower_lmt=q1-1.5*iqr
upper_lmt=q3+1.5*iqr
dataset=dataset[((dataset["LotFrontage"]>lower_lmt) & (dataset["LotFrontage"]<upper_lmt))]
sns.boxplot(dataset["Property_Sale_Price"])
sns.boxplot(dataset["LotFrontage"])
sns.boxplot(dataset["Dwell_Type"])
q1=dataset["Dwell_Type"].quantile(0.25)
q3=dataset["Dwell_Type"].quantile(0.75)
iqr=q3-q1
print(q1,q3)
lower_lmt=q1-1.5*iqr
upper_lmt=q3+1.5*iqr
dataset=dataset[((dataset["Dwell_Type"]>lower_lmt) & (dataset["Dwell_Type"]<upper_lmt))]
sns.boxplot(dataset["Dwell_Type"])
dataset.info()
# ENCODING
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
def encode(data,columns):
  for column in columns:
    data[column]=le.fit_transform(data[column])
  return data
columns=dataset.select_dtypes(include ='object')
encode(dataset,columns)
# SPLITTING THE DATASET
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
print(x_train)
print(y_train)
# APPLYING LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))