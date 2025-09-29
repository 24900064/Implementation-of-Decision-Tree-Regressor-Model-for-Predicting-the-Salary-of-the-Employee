# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Pragatheeshraaj D
RegisterNumber: 212224230199


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

print("Name : Pragatheeshraaj D")
print("Register No.: 212224230199")
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
r2=metrics.r2_score(y_test,y_pred)
print("Mean Square Error: ",mse)

r2=metrics.r2_score(y_test,y_pred)
print("Root Mean Square Error: ",r2)

dt.predict([[5,6]])
*/
```

## Output:
<img width="1251" height="218" alt="image" src="https://github.com/user-attachments/assets/c711298f-9a94-4bf3-a235-4b73f85b992f" />
<img width="1279" height="234" alt="image" src="https://github.com/user-attachments/assets/482bcbc5-bac1-42ec-aba4-f3f054c94ab8" />
<img width="1275" height="107" alt="image" src="https://github.com/user-attachments/assets/c2a58000-e2f6-4011-aae1-11a47a875191" />
<img width="1250" height="43" alt="image" src="https://github.com/user-attachments/assets/793734ad-2a74-4954-8dc8-91da0cfd7635" />
<img width="1246" height="102" alt="image" src="https://github.com/user-attachments/assets/dee4b398-f2a0-4de5-8430-e49763476b49" />
<img width="1232" height="136" alt="image" src="https://github.com/user-attachments/assets/37056209-f439-435f-b19b-613fcad79dba" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
