# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1: Start
#### Step 2: Import the required libraries.
#### Step 3: Upload and read the dataset.
#### Step 4: Check for any null values using the isnull() function.
#### Step 5: From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
#### Step 6: Find the accuracy of the model and predict the required values by importing the required module from sklearn.
#### Step 7: End

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Mahasri P
RegisterNumber: 212223100029
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
Data.head():

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/9bcaad51-9aea-41a3-8ef2-d3ce7f06958d)


Data.info():

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/b7704ad6-9b7c-4d44-bb50-b02cda461721)


isnull() and sum():

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/673f7e80-404c-4884-9cec-cb25d5319056)


Data Value Counts():

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/7e3b45df-a46a-4817-8aad-284ce5112b18)


Data.head() for salary:

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/fc253d0f-08cb-40ae-a87d-dca204a30af8)


x.head:

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/23629d0d-1496-4083-9974-d8e67a9714be)


Accuracy Value:

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/31074d56-b702-4696-8373-44d982ca1d40)


Data Prediction:

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/f3d4c1ef-123c-4c1e-bd27-9bae684e013c)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
