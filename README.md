# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.Sajetha
RegisterNumber: 212223100049

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
*/

```

## Output:
Data.head():

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/621c6af1-d694-47de-92c4-500d868d2461)


Data.info():

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/3bf9def7-0241-49dc-94ae-1e30ed15c1f5)


isnull() and sum():

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/9d775f74-b34d-4f00-b520-e00fd2fcbde0)


Data Value Counts():

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/02e21d36-6701-4127-b854-e02ada1f05e8)


Data.head() for salary:

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/d5a06237-8ccf-4e2d-90bd-1057d6c5a434)


x.head:
![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/9cc4ff77-63de-4fb3-b9de-a2a008d326eb)



Accuracy Value:

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/74c7e7ba-b07c-4c1f-b9cb-ee0810dae164)


Data Prediction:

![image](https://github.com/Sajetha13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849316/a43f66b3-aeb8-41f7-8cfa-8c9524fa95d8)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
