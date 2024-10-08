# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JAVITH M
RegisterNumber: 212222110014
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
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
## Data.head()
![image](https://github.com/user-attachments/assets/86fcefb5-d00f-4f4b-93fd-e10ce7d8f070)

## Data.info()
![image](https://github.com/user-attachments/assets/dc78d66a-a8a7-4524-a589-93c6a84ba4ea)

## Data.isnull().sum()
![image](https://github.com/user-attachments/assets/310dc6d2-6c19-4d2e-a051-6d7377b04756)

## Data value count
![image](https://github.com/user-attachments/assets/868949a4-8ec0-4965-a585-835b644817e2)

##   Data.head() for salary
![image](https://github.com/user-attachments/assets/4259ac0f-1b68-4e46-8f03-9bf2c0ef6213)

## x.head()
![image](https://github.com/user-attachments/assets/3ec27a02-2a40-4788-a493-43055fface58)

## Accuracy value
![image](https://github.com/user-attachments/assets/fafd3b5c-6264-4fe4-a499-48b879ae001b)

## Data prediction
![image](https://github.com/user-attachments/assets/1b823bfc-5108-41f6-8618-a57d059f99fe)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
