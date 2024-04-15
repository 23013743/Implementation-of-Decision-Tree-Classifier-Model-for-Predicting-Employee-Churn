# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NIRMAL.n
RegisterNumber: 212223240107
*/

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
````````

## Output:

##  Data.head():

     ![Screenshot 2024-04-15 035837](https://github.com/23013743/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/203ddd65-b86f-4014-98af-aacf3b9b3be9)


##   Data.info():
        ![Screenshot 2024-04-15 040000](https://github.com/23013743/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/8c185ecf-3457-434a-b008-429ff02cb33f)

   
##  isnull() and sum():    
      ![Screenshot 2024-04-15 040218](https://github.com/23013743/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/c1c150a6-1aa9-40f4-8f95-97cbeed2e7c7)


##    Data Value Counts():
    ![Screenshot 2024-04-15 040230](https://github.com/23013743/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/b07718ec-e447-4740-8b8a-f5fc6328995f)


##   Data.head() for salary:
    ![Screenshot 2024-04-15 040411](https://github.com/23013743/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/c8a43d3e-b427-4cf5-96ef-707fd3935647)


##   x.head:
     ![Screenshot 2024-04-15 040419](https://github.com/23013743/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/9c1f4de9-043a-4d60-914d-981d53d02e3f)

##    Accuracy Value:
      ![Screenshot 2024-04-15 040425](https://github.com/23013743/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/947fef8a-1cb4-4ab8-b169-52eb8bd80e30)

##   Data Prediction:
     ![Screenshot 2024-04-15 040500](https://github.com/23013743/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/428b286a-ee1a-4c8a-97e8-1acfa48626e4)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
