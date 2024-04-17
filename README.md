# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


Import the required libraries.
Upload and read the dataset.
Check for any null values using the isnull() function.
From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
Find the accuracy of the model and predict the required values by importing the required module from sklearn.
Program:

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
Developed by: NIRMAL.N
RegisterNumber: 212223240107
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
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
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
![decision tree classifier model](sam.png)
## HEAD() AND INFO():
   ![Screenshot 2024-04-17 100041](https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/b39eb792-0496-42ef-85bd-27abb14928bd)

## NULL & COUNT:
   ![Screenshot 2024-04-17 100058](https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/38559536-42e8-4f21-9f81-b892370984ba)
   ![Screenshot 2024-04-17 100108](https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/30860a3a-9ac6-496f-985b-35d00a5af4a0)

## ACCURACY SCORE:

   ![Screenshot 2024-04-17 100119](https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/b0abab31-e351-45fa-83a9-a226833a7eec)

## DECISION TREE CLASSIFIER MODEL:

  ![Screenshot 2024-04-17 100128](https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161271714/9d1d9a95-def0-432a-86ad-7f89f49450c2)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
