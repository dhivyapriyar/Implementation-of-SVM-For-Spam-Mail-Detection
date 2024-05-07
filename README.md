# Ex 09 Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:

1. Hardware – PCs
2.Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Dhivyapriya. R
RegisterNumber: 212222230032
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

### Encoding:

![image](https://github.com/dhivyapriyar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477552/aaaa0519-8c2c-49bd-9a7c-8719e06a7d6a)


### Head():

![image](https://github.com/dhivyapriyar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477552/94579423-dc6b-4e5e-84da-603132ab32ec)

### Info():

![image](https://github.com/dhivyapriyar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477552/de57f6d2-f975-4bdf-b739-449706356f78)

### isnull().sum():

![image](https://github.com/dhivyapriyar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477552/81c7675d-c0e1-42a5-9206-f933d47d3779)

### Prediction of y:

![image](https://github.com/dhivyapriyar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477552/b0a7436d-9bf4-4300-a860-8e75beefac79)

### Accuracy:

![image](https://github.com/dhivyapriyar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477552/f19c3f8e-41da-47e4-b9d5-e72939526491)

## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
