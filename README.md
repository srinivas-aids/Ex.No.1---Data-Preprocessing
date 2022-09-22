# Ex.No.1---Data-Preprocessing
##AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


##ALGORITHM:
```
~~~
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train
~~~
```

##PROGRAM:
```
~~~
import pandas as pd
import numpy as np
df = pd.read_csv("/content/Churn_Modelling.csv")
df.info()
df.isnull().sum()
df.duplicated()
df.describe()
df['Exited'].describe()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df.copy()
df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))
df1
df1.describe()
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)
y = df1.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))
X_train.shape
~~~
```

##OUTPUT:

##Dataset:

![1](https://user-images.githubusercontent.com/93427183/191670956-f235f7b1-3a74-4884-b3c5-85950e8296b4.png)


##Checking for Null Values:


![2](https://user-images.githubusercontent.com/93427183/191670966-387b0dc2-874a-487a-8b42-e20ba029e3fd.png)


##Checking for duplicate values:

![3](https://user-images.githubusercontent.com/93427183/191670979-775abc9b-7817-4379-acc4-aa71447c97b1.png)


##Describing Data:


![4](https://user-images.githubusercontent.com/93427183/191670988-716d8115-f5ff-41ae-989c-11ca2df2634f.png)

![5](https://user-images.githubusercontent.com/93427183/191671017-b13907a3-f3ff-40c9-82ba-3d6ecce49f2e.png)


##X - Values:

![6](https://user-images.githubusercontent.com/93427183/191671067-2ade047d-fc54-4b45-951d-485f1a988d24.png)


##Y - Value:

![7](https://user-images.githubusercontent.com/93427183/191671075-da1c6a9d-2782-4fe6-a9cb-061a25a3ed90.png)

##X_train values and X_train Size:

![8](https://user-images.githubusercontent.com/93427183/191671084-ea0b50f5-2f99-44f1-b0de-f6e43a2ddc19.png)

##X_test values and X_test Size:

![9](https://user-images.githubusercontent.com/93427183/191671107-01916613-1bef-4d41-9125-3509531ebb91.png)


##X_train shape:

![10](https://user-images.githubusercontent.com/93427183/191671113-667f40f1-d3bf-4f11-a393-c0111dd7518c.png)


##RESULT

Data preprocessing is performed in a data set downloaded from Kaggle.
