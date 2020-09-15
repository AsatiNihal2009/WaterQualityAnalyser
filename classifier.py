import numpy as np
import pandas as pd
import sklearn
dataset=pd.read_excel('Table_3.xls')

#Handling Missin Values
u=dataset.columns
dataset[u[3]]=dataset[u[3]].fillna(value=dataset[u[3]].mean())
dataset[u[4]]=dataset[u[4]].fillna(value=dataset[u[4]].mean())
dataset[u[5]]=dataset[u[5]].fillna(value=dataset[u[5]].mean())
dataset[u[6]]=dataset[u[6]].fillna(value=dataset[u[6]].mean())


###Constructing Training Data
X_train=dataset.iloc[:,[2,3,4,5,6]].values
#X=X.fillna(value=0)
#X_train=np.array(X).astype(int)
Y_train=dataset.iloc[:,[7]].values


#Checking Missing Values And Handling them
#u=X.columns
#missing=np.isnan(X[u[0]])
#from sklearn.preprocessing import Imputer
#imputer=Imputer(missing_values='nan',strategy='mean',axis=)
#X_train[:,[1,2,3,4]]=imputer.fit(X_train[:,[1,2,3,4]])


##Categorical Variables Optimization

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
Y_train=labelencoder.fit_transform(Y_train)
Y_train=Y_train.reshape(len(Y_train),1)
#onehotencoder=OneHotEncoder()
#Y_train=onehotencoder.fit_transform(Y_train)
#Y_train=Y_train.toarray()



#Splitting Training data into 80 % and rest 20% is test data
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X_train,Y_train,test_size=0.1,random_state=0) 

#Selecting Support Vector Machine Algorithm for training the model
from sklearn.svm import SVC
classifier= SVC(kernel='linear',random_state=0)
classifier.fit(xtrain,ytrain)

y_pred=classifier.predict(xtest)


#Calculating the accuracy
from sklearn.metrics import accuracy_score

print("Accuracy is :",accuracy_score(ytest,y_pred))


#####Plotting

import matplotlib.pyplot as plt
plt.plot(xtest,ytest)

