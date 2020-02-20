import  pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

#reading the iris csv file
iris = pd.read_csv('Iris.csv')

print(iris.head(10))

#checking for NaN values in the dataset
iris .info()
iris.isnull().sum()

#now we do not need the id coloumn as it is a reddudndancy
iris.drop('Id' , axis=1, inplace = True)

#checking the new datatset
print(iris.head(10))

#now we draw a few histograms between lenghth and width

iris.hist(edgecolor='red', linewidth=1)
fig=plt.gcf()
fig.set_size_inches(10,7)
plt.show()

#used a pre made module so as to see density of lenghth and width vary according to fat
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
plt.show()

#plotting an appropriate heatmap to see how paramerts interact with target data correlatiom matrix
plt.figure(figsize=(10,8))
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')
plt.show()

#assigning the inputs before training the model
X= iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y=iris['Species']

#checking wether the input passed was correctly initialsed
print (X.head(10))
print(Y.head(10))

#Splitting the data into test train sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#importing the required models to perform analysis
from sklearn.neighbors import KNeighborsClassifier

#perfomring action on KNN model
#using all parametrs to predict Species
model = KNeighborsClassifier(n_neighbors = 3)

model .fit(X_train,Y_train)
Y_pred = model.predict(X_test)

#accurrcy of the model is to measured using metrics analysis as trget is text based data
from sklearn import metrics
print('The accuracy of the KNN is',metrics.accuracy_score(Y_pred,Y_test))