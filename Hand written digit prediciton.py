#Hand written Digit Prediciton_ classification analysis

#Initiating project by importing the libraries which would be used in this project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Now importing data
from sklearn.datasets import load_digits

df = load_digits()

_,axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))
for ax,image,label in zip(axes,df.images, df.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation = "nearest")
    ax.set_title("Training : %i" % label)


print(df.images.shape)
print(df.images[0])
print(df.images[0].shape)
print(len(df.images))

n_samples = len(df.images)
data = df.images.reshape((n_samples, -1))

print(data[0])

print(data[0].shape)

print(data.shape)

#Scaling image Data

print(f"Scaling image min data : {data.min()}")

print(f"Scaling image max data : {data.max()}")

data = data/16

print(f"Minimum data : {data.min()}")

print(f"Max data : {data.max()}")

print(f"Data set [0] {data[0]}")

#Train test split data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data. df.target,test_size = 0.3)
print(X_train.shape, X_test.shape , y_train.shape , y_test.shape)

#Random forest model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
print(rf.fit(X_train, y_train))

#Predict test data

y_pred = rf.predict(X_test)
print(y_pred)

#model accuracy

from sklearn.metrics import confusion_matrix,classification_report
print(f"Confusion martrix : {confusion_matrix(y_test,y_pred)}")
print(f"Classification report :  {classification_report(y_test,y_pred)}")
