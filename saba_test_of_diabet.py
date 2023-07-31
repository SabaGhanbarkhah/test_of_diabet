# -*- coding: utf-8 -*-
"""saba_test_of_diabet.ipynb
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense

df=pd.read_csv("diabetes.csv")

df.head()

df.describe()

df.shape

sns.countplot(x='Outcome',data=df)
print('Number of Outcome for each 0 and 1 are :\n',df['Outcome'].value_counts())

plt.subplots(figsize=(9,9))
sns.heatmap(df.corr(),annot=True)

x=df.drop("Outcome",axis=1)
y=df.Outcome

x.shape

y.shape

y.head

from sklearn.model_selection import train_test_split
X_train,X_test ,Y_train,Y_test=train_test_split(x,y,test_size=0.2)

from keras.engine.sequential import Sequential
model=Sequential()
model.add(Dense(12,input_dim=8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=200,batch_size=10)

scores=model.evaluate(X_train,Y_train)
print("Training Accuracy:%.2f%%\n"%(scores[1]*100))
scores=model.evaluate(X_test,Y_test)
print("Training Accuracy:%.2f%%\n"%(scores[1]*100))

Y_test_pred=model.predict(X_test)

print(model.summary())

saba=np.array([[3,200,80,50,100,30,0.1,19]])
out=print(model.predict(saba))