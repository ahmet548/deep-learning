import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('Student_Grades.csv')

print(df.head())

y = df['Scores']
x = df.drop(columns=['Scores', 'Grade'])

y = y.values.astype('float32')
x = x.values.astype('float32')

model = Sequential()

model.add(Dense(16))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile('adam', loss='mse')

model.fit(x,y,epochs=40,batch_size=16)

print(model.predict(np.array([[3,1,1,2,3]])))