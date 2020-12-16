from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np

# reading and dropping time column
df = pd.read_csv('creditcard.csv')
df = df.drop("Time", axis=1)
df = df.dropna()

# dividing dataset on fraud and not fraud transactions
df = df.sort_values('Class', ascending=False)
df_pos = df[:492]
df_neg = df[492:100000]

l = []
for i in range(5):
    l.append(df_pos[:394])
l.append(df_neg[:60000])

# separating dataset on test and train data
train_data = pd.concat(l)
test_data = pd.concat([df_pos[394:], df_neg[80000:]])

train_data = np.array(train_data)
test_data = np.array(test_data)

def train(data, pred_data, epochs=20, verbose=0, val=0):
    
    dataX = data[:,:-1]
    dataY = data[:,-1:]
    
    predX = pred_data[:,:-1]
    predY = pred_data[:,-1:]
    
    model = Sequential()
    model.add(Dense(30, input_dim=dataX.shape[1], activation='linear'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(5, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
    model.fit(dataX, dataY, 
              validation_split=val,
              epochs=epochs,
              batch_size=32,
              verbose=verbose)
    
    pp = model.predict(predX)
    
    return pp
    
# training the model and obtaining the predictions
pp = train(train_data, test_data, epochs=40, verbose=0)

# printing out the result
result = pp > .0001

tp_result = result.astype(int)[:98].sum() * 100 / 98
print(str(round(tp_result, 2)) + '% of detected frauds')

tp_result_fp = result.astype(int)[98:].sum() * 100 / len(test_data[98:])
print(str(round(tp_result_fp, 2)) + '% of false positives, i.e ' +
      str(result.astype(int)[98:].sum()) + ' transactions wrongly detected as frauds')
