import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

pd.set_option('display.max_columns', None)

df = pd.read_csv('auto-mpg.csv')
df = df.replace('?', np.nan)
df  = df.dropna()
df = df.drop('car name', axis=1)

X = df.drop('mpg', axis=1).values
y = df['mpg'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42 )

lr  = LinearRegression()
lr.fit(X,y)

# Creating a pickle file for the classifier
filename = 'Auto-mpg-lr-model.pkl'
pickle.dump(lr, open(filename, 'wb'))