
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/creditcard.csv")

df.head()

df.isnull().sum()

df = df.fillna(df.mean())

df.info()

sns.countplot(x = df['Class'])

Scatter_plot1 = plt.scatter(x= df['Time'], y=df['Class'])
plt.title('Scatter Distribution of Class vs Time', weight='bold',fontsize = 16)
plt.xlabel('Time',weight = 'bold',fontsize = 13)
plt.ylabel('Class', weight='bold', fontsize= 13)

Scatter_plot2 = plt.scatter(x= df['Amount'], y=df['Class'])
plt.title('Scatter Distribution of Class vs Amount', weight='bold',fontsize = 16)
plt.xlabel('Amount',weight = 'bold',fontsize = 13)
plt.ylabel('Class', weight=  'bold',fontsize = 13)

plt.hist(df['Amount'], bins= range(1,10))
plt.ylabel('Amount',weight = 'bold',fontsize = 13)

df['Amount'].mode()

column1 = df['Amount']
max_value = column1.max()
max_value

column2 = df['Amount']
min_value = column2.min()
min_value

Dataframe_with_Normal_Transactions= df.loc[df['Class']== 0]
Dataframe_with_Normal_Transactions

Dataframe_with_Fraudulent_Transactions = df.loc[df['Class']== 1]
Dataframe_with_Fraudulent_Transactions

Histogram_for_Normal_transactions = plt.hist(Dataframe_with_Normal_Transactions['Amount'], bins = range(1,10))
Histogram_for_Normal_transactions 
plt.title('Frequency of Amount', weight='bold',fontsize = 16)
plt.ylabel('Amount',weight = 'bold',fontsize = 13)

"""Histogram plot for Amount in Fraudulent Transactions"""

Histogram_for_Fraudlent_transactions = plt.hist(Dataframe_with_Fraudulent_Transactions['Amount'], bins = range(1,10))
plt.title('Frequency of Amount', weight='bold',fontsize = 16)
plt.ylabel('Amount',weight = 'bold',fontsize = 13)

Dataframe_with_Normal_Transactions['Amount'].mode()

column1 = Dataframe_with_Normal_Transactions['Amount']
max_value = column1.max()
max_value

column2 = Dataframe_with_Normal_Transactions['Amount']
min_value = column2.min()
min_value

Dataframe_with_Fraudulent_Transactions ['Amount'].mode()

column1 = Dataframe_with_Fraudulent_Transactions ['Amount']
max_value = column1.max()
max_value

column2 = Dataframe_with_Fraudulent_Transactions ['Amount']
min_value = column2.min()
min_value

plt.figure()
fig, ax = plt.subplots(7,4,figsize=(16,28))

for i in range(1,29):
  plt.subplot(7,4,i)
  sns.kdeplot(df[f'V{i}'])
plt.show()

plt.figure()
fig, ax = plt.subplots(7,4,figsize=(16,28))

for i in range(1,29):
  plt.subplot(7,4,i)
  sns.kdeplot(Dataframe_with_Normal_Transactions[f'V{i}'])
plt.show()

plt.figure()
fig, ax = plt.subplots(7,4,figsize=(16,28))

for i in range(1,29):
  plt.subplot(7,4,i)
  sns.kdeplot(Dataframe_with_Fraudulent_Transactions[f'V{i}'])
plt.show()

y = df['Class']
X = df.drop(['Class'], axis =1)

X = df.drop(['Time'], axis =1 )
 df = df.drop(['Time'], axis =1 )

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(X[['Amount']]))
StandardScaler()

print(scaler.mean_)

print(scaler.transform(X[['Amount']]))
df['Amount'] = scaler.transform(X[['Amount']])

df.skew()

X1 = X.columns
X1

for column in df.columns:
  print(column)

for column in df.columns:
  a = df[column].skew()
  if a<-1 or a>1:
    df[column] = np.cbrt(df[column])
    print('new_skewness'+ str(column), df[column].skew())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size =0.8, test_size = 0.2, random_state =42)

# Implementing LogisticRegression Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""Decision Tree Classifier Model"""

# Implementing DecisionTreeClassifier Model
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""Random Forest Classifier Model"""

# Implementing RandomForestClassifier Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""Support Vector Machine Model"""

# Implementing SupportVectorMachine Model
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""XgBoost Model"""

# Implementing XgBoost Model
# We can use GradientBoostingClassifier or XGBClassifier
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train,y_train)

y_pred = xgb_classifier.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

pd.Series(y_resampled).value_counts().plot(kind='bar', title='Class distribution after applying SMOTE')

"""Logistic Regression Model"""

# Implementing LogisticRegression Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""Decision Tree Classifier Model

"""

# Implementing DecisionTreeClassifier Model
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""Random Forest Classifier Model"""

# Implementing RandomForestClassifier Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_resampled,y_resampled)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""Support Vector Machine Model"""

# Implementing SupportVectorMachine Model
from sklearn.svm import SVC
model = SVC()
model.fit(X_resampled,y_resampled)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""XgBoost Model"""

# Implementing XgBoost Model
# We can use GradientBoostingClassifier or XGBClassifier
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_resampled,y_resampled)

y_pred = xgb_classifier.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""**Deep learning model**"""

scaler = StandardScaler()
df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

df = df.drop(['Amount','Time'], axis = 1)
y = df['Class']
X = df.drop(['Class'], axis = 1)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Dense, Activation

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

"""We will write 30 in input dimension so that it can match , if we write 29 it can create an error."""

model = Sequential([
Dense(input_dim = 30, units = 16, activation = 'relu'),
Dense(units = 24, activation = 'relu'),
Dropout(0.5),
Dense(units = 20, activation = 'relu'),
Dense(units = 24, activation = 'relu'),
Dense(units =1, activation = 'sigmoid'),])

"""Training model with X_train, y_train"""

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 15, epochs = 5)

score = model.evaluate(X_test, y_test)
print(score)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))

"""Training model with X_resampled, y_resampled"""

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_resampled, y_resampled, batch_size = 15, epochs = 5)

score = model.evaluate(X_test, y_test)
print(score)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred ))
