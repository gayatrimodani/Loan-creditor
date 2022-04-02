# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sn  
import seaborn as sns                   # For plotting graphs
#%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")



#dataset = pd.read_csv('hiring.csv')
# loading the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = train['subscribed']

train = train.drop('subscribed',1)
#print(train.dtypes)
train = pd.get_dummies(train)
test=pd.get_dummies(test)
#print(train.head())
#train=train[["age","balance","job","marital","education","default"]]
#print(train.loc[:, ['age', 'balance', 'job']])
from sklearn.model_selection import train_test_split
# splitting into train and validation with 20% data in validation set and 80% data in train set.
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=4, random_state=0)
clf.fit(X_train,y_train)
# making prediction on the validation set
#Converting words to integer values

#def convert_to_int(word):
 #   word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
  #              'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
   # return word_dict[word]

#X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

#y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()

#Fitting model with trainig data
#regressor.fit(X, y)

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
a=model.predict(test)
