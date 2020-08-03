# Random Forest Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
'''
def sum(oneee,twooo):
    third=oneee+twooo
    print(third)
    return third

def randomforest:
'''
'''
# Importing the dataset
dataseta = pd.read_csv('Mat_Training.tsv', sep='\t',dtype=object)
dataset=dataseta[dataseta.duplicated(subset=["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR"],keep='last')]
print(dataset.isna().any(axis=0))
print(dataset.isnull().sum())
#dataset.dropna()
print(dataset.info())
print(dataset.describe())

dataset1 = pd.read_csv('Mat_test.tsv', sep='\t')
dataset['istrainset'] = 1
dataset1['MATNR']=None
dataset1['istrainset'] = 0
dataset_p = pd.concat(objs=[dataset, dataset1], axis=0)

# Encoding categorical data
LIFNR=pd.get_dummies(dataset_p['LIFNR'],drop_first=True)
MAT1=pd.get_dummies(dataset_p['MAT1'],drop_first=True)
MAT2=pd.get_dummies(dataset_p['MAT2'],drop_first=True)
MAT3=pd.get_dummies(dataset_p['MAT3'],drop_first=True)
SP_KUNNR=pd.get_dummies(dataset_p['SP_KUNNR'],drop_first=True)

dataset_combined=pd.concat([dataset_p,LIFNR,MAT1,MAT2,MAT3,SP_KUNNR],axis=1)
X_trainn=dataset_combined[dataset_combined['istrainset'] == 1]
X_xls=dataset_combined[dataset_combined['istrainset'] == 0]

y_trainn=dataset_combined[dataset_combined['istrainset'] == 1]['MATNR']
y_xls=dataset_combined[dataset_combined['istrainset'] == 0]['MATNR']


X= X_trainn.drop(["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR","MATNR","istrainset"], axis=1)
XX=X_xls.drop(["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR","MATNR","istrainset"], axis=1)

label_encoder_y = preprocessing.LabelEncoder()
y1= label_encoder_y.fit_transform(y_trainn)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y1, test_size=0.00001,random_state=1)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 75, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Importing the dataset
y_pred1 = classifier.predict(X_test)
y_pred=label_encoder_y.inverse_transform(y_pred1)
y_test_in=label_encoder_y.inverse_transform(y_test)
y_train1 = classifier.predict(X_train)
y_predtrain_in=label_encoder_y.inverse_transform(y_train1)
# Making the Confusion Matrix

cm = confusion_matrix(y_train, y_train1)
print( accuracy_score(y_train, y_train1))
y_predict1 = classifier.predict(XX)
y_predict_xls=label_encoder_y.inverse_transform(y_predict1)
dataset1['MATNR']=y_predict_xls
print(dataset1)

dataset1.to_csv('Predict.csv')
'''
