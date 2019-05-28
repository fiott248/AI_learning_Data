# import packages
import numpy as np
import pandas as pd
#defined columns into array
columns = ["ClickXPos", "ClickYPos", "Scroll", "Time2Click", "Agent", "IPAddress", "TZone", "Language", "Fonts", "Cookies", "Session", "Resolution", "ColorDepth", "Bot"]
columnswithoutbot = ["ClickXPos", "ClickYPos", "Scroll", "Time2Click", "Agent", "IPAddress", "TZone", "Language", "Fonts", "Cookies", "Session", "Resolution", "ColorDepth"]
#reading csv files
data2 = pd.read_csv('learningdata.csv', header=0, names=columns)
newdata = pd.read_csv('newdata.csv', header=0, names=columnswithoutbot)
#more modules used in AI algorithm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
#encoder to
le = preprocessing.LabelEncoder()
#transforming data type into int64 for data2 and newdata
for column_name in data2.columns:
    if data2[column_name].dtype == object:
        data2[column_name] = le.fit_transform(data2[column_name])
    else:
        pass
for column_name in newdata.columns:
    if newdata[column_name].dtype == object:
        newdata[column_name] = le.fit_transform(newdata[column_name])
    else:
        pass
# defining X, Y for data algorithm
y = data2['Bot']
X = data2.drop('Bot', axis=1)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
#function to use DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=5, random_state=0)
scores = cross_val_score(clf, X, y)
#print(scores.mean())  used for testing
#function to use ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
#print(scores.mean()) used for testing
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
#print(scores.mean()) used for testing
clf.fit(X,y)
#running predirct function on the new data
lastdata = clf.predict(newdata.iloc[:,])
#printing data
print lastdata
