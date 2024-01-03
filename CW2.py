"""
Jason Soetandar / 13423803

Coursework 2 Project

1. Importing Library
2. Oversampling Method
3. Splitting Data
4. Model Building
5. Standardization
6. Model Fine-Tuning
7. Ensemble Method
8. Finalise Model
9. Saving the Model
"""
#%% Importing the Library
from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot
import seaborn as sn
import numpy
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#%% Loading the data

filename = "WeatherData.csv"
names = ['date', 'precipitation', 'temp_max', 'temp_min', 'wind', 'weather']
datas = read_csv(filename, names=names)
peek = datas.head(20)
print(peek)

# Analyse the data / EDA (Explanatory Data Analysis)
#%% Dimensions of data
data = datas.drop(['date'], axis = 1)
shape = data.shape
print(shape)
#%% Attributes Data Types

types = data.dtypes
print(types)
#%% Turning Weather from String into Int

data.replace(to_replace ="snow", value = 1, inplace = True)
data.replace(to_replace ="rain", value = 2, inplace = True)
data.replace(to_replace ="drizzle", value = 3, inplace = True)
data.replace(to_replace ="fog", value = 4, inplace = True)
data.replace(to_replace ="sun", value = 5, inplace = True)
print(data)
types = data.dtypes
print(types)
#%% Descriptive statistics

set_option('display.width', 100)
set_option('display.precision', 3)
description = data.describe()
print(description)
#%% Class distribution for Weather 

class_counts = data.groupby('weather').size()
print(class_counts)

#%% Correlations between attributes

correlations = data.corr(method = 'pearson')
print(correlations)
#%% Skewness

skew = data.skew()
print(skew)

#3 Visualizing the Data
#%% Histograms

data.hist(figsize = (10,10))
pyplot.show()
#%% Density plots

data.plot(kind = 'density',subplots = True, layout = (5,5), sharex = False, figsize = (10,10))
pyplot.show()

#%% Correlation matrix

sn.heatmap(correlations, annot=True)
plt.show

#%% Scatter plot matrix

scatter_matrix(data, figsize = (5,5))
pyplot.show()

# 4 Preprocessing the data
#%% Standardize

array = data.values
X = array[:,1:4]
Y = array[:,4]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

set_printoptions(precision=3)
print(rescaledX[0:5,:])

#5 Feature Selection
#%% Univariate Selection

test = SelectKBest(score_func = f_classif, k = 3)
fit = test.fit(X,Y)
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
print(features[0:4,:])
#%% Recursive Feature Elimination  

Y=Y.astype('int')
model = LogisticRegression(solver='liblinear', random_state=11)
rfe = RFE(model, 3)
fits = rfe.fit(X,Y)
print("Num features %d" %fits.n_features_)
print("Selected Features %s" % fits.support_)
print("Feature Ranking %s" % fits.ranking_)
# %% Feature Importance

model = ExtraTreesClassifier(n_estimators=100, random_state=11)
model.fit(X,Y)
print(model.feature_importances_)

#%% Defining the Input and Output
selected_features = ['precipitation', 'temp_max', 'temp_min', 'wind']
x = data[selected_features]
y = data['weather']
# %% Oversampling
from collections import Counter
from imblearn.over_sampling import SMOTE
oversample = SMOTE(random_state=50)
x, y = oversample.fit_resample(x, y)
print(Counter(y))
# %% Splitting Data for train and validation
X_train, X_val, y_train, y_val =  train_test_split(x, y, test_size=0.2, random_state=7)
# %% Model Building
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier(random_state = 7)))
models.append(('NB',GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 7, shuffle=True)
    cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" %(name,cv_results.mean(),cv_results.std())
    print(msg)

#%% boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
#%% Standardize data

pipelines = []
pipelines.append(('ScaledLR',Pipeline([('Scaler',StandardScaler()),('LR',LogisticRegression(solver='liblinear'))])))
pipelines.append(('ScaledLDA',Pipeline([('Scaler',StandardScaler()),('LDA',LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN',Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsClassifier())])))
pipelines.append(('ScaledCART',Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeClassifier(random_state = 7))])))
pipelines.append(('ScaledNB',Pipeline([('Scaler',StandardScaler()),('NB',GaussianNB())])))
pipelines.append(('ScaledSVM',Pipeline([('Scaler',StandardScaler()),('SVM',SVC())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits = 10, random_state = 7, shuffle=True)
    cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" %(name,cv_results.mean(),cv_results.std())
    print(msg)
#%% boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# %% Searching for CART hyperparameters
cart = DecisionTreeClassifier()
print(cart.get_params())

# %% Fine Tuning CART
from sklearn.model_selection import GridSearchCV
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
max_depth = [3, 5, 7, 10, 15]
min_samples_leaf = [1, 2, 3, 4, 5]
max_features = [1, 2, 3, 4]
seed = [21]
param_grid = dict(max_depth=max_depth, max_features=max_features, min_samples_leaf = min_samples_leaf, random_state = seed)
model = DecisionTreeClassifier()
kfold =KFold(n_splits=10,random_state=7,shuffle=True) 
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(rescaledX, y_train)
print("Best: %f using %s"%(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev,param in zip(means,stds,params):
    print("%f (%f) with :%r"%(mean,stdev,param))
#%% Ensemble methods

ensembles = []
ensembles.append(('AB',AdaBoostClassifier()))
ensembles.append(('GBM',GradientBoostingClassifier()))
ensembles.append(('RF',RandomForestClassifier(random_state=7)))
ensembles.append(('ET',ExtraTreesClassifier(random_state=7)))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits = 10,random_state=7,shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)

# %% Compare algorithms
fig = pyplot.figure(figsize = (10,10))
fig.suptitle('Ensemble algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# %% Searching for ExtraTreesClassifier hyperparameters
cart = ExtraTreesClassifier()
print(cart.get_params())
# %% Fine Tuning ET
from sklearn.model_selection import GridSearchCV
max_depth = [3, 5, 7, 10, 15, 20]
min_samples_leaf = [1, 2, 3, 4, 5]
max_features = [1, 2, 3, 4]
seed = [21]
param_grid = dict(max_depth=max_depth, max_features=max_features, min_samples_leaf = min_samples_leaf, random_state = seed)
model = ExtraTreesClassifier()
kfold =KFold(n_splits=10,random_state=7,shuffle=True) 
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s"%(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev,param in zip(means,stds,params):
    print("%f (%f) with :%r"%(mean,stdev,param))
#%%
model = ExtraTreesClassifier(random_state = 21)
model.fit(X_train,y_train)
predictions = model.predict(X_val)
print(accuracy_score(y_val,predictions))
print(confusion_matrix(y_val,predictions))
print(classification_report(y_val,predictions))
#%% Saving the Model with Joblib

from joblib import dump
from joblib import load

filename = 'joblib_finalised.sav'
dump(model,filename)

loaded_model = load(filename)
result = loaded_model.score(X_val,y_val)
print(result)
