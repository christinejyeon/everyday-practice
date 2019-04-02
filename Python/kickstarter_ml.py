import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
import matplotlib.pyplot as plt


ks = pandas.read_excel("/Users/Christine/Documents/INSY 652/Kickstarter.xlsx")
ks = ks.drop(['project_id','name','pledged','currency','deadline','state_changed_at', 'created_at',
              'launched_at','name_len','name_len_clean','blurb_len','blurb_len_clean','state_changed_at_weekday',
             'created_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr',
              'state_changed_at_hr','created_at_month','created_at_day',
              'created_at_yr','created_at_hr','launch_to_state_change_days'], axis=1)
ks.shape #(18568,22)
ks.isna().sum()
ks['category'].value_counts()
ks = ks.dropna(axis=0, how='any') # dropping 9.07% of total data
# final shape: (16884, 22)

ks_dummied = ks.copy()
for elem in ks_dummied['country'].unique():
    ks_dummied['country_'+str(elem)] = ks_dummied['country'] == elem
for elem in ks_dummied['category'].unique():
    ks_dummied['category_'+str(elem)] = ks_dummied['category'] == elem
for elem in ks_dummied['deadline_weekday'].unique():
    ks_dummied['deadline_weekday_'+str(elem)] = ks_dummied['deadline_weekday'] == elem
for elem in ks_dummied['launched_at_weekday'].unique():
    ks_dummied['launched_at_weekday_'+str(elem)] = ks_dummied['launched_at_weekday'] == elem

ks_dummied = ks_dummied.drop(["country","category","deadline_weekday","launched_at_weekday"], axis=1)



###################################### Question 1 ######################################
# Provide summary statistics for the variables that are interesting/relevant to your analyses.
og_description = ks.describe(include='all')
print(og_description)



###################################### Question 2 ######################################
# Develop a regression model (i.e., a supervised-learning model where the target variable is a continuous variable)
# to predict the value of the variable “usd_pledged.” After you obtain the final model,
# explain the model and justify the predictors you include/exclude.

ks_regression = ks_dummied.copy()
for elem in ks_regression['state'].unique():
    ks_regression['state_'+str(elem)] = ks_regression['state'] == elem
ks_regression = ks_regression.drop(["state"], axis=1)

################### Linear Regression


################### KNN regressor
from sklearn.neighbors import KNeighborsRegressor
X = ks_regression.drop(["usd_pledged"], axis=1)
y = ks_regression["usd_pledged"]

knnr_scaler = StandardScaler()
X_std = knnr_scaler.fit_transform(X)

model = Lasso(alpha=0.01, positive=True) # alpha here is the penalty term ?????
model.fit(X_std,y)
temp = pandas.DataFrame(list(zip(X.columns,model.coef_)), columns=['predictor','coefficient'])
# backers_count,//4D//category_Hardware//category_Web,category_Gadgets,staff_pick,category_Wearables,category_Software

X = ks_regression[["backers_count","category_Hardware","category_Web","category_Gadgets","staff_pick","category_Wearables","category_Software"]]
y = ks_regression["usd_pledged"]
X_std = knnr_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

knnr = KNeighborsRegressor(n_neighbors=10)
knnrmodel = knnr.fit(X_train, y_train)
y_test_pred = knnrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# 2582064033.321328 lol wtf


################### Random forest regressor
from sklearn.ensemble import RandomForestRegressor
X = ks_regression.drop(["usd_pledged"],axis=1)
y = ks_regression['usd_pledged']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

rfr = RandomForestRegressor()
rfrmodel = rfr.fit(X_train, y_train)
y_test_pred = rfrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# 3148259465.0089626


################### SVM regressor

from sklearn.svm import SVR
X = ks_regression.drop(["usd_pledged"],axis=1)
y = ks_regression['usd_pledged']
svmr_scaler = StandardScaler()
X_std = svmr_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

svr = SVR(kernel="linear", epsilon=0.1)
svrmodel = svr.fit(X_train, y_train)
y_test_pred = svrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# 9372579465.64207


################### ANN regressor

from sklearn.neural_network import MLPRegressor
X = ks_regression.drop(["usd_pledged"],axis=1)
y = ks_regression['usd_pledged']
mlpr_scaler = StandardScaler()
X_std = mlpr_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

annr = MLPRegressor(hidden_layer_sizes=(9), max_iter=100000)
annrmodel = annr.fit(X_train, y_train)
y_test_pred = svrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)





###################################### Question 3 ######################################
# Develop a classification model (i.e., a supervised-learning model where the target variable is a categorical variable)
# to predict whether the variable “state” will take the value “successful” or “failure.”
# After you obtain the final model, explain the model and justify the predictors you include/exclude.

ks_classification = ks_dummied.loc[(ks_dummied["state"]=="successful")|(ks_dummied["state"]=="failed")]

################### Logistic Regression

from sklearn.linear_model import LogisticRegression
X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]
logr_scaler = StandardScaler()
X_std = logr_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)

logr = LogisticRegression()
logrmodel = logr.fit(X, y)
sfm = SelectFromModel(logrmodel, threshold=1) # only get the variables above the threshold
sfm.fit(X,y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
rfe = RFE(logrmodel, 5)
model = rfe.fit(X,y)
temp = pandas.DataFrame(list(zip(X.columns,model.ranking_)), columns=['predictor','ranking'])
# staff_pick, category_Blues, category_Places, category_Shorts, maybe category_Web,
# thrillers, webseries


X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=5)

logr = LogisticRegression()
logrmodel = logr.fit(X_train, y_train)
logrmodel.intercept_
logrmodel.coef_

y_test_pred = logrmodel.predict(X_test)
accuracy_score(y_test,y_test_pred)

temp = []
for i in range(1,33):
    logrmodel = logr.fit(X_train, y_train)
    y_test_pred = logrmodel.predict(X_test)
    temp.append(accuracy_score(y_test, y_test_pred))
sum(temp)/float(len(temp))

# average accuracy score doesn't really go up to 0.85



################### kNN
from sklearn.neighbors import KNeighborsClassifier
X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]
knn_scaler = StandardScaler()
X_std = knn_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)

for i in range(11,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knnmodel = knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)
    print(accuracy_score(y_test,y_test_pred))
# best n_neighbors is 19 and accuracy score is 73.71



################### Random Forest

from sklearn.ensemble import RandomForestClassifier
X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]

randomforest = RandomForestClassifier()
rfmodel = randomforest.fit(X, y)

sfm = SelectFromModel(rfmodel, threshold=0.04) # only get the variables above the threshold
sfm.fit(X,y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
# when threshold = 0.01
# goal, backers_count, usd_pledged, spotlight, staff_pick (before that, only the first four)
# 0.05, staff_pick, backers_count, create_to_launch_days, 0.04 launched_at_day

rfe = RFE(rfmodel, 4)
model = rfe.fit(X,y)
pandas.DataFrame(list(zip(X.columns,model.ranking_)), columns=['predictor','ranking'])
# goal, backers_count, usd_pledged, spotlight, launch_to_deadline_days
# backers_count, create_to_launch_days, deadline_day (launched_at_day), deadline_hr, launch_to_deadline_days


# real model building
X = ks_classification[["backers_count","create_to_launch_days","staff_pick","launched_at_day","deadline_day","deadline_hr","launch_to_deadline_days"]]
#"backers_count","create_to_launch_days","staff_pick","launched_at_day","deadline_day"
y = ks_classification["state"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=5)

randomforest = RandomForestClassifier(max_features=3,max_depth=9)
rfmodel = randomforest.fit(X_train, y_train)
y_test_pred = rfmodel.predict(X_test)

accuracy_score(y_test, y_test_pred)

temp = []
for i in range(1,33):
    rfmodel = randomforest.fit(X_train, y_train)
    y_test_pred = rfmodel.predict(X_test)
    temp.append(accuracy_score(y_test, y_test_pred))
sum(temp)/float(len(temp))

# at least 0.8525

for i in range(2,20):
    model2 = RandomForestClassifier(random_state=5, max_depth=i)
    scores = cross_val_score(estimator = model2, X=X, y=y, cv=10)
    print(i, ":", numpy.average(scores))



################### SVM

from sklearn.svm import SVC
X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]
svc_scaler = StandardScaler()
X_std = svc_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)


svc = SVC(kernel="rbf", random_state=5, C=15)
svcmodel = svc.fit(X_train,y_train)

y_test_pred = svcmodel.predict(X_test)
accuracy_score(y_test, y_test_pred)

# 80.23

for i in range (1,20):
    svc = SVC(kernel="rbf", random_state=5, C=15, gamma=i)
    svcmodel = svc.fit(X_std, y)
    scores = cross_val_score(estimator=svcmodel, X=X_std, y=y, cv=5)
    print(i,":",numpy.average(scores))
# C at 15, 80.58



################### ANN

from sklearn.neural_network import MLPClassifier
X = ks_classification.drop(["state","spotlight","goal","usd_pledged"], axis=1)
y = ks_classification["state"]
mlp_scaler = StandardScaler()
X_std = mlp_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)

mlp = MLPClassifier(hidden_layer_sizes=(5), max_iter=1000, random_state=5)
annmodel = mlp.fit(X_train, y_train)
y_test_pred = annmodel.predict(X_test)
accuracy_score(y_test, y_test_pred)
#87.50

for i in range(2,21):
    mlp = MLPClassifier(hidden_layer_sizes=(5,i), max_iter=1000, random_state=5)
    scores = cross_val_score(estimator=mlp, X=X_std, y=y, cv=5)
    print(i,":",numpy.average(scores))




###################################### Question 4 ######################################
# Develop a cluster model (i.e., an unsupervised-learning model which can group observations together)
# to group projects together.

ks_clustering = ks_dummied.copy()
for elem in ks_clustering['state'].unique():
    ks_clustering['state_'+str(elem)] = ks_clustering['state'] == elem
ks_clustering = ks_clustering.drop(["state"],axis=1)

X = ks_clustering.copy()
kmeans_scaler = StandardScaler()
X_std = kmeans_scaler.fit_transform(X)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
model = kmeans.fit(X_std)
labels = model.predict(X_std)

withinss = []
for i in range(2,8):
    kmeans = KMeans(n_clusters=i)
    kmeansmodel = kmeans.fit(X_std)
    withinss.append(kmeansmodel.inertia_)

pyplot.plot([2,3,4,5,6,7], withinss)
plt.show()

from sklearn.metrics import silhouette_samples
silhouette_samples(X, labels)

from sklearn.metrics import silhouette_score
silhouette_score(X, labels)

from sklearn.metrics import calinski_harabaz_score
score = calinski_harabaz_score(X, labels)
from scipy.stats import f
df1 = 3 # df1=k-1
df2 = 16880  # df2=n-k
pvalue = 1-f.cdf(score, df1, df2)
pvalue

X_train, X_test, y_train, y_test = train_test_split(X_std, X, test_size=0.5, random_state=5)

kmeans_train = KMeans(n_clusters=3, random_state=0)
model_train = kmeans_train.fit(X_train)
labels_train = model_train.labels_
pyplot.scatter(X_train[:,0], X_train[:,1], c = labels_train, cmap='rainbow')
pyplot.show()

model_train.cluster_centers_

df_train = pandas.DataFrame(X_train, columns=['sepal_length','sepal_width'])
df_train["cmember"] = labels_train


kmeans_test = KMeans(n_clusters=3, random_state=0)
model_test = kmeans_test.fit(X_test)
labels_test = model_test.labels_
pyplot.scatter(X_test[:,0], X_test[:,1], c = labels_test, cmap='rainbow')
pyplot.show()

model_test.cluster_centers_

df_test = pandas.DataFrame(X_test, columns=['sepal_length','sepal_width'])
df_test["cmember"] = labels_test

from scipy.stats import ttest_ind

ttest_ind(df_train.loc[df_train['cmember']==0,['sepal_length']], df_test.loc[df_test['cmember']==1, ['sepal_length']])
ttest_ind(df_train.loc[df_train['cmember']==0,['sepal_width']], df_test.loc[df_test['cmember']==1, ['sepal_width']])
ttest_ind(df_train.loc[df_train['cmember']==1,['sepal_length']], df_test.loc[df_test['cmember']==2, ['sepal_length']])
ttest_ind(df_train.loc[df_train['cmember']==1,['sepal_width']], df_test.loc[df_test['cmember']==2, ['sepal_width']])
ttest_ind(df_train.loc[df_train['cmember']==2,['sepal_length']], df_test.loc[df_test['cmember']==0, ['sepal_length']])
ttest_ind(df_train.loc[df_train['cmember']==2,['sepal_width']], df_test.loc[df_test['cmember']==0, ['sepal_width']])



from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)






profs_data = pandas.read_excel("/Users/Christine/Documents/INSY 652/Kickstarter.xlsx")

###### Regression

###### Classification
annmodel

###### Clustering

