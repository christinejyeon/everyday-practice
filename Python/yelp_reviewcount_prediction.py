import pandas
import numpy

yelp_biz_hrs = pandas.read_csv("/Users/Christine/Downloads/yelp_business_hours.csv")
yelp_biz = pandas.read_csv("/Users/Christine/Downloads/yelp_business.csv")
yelp_biz.columns
yelp_biz = yelp_biz.drop(["name", "neighborhood", "address", "city", "postal_code"], axis=1)
og_categories = ["Restaurants", "Shopping", "Nightlife", "Active Life", "Beauty & Spas", "Automotive"
                 "Home Services",
                 "Coffee & Tea", "Food", "Arts & Entertainment", "Health & Medical",
                 "Professional Services", "Pets", "Real Estate", "Hotels & Travel", "Local Services",
                 "Event Planning & Services", "Public Services & Government", "Financial Services",
                 "Education", "Religious Organizations", "Local Flavor", "Mass Media"]

temp = yelp_biz_hrs.copy()
temp = temp.drop(["business_id"], axis=1)
temp["Flag_notaddedyet"] = temp.apply(lambda x: min(x) == max(x), axis=1)
temp = temp.drop(["monday","tuesday","wednesday","thursday","friday","saturday","sunday"], axis=1)

yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),temp], axis=1)

def totalhrs(df,colnum):
    hrs = df.iloc[:,colnum].str.split("-", expand=True)
    hrs.columns = ["start", "end"]
    hrs.loc[hrs["start"].str.contains(":30"), "start"] = hrs.loc[hrs["start"].str.contains(":30"), "start"].str.split(":").str[0].astype(float) + 0.5
    hrs.loc[(hrs["start"].str.contains(":30") == False) & (hrs["start"] != "None"), "start"] = hrs.loc[(hrs["start"].str.contains(":30") == False) & (hrs["start"] != "None"), "start"].str.split(":").str[0].astype(float)
    hrs.loc[(hrs["start"].str.contains(":30") == False) & (hrs["start"] == "None"), "start"] = 0
    hrs.loc[hrs["end"].str.contains(":30") == True, "end"] = hrs.loc[hrs["end"].str.contains(":30") == True, "end"].str.split(":").str[0].astype(float) + 0.5
    hrs.loc[(hrs["end"].str.contains(":30") == False) & (hrs["end"] != "None"), "end"] = hrs.loc[(hrs["end"].str.contains(":30") == False) & (hrs["end"] != "None"), "end"].str.split(":").str[0].astype(float)
    hrs.loc[(hrs["end"].str.contains(":30") == False) & (hrs["end"] == "None"), "end"] = 0
    hrs.loc[hrs["start"] != "None", "total"] = hrs["end"] - hrs["start"]
    hrs.loc[hrs["total"] <= 0, "total"] = hrs["total"] + 24
    return hrs["total"]

yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,1)], axis=1)
yelp_biz_hrs.rename(columns={"total":"monday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,2)], axis=1)
yelp_biz_hrs.rename(columns={"total":"tuesday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,3)], axis=1)
yelp_biz_hrs.rename(columns={"total":"wednesday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,4)], axis=1)
yelp_biz_hrs.rename(columns={"total":"thursday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,5)], axis=1)
yelp_biz_hrs.rename(columns={"total":"friday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,6)], axis=1)
yelp_biz_hrs.rename(columns={"total":"saturday_total"}, inplace=True)
yelp_biz_hrs = pandas.concat([yelp_biz_hrs.reset_index(drop=True),totalhrs(yelp_biz_hrs,7)], axis=1)
yelp_biz_hrs.rename(columns={"total":"sunday_total"}, inplace=True)

yelp_biz_hrs = yelp_biz_hrs.drop(["monday","tuesday","wednesday","thursday","friday","saturday","sunday"], axis=1)
yelp_biz_hrs["week_total"] = yelp_biz_hrs.fillna(0)["monday_total"]+yelp_biz_hrs.fillna(0)["tuesday_total"]+yelp_biz_hrs.fillna(0)["wednesday_total"]+yelp_biz_hrs.fillna(0)["thursday_total"]+yelp_biz_hrs.fillna(0)["friday_total"]+yelp_biz_hrs.fillna(0)["saturday_total"]+yelp_biz_hrs.fillna(0)["sunday_total"]


yelp_biz.shape
yelp_biz_hrs.shape

og_data = pandas.merge(yelp_biz,yelp_biz_hrs)
og_data = og_data.drop(["business_id"], axis=1)
#og_data.to_csv("og_data.csv")

#### Preparing data for regression (Restaurants)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
og_restaurants = og_data[og_data["categories"].str.contains("Restaurants")]
og_restaurants = og_restaurants.fillna(0)
og_restaurants = og_restaurants.drop(["categories"], axis=1)
og_restaurants = og_restaurants.drop(["state"],axis=1)

#### Feature selection for regression tasks
X = og_restaurants.drop(["review_count"], axis=1)
y = og_restaurants["review_count"]
lasso_scaler = StandardScaler()
X_std = lasso_scaler.fit_transform(X)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01, positive=True) # alpha here is the penalty term ?????
model.fit(X_std,y)
temp = pandas.DataFrame(list(zip(X.columns,model.coef_)), columns=['predictor','coefficient'])
temp = temp.sort_values(by='coefficient',ascending=False)

################### Linear Regression
from sklearn.linear_model import LinearRegression
X = og_restaurants[["stars","sunday_total","is_open","saturday_total"]]
y = og_restaurants["review_count"]
lir_scaler = StandardScaler()
X_std = lir_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

lir = LinearRegression()
lirmodel = lir.fit(X_train, y_train)
y_test_pred = lirmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
#20014.579379


################### KNN regressor
from sklearn.neighbors import KNeighborsRegressor
X = og_restaurants[["stars","sunday_total","is_open","saturday_total"]]
y = og_restaurants["review_count"]
knnr_scaler = StandardScaler()
X_std = knnr_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

for i in range(49,71):
    knnr = KNeighborsRegressor(n_neighbors=i)
    knnrmodel = knnr.fit(X_train, y_train)
    y_test_pred = knnrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)
# The best number for n_neighbors is..
# When n_neighbors = 23 -> mse = 20119.1049
# But it decreases as n_neighbors increases.
# So far when n_neighbors = 68 -> mse = 19495.2602

knnr = KNeighborsRegressor(n_neighbors=23)
knnrmodel = knnr.fit(X_train, y_train)
y_test_pred = knnrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# Average mse: 20119.1049


################### Random forest regressor
from sklearn.ensemble import RandomForestRegressor
X = og_restaurants[["stars","sunday_total","is_open","saturday_total"]]
y = og_restaurants["review_count"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

for i in range(2,5):
    rfr = RandomForestRegressor(max_features=i, random_state=5)
    rfrmodel = rfr.fit(X_train, y_train)
    y_test_pred = rfrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)
# Best max_features = 3

for i in range(2,20):
    rfr = RandomForestRegressor(max_features=3, max_depth=i, random_state=5)
    rfrmodel = rfr.fit(X_train, y_train)
    y_test_pred = rfrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)
# Best max_depth = 6

for i in range(2,10):
    rfr = RandomForestRegressor(max_features=2, max_depth=6, min_samples_split=i, random_state=5)
    rfrmodel = rfr.fit(X_train, y_train)
    y_test_pred = rfrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)
# Best min_samples_split = 2

temp = []
for i in range(10):
    rfr = RandomForestRegressor(max_features=2, max_depth=6, min_samples_split=2)
    rfrmodel = rfr.fit(X_train, y_train)
    y_test_pred = rfrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    temp.append(mse)
sum(temp)/len(temp)

rfr = RandomForestRegressor(max_features=2, max_depth=6, min_samples_split=2)
rfrmodel = rfr.fit(X_train, y_train)
y_test_pred = rfrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# Average mse: 19448.1278 (iteration:10)


################### SVM regressor
from sklearn.svm import SVR
X = og_restaurants[["stars","sunday_total","is_open","saturday_total"]]
y = og_restaurants["review_count"]
svmr_scaler = StandardScaler()
X_std = svmr_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=5)

for i in range(1,10):
    svr = SVR(kernel="linear", epsilon=0.1, C=i)
    svrmodel = svr.fit(X_train, y_train)
    y_test_pred = svrmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(mse)
# Best C = 5

svr = SVR(kernel="linear", epsilon=0.1, C=5)
svrmodel = svr.fit(X_train, y_train)
y_test_pred = svrmodel.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
print(mse)
# Average mse: 21927.4464 (iteration:10)


