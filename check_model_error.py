import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Load the Dataset
df_features= pd.read_csv("delhi_traffic_features.csv")
df_targets= pd.read_csv("delhi_traffic_target.csv")

df=pd.merge(df_features,df_targets, on="Trip_ID" )
df.to_csv("real_features.csv",index=False)
df=pd.read_csv("real_features.csv")
# Create a stratified Test set
df['distance_bins']=pd.cut(df['distance_km'],bins=[0.5,5.5,10.5,15.5,20.5,np.inf],labels=[1,2,3,4,5])

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in split.split(df,df['distance_bins']):
    strat_train_set=df.loc[train_index].drop("distance_bins",axis=1)
    strat_test_set=df.loc[test_index].drop("distance_bins",axis=1)

# Now we will work on this data
training= strat_train_set.copy()

# separate features and labels
training_features = training.drop(['Trip_ID','travel_time_minutes'],axis=1)
training_labels = training['travel_time_minutes'].copy()

# numattribute and cat attribute
training_cat= training_features.drop(['distance_km','average_speed_kmph'],axis=1).columns.tolist()
training_num=['distance_km','average_speed_kmph']

#lets make the pipe line 
#for numerical column
num_pipeline= Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
# for categorial columns
cat_pipeline= Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])
# construct Full pipeline

full_pipeline= ColumnTransformer([
    ("num",num_pipeline,training_num),
    ("cat",cat_pipeline,training_cat)
])

training_prep= full_pipeline.fit_transform(training_features)


# Now we train the model which is best for the data

# Linear regression Model
linear_reg= LinearRegression()
linear_reg.fit(training_prep,training_labels)
linear_preds= linear_reg.predict(training_prep)
linear_rmses= -cross_val_score(linear_reg,training_prep,training_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(linear_rmses).describe())

# DecisionTree regressor Model
des_reg= DecisionTreeRegressor()
des_reg.fit(training_prep,training_labels)
des_preds= des_reg.predict(training_prep)
des_rmses= -cross_val_score(des_reg,training_prep,training_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(des_rmses).describe())

# randomfoest regression Model
random_reg= RandomForestRegressor()
random_reg.fit(training_prep,training_labels)
random_preds= random_reg.predict(training_prep)
random_rmses= -cross_val_score(random_reg,training_prep,training_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(random_rmses).describe())
