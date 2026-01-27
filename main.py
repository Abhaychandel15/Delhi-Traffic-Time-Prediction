import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

MODEL_FILE= "model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(training_num,training_cat):
    # for numerical columns
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

    return full_pipeline

if not os.path.exists(MODEL_FILE):  # so if path is not exist so i have to train my model
    # Load the Dataset
    df=pd.read_csv("real_features.csv")
    df['distance_bins']=pd.cut(df['distance_km'],bins=[0.5,5.5,10.5,15.5,20.5,np.inf],labels=[1,2,3,4,5])

    # stratified shuffle split the data
    split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index,test_index in split.split(df,df['distance_bins']):
        df.loc[test_index].drop(["distance_bins","Trip_ID"],axis=1).to_csv("input.csv",index=False)
        training=df.loc[train_index].drop("distance_bins",axis=1)
    
    # separate features and labels
    training_features = training.drop(['Trip_ID','travel_time_minutes'],axis=1)
    training_labels = training['travel_time_minutes'].copy()
    
    # numattribute and cat attribute
    training_cat= training_features.drop(['distance_km','average_speed_kmph'],axis=1).columns.tolist()
    training_num=['distance_km','average_speed_kmph']

    # construct the pipeline
    pipeline=build_pipeline(training_num,training_cat)

    # transform the data
    training_prep= pipeline.fit_transform(training_features)

    # train the model
    model= RandomForestRegressor()
    model.fit(training_prep,training_labels)

    #now dump the data into pkl files
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Your Model is Trained!")

else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)

    input_data= pd.read_csv("input.csv")
    transform_data= pipeline.transform(input_data)
    predictions= model.predict(transform_data)
    input_data["travel_time_minutes"]=predictions

    input_data.to_csv("output.csv",index=False)
    print("inference is complete output saved to output.csv enjoy!!")
