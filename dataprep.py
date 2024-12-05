import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import set_config
from sklearn.model_selection import train_test_split




def load_data():
    df = pd.read_csv('updated_pollution_dataset.csv')
    print(f"Df Shape: {df.shape}")
    return df




def load_prepped(test_ratio,dev_set,target_col,polydegs=1):
    df = load_data()
    if dev_set == False:
        print("Prepping with no Dev Set!")
        train, test = split(df,test_ratio,dev_set)
        Xtrain_prepared, ytrain_prepared, Xtest_prepared, ytest= pipline(train, test,target_col,polydegs)
        print('Train X,y shapes: ',Xtrain_prepared.shape,Xtrain_prepared.shape)
        print('Test X,y shapes: ',Xtest_prepared.shape,Xtest_prepared.shape)
        return Xtrain_prepared, ytrain_prepared, Xtest_prepared, ytest
        
    if dev_set == True:
        print("Prepping with Dev Set!")
        train, test, dev = split(df,test_ratio,dev_set)
        Xtrain_prepared, ytrain_prepared, Xtest_prepared, ytest, Xdev_prepared, ydev = pipline_dev(train, test, dev,target_col,polydegs)
        print('Train X,y shapes: ',Xtrain_prepared.shape,Xtrain_prepared.shape)
        print('Test X,y shapes: ',Xtest_prepared.shape,Xtest_prepared.shape)
        print('Dev X,y shapes: ',Xdev_prepared.shape,Xdev_prepared.shape)
        return Xtrain_prepared, ytrain_prepared, Xtest_prepared, ytest, Xdev_prepared, ydev
        
    else: return print("Dev_Set must be Boolean!")




def split(df,test_ratio,dev_set):
    if dev_set == False:
        train, test = train_test_split(df, test_size=test_ratio)
        return train, test
    else:
        test_ratio *= 2
        train, test = train_test_split(df, test_size=test_ratio)
        dev, test = train_test_split(test, test_size=0.5)
        return train, test,dev
    




def pipline_dev(dftrain, dftest, dfdev,target_col,polydegs):

    set_config(transform_output="pandas")
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('ratio_adder', FunctionTransformer(add_columns)),  
        ('log_transformer', FunctionTransformer(replace_columns)),   
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=polydegs, include_bias=False))
    ])
    
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(sparse_output=False)
    )
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object))
    ])
    

    Xtrain, ytrain = dftrain.drop(columns=[target_col]), dftrain[target_col]
    Xtest, ytest = dftest.drop(columns=[target_col]), dftest[target_col]
    Xdev, ydev = dfdev.drop(columns=[target_col]), dfdev[target_col]
    

    Xtrain_prepared = full_pipeline.fit_transform(Xtrain)
    Xtest_prepared = full_pipeline.transform(Xtest)
    Xdev_prepared = full_pipeline.transform(Xdev)
    

    Xtrain_prepared, ytrain_prepared = handle_outlier(Xtrain_prepared, ytrain)
    
    return Xtrain_prepared, ytrain_prepared, Xtest_prepared, ytest, Xdev_prepared, ydev





def pipline(dftrain, dftest,target_col,polydegs):

    set_config(transform_output="pandas")
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('ratio_adder', FunctionTransformer(add_columns)),  
        ('log_transformer', FunctionTransformer(replace_columns)),   
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=polydegs, include_bias=False))
    ])
    
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(sparse_output=False)
    )
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object))
    ])
    

    Xtrain, ytrain = dftrain.drop(columns=[target_col]), dftrain[target_col]
    Xtest, ytest = dftest.drop(columns=[target_col]), dftest[target_col]


    Xtrain_prepared = full_pipeline.fit_transform(Xtrain)
    Xtest_prepared = full_pipeline.transform(Xtest)

    

    Xtrain_prepared, ytrain_prepared = handle_outlier(Xtrain_prepared, ytrain)
    
    return Xtrain_prepared, ytrain_prepared, Xtest_prepared, ytest






def get_outlier_indices(X):
    model = LocalOutlierFactor()
    return model.fit_predict(X)




def handle_outlier(X, y):
    outlier_ind = get_outlier_indices(X)
    return X[outlier_ind == 1], y[outlier_ind == 1]




def add_columns(df):
    df['numeric_ratio_1/2'] = df['numeric_1'] / df['numeric_1']
    return df




def replace_columns(df):
    df['log_numeric_ratio_1/2'] = np.log10(df['numeric_ratio_1/2'])
    return df
