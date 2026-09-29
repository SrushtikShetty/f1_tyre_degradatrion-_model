import os,joblib,numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold,cross_val_predict
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .common import prepare_dataset,build_preprocessor,RANDOM_STATE

def train(data_dir='.',out_dir='artifacts'):
 os.makedirs(out_dir,exist_ok=True); X,y,g,_,cols=prepare_dataset(data_dir); cv=GroupKFold(n_splits=5)
 # sklearn MLP is used here so the deep-learning model stays easy to run and serialize alongside the tree models.
 pipe=Pipeline([('prep',build_preprocessor(X)),('scale',StandardScaler(with_mean=False)),('model',MLPRegressor(hidden_layer_sizes=(256,128,64),activation='relu',solver='adam',learning_rate_init=.001,max_iter=80,early_stopping=True,validation_fraction=.1,batch_size=512,random_state=RANDOM_STATE))])
 p=cross_val_predict(pipe,X,y,cv=cv,groups=g,n_jobs=1); m={'model':'NeuralNetworkMLP','r2':r2_score(y,p),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**0.5}; pipe.fit(X,y); joblib.dump({'model':pipe,'feature_columns':cols,'target':'future_tire_wear_pct','metrics':m},f'{out_dir}/neural_network.joblib'); return m
if __name__=='__main__': print(train())
