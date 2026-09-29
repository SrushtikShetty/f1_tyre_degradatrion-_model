import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from .common import prepare_dataset,build_preprocessor,RANDOM_STATE

def train(data_dir='.',out_dir='artifacts'):
 X,y,g,_,cols=prepare_dataset(data_dir); cv=GroupKFold(n_splits=5)
 pipe=Pipeline([('prep',build_preprocessor(X)),('model',Ridge(alpha=10.0))])
 p=cross_val_predict(pipe,X,y,cv=cv,groups=g,n_jobs=-1); m={'model':'Ridge','r2':r2_score(y,p),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**0.5}; pipe.fit(X,y)
 joblib.dump({'model':pipe,'feature_columns':cols,'target':'future_tire_wear_pct','metrics':m},f'{out_dir}/ridge.joblib'); return m
if __name__=='__main__': print(train())
