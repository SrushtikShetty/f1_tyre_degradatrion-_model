import joblib,os,xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold,cross_val_predict
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from .common import prepare_dataset,build_preprocessor,RANDOM_STATE

def train(data_dir='.',out_dir='artifacts'):
 os.makedirs(out_dir,exist_ok=True); X,y,g,_,cols=prepare_dataset(data_dir); cv=GroupKFold(n_splits=5)
 device='cuda'
 try: xgb.XGBRegressor(n_estimators=2,tree_method='hist',device='cuda').fit([[0],[1]],[0,1])
 except Exception: device='cpu'
 model=xgb.XGBRegressor(n_estimators=500,max_depth=6,learning_rate=.05,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',tree_method='hist',device=device,n_jobs=-1,random_state=RANDOM_STATE)
 pipe=Pipeline([('prep',build_preprocessor(X)),('model',model)]); p=cross_val_predict(pipe,X,y,cv=cv,groups=g,n_jobs=1); m={'model':'XGBoost','r2':r2_score(y,p),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**0.5,'device':device}; pipe.fit(X,y); joblib.dump({'model':pipe,'feature_columns':cols,'target':'future_tire_wear_pct','metrics':m},f'{out_dir}/xgboost.joblib'); return m
if __name__=='__main__': print(train())
