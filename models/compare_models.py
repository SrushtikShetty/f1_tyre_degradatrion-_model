import json,os,pandas as pd
from .linear_regression import train as train_linear
from .random_forest import train as train_rf
from .xgboost_model import train as train_xgb
from .neural_network import train as train_nn

def run(data_dir='.',out_dir='artifacts'):
 os.makedirs(out_dir,exist_ok=True); results=[train_linear(data_dir,out_dir),train_rf(data_dir,out_dir),train_xgb(data_dir,out_dir),train_nn(data_dir,out_dir)]; df=pd.DataFrame(results).sort_values('r2',ascending=False); df.to_csv(f'{out_dir}/model_comparison.csv',index=False); json.dump(results,open(f'{out_dir}/model_metrics.json','w'),indent=2); print(df.to_string(index=False)); return df
if __name__=='__main__': run()
