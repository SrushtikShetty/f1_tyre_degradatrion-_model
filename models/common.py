from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE=42
TARGET='next_lap_wear_increment'
ALLOWED_RAW_FEATURES={'season','race_round','circuit_id','circuit_length_km','circuit_turns','circuit_drs_zones','circuit_overtake_difficulty','circuit_base_lap_time_sec','driver_skill_rating','driver_aggression_rating','driver_consistency_rating','team_id','team_budget_tier','team_car_speed_rating','team_car_downforce_rating','team_engine_supplier','team_car_reliability_rating','team_pit_crew_rating','grid_position','q1_time_sec','q2_time_sec','q3_time_sec','race_total_laps','race_weather_start','lap','position','lap_time_sec','s1_time_sec','s2_time_sec','s3_time_sec','tire_compound','tire_age_laps','fuel_load_kg','ers_deploy_pct','ers_harvest_pct','drs_activated','gap_to_leader_sec','gap_ahead_sec','gap_behind_sec','track_status','weather_current','race_air_temp_c','race_track_temp_c','race_humidity_pct'}
FORBIDDEN_FEATURES={'tire_wear_pct','future_tire_wear_pct','next_lap_wear_increment','next_lap','next_tire_age','next_compound','finish_position','status','points','fastest_lap','race_weather_end','race_safety_car_deployed','race_red_flag','pit_stop_this_lap','pit_stop_duration_sec','pit_new_compound','race_id','driver_id'}
ENGINEERED=['sector_total_sec','lap_time_per_km','tire_age_ratio','aggression_x_tire_age','consistency_x_tire_age','track_temp_x_tire_age','air_temp_x_tire_age','overtake_difficulty_x_tire_age','fuel_ratio','total_nearest_gap','position_change','race_progress','car_performance_index']

def load_data(data_dir='.'):
    p=Path(data_dir); train=pd.read_csv(p/'train.csv'); test=pd.read_csv(p/'test.csv')
    if TARGET in test: test=test.drop(columns=[TARGET])
    return train,test

def clean_data(df):
    df=df.copy()
    for c in df.select_dtypes(include=['bool']).columns: df[c]=df[c].astype(int)
    return df.drop(columns=[c for c in df.columns if df[c].isna().all()])

def create_future_target(train):
    req={'race_id','driver_id','lap','tire_age_laps','tire_compound','tire_wear_pct'}; miss=req-set(train.columns)
    if miss: raise ValueError(f'Missing columns: {sorted(miss)}')
    df=train.copy().sort_values(['race_id','driver_id','lap']).reset_index(drop=True); g=df.groupby(['race_id','driver_id'],sort=False)
    df['_next_lap']=g['lap'].shift(-1); df['_next_age']=g['tire_age_laps'].shift(-1); df['_next_compound']=g['tire_compound'].shift(-1); df['_next_wear']=g['tire_wear_pct'].shift(-1); df['future_tire_wear_pct']=df['_next_wear']; df[TARGET]=df['_next_wear'] - df['tire_wear_pct']
    ok=(df['_next_lap']==df['lap']+1)&(df['_next_age']==df['tire_age_laps']+1)&(df['_next_compound']==df['tire_compound'])&df['_next_wear'].notna()&np.isfinite(df['_next_wear'])
    return df.loc[ok].drop(columns=['_next_lap','_next_age','_next_compound','_next_wear']).copy()

def create_current_features(df):
    df=df.copy()
    if {'s1_time_sec','s2_time_sec','s3_time_sec'}<=set(df): df['sector_total_sec']=df[['s1_time_sec','s2_time_sec','s3_time_sec']].sum(axis=1)
    if {'lap_time_sec','circuit_length_km'}<=set(df): df['lap_time_per_km']=df['lap_time_sec']/df['circuit_length_km'].replace(0,np.nan)
    if {'tire_age_laps','race_total_laps'}<=set(df): df['tire_age_ratio']=df['tire_age_laps']/df['race_total_laps'].replace(0,np.nan)
    for a,b,o in [('driver_aggression_rating','tire_age_laps','aggression_x_tire_age'),('driver_consistency_rating','tire_age_laps','consistency_x_tire_age'),('race_track_temp_c','tire_age_laps','track_temp_x_tire_age'),('race_air_temp_c','tire_age_laps','air_temp_x_tire_age'),('circuit_overtake_difficulty','tire_age_laps','overtake_difficulty_x_tire_age')]:
        if {a,b}<=set(df): df[o]=df[a]*df[b]
    if {'fuel_load_kg','race_total_laps'}<=set(df): df['fuel_ratio']=df['fuel_load_kg']/df['race_total_laps'].replace(0,np.nan)
    if {'gap_ahead_sec','gap_behind_sec'}<=set(df): df['total_nearest_gap']=df['gap_ahead_sec'].abs()+df['gap_behind_sec'].abs()
    if {'grid_position','position'}<=set(df): df['position_change']=df['grid_position']-df['position']
    if {'lap','race_total_laps'}<=set(df): df['race_progress']=df['lap']/df['race_total_laps'].replace(0,np.nan)
    if {'team_car_speed_rating','team_car_downforce_rating'}<=set(df): df['car_performance_index']=df['team_car_speed_rating']*df['team_car_downforce_rating']
    return df

def feature_columns(df):
    cols=[c for c in ALLOWED_RAW_FEATURES if c in df.columns]+[c for c in ENGINEERED if c in df.columns]; cols=list(dict.fromkeys(cols)); bad=set(cols)&FORBIDDEN_FEATURES
    if bad: raise ValueError(f'Forbidden features: {sorted(bad)}')
    return cols

def prepare_dataset(data_dir='.'):
    train,test=load_data(data_dir); train=clean_data(train); test=clean_data(test); train=create_current_features(create_future_target(train)); test=create_current_features(test); cols=feature_columns(train); X=train[cols].copy(); y=train[TARGET].astype(float); groups=train['race_id'].copy()
    for c in cols:
        if c not in test: test[c]=np.nan
    return X,y,groups,test,cols

def build_preprocessor(X):
    cat=X.select_dtypes(include=['object','category']).columns.tolist(); num=[c for c in X.columns if c not in cat]
    return ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median'))]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),cat)])
