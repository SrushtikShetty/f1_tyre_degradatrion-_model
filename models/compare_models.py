import json
import os

import pandas as pd

from .linear_regression import train as train_linear
from .random_forest import train as train_rf
from .xgboost_model import train as train_xgb
from .neural_network import train as train_nn


def build_grouped_cv_report(results):
	return {
		'status': 'SUPPLEMENTARY_GROUPED_CV_ONLY',
		'metric_scope': 'supplementary_grouped_cv',
		'metric_protocol': '5-fold GroupKFold grouped by race_id using out-of-fold predictions',
		'primary_metric_source': 'artifacts/evaluation/final_metrics.json',
		'models': results,
	}


def run(data_dir='.', out_dir='artifacts'):
	os.makedirs(out_dir, exist_ok=True)
	results = [
		train_linear(data_dir, out_dir),
		train_rf(data_dir, out_dir),
		train_xgb(data_dir, out_dir),
		train_nn(data_dir, out_dir),
	]
	comparison = pd.DataFrame(results).sort_values('r2', ascending=False)
	comparison.to_csv(f'{out_dir}/model_comparison.csv', index=False)
	payload = build_grouped_cv_report(results)
	with open(f'{out_dir}/model_metrics.json', 'w', encoding='utf-8') as handle:
		json.dump(payload, handle, indent=2)
	print('SUPPLEMENTARY ONLY: grouped CV is not future-race performance.')
	print(comparison.to_string(index=False))
	return comparison


if __name__ == '__main__':
	run()
