import tensorflow as tf
import keras_tuner as kt
import json

from Model.protbert import ProbertModel
from Model.Configs import Config
from Model.optimization import create_optimizer
from start_and_output.Input_module import get_features
from start_and_output.output_compute import ComputeMetrics, compute_loss
tf.config.run_functions_eagerly(True)


def tuner_main() -> None:
	print(tf.__version__)
	
	def model_builder(hp: kt.HyperParameters) -> tf.keras.Model:
		config = Config(train=True, hp=hp)
		model = ProbertModel(config)
		optimizer = create_optimizer(config)
		model.compile(
			optimizer=optimizer,
			loss=compute_loss,
			metrics=ComputeMetrics(
				save_path=config.metrics_save_path,
				return_metrics=config.return_metrics,
				thresholds=config.thresholds),
		)
		return model
	
	data_train: tf.data.Dataset = get_features(Config(train=True))
	data_eval: tf.data.Dataset = get_features(Config(train=False))
	
	tuner = kt.Hyperband(
		model_builder,
		objective=kt.Objective("val_auc", direction="max"),  # 根据验证集 AUC 优化
		max_epochs=30,
		factor=3,
		directory='tuner_results',
		project_name='fe2_optimization',
		overwrite=True
	)
	tuner.search(
		data_train,
		validation_data=data_eval,
		epochs=50,
		steps_per_epoch=20,
		callbacks=[
			tf.keras.callbacks.EarlyStopping(patience=5),
			tf.keras.callbacks.TensorBoard(log_dir='./logs')
		]
	)
	best_hp = tuner.get_best_hyperparameters()[0]
	print("=== 最佳超参数组合 ===")
	for param, value in best_hp.values.items():
		print(f"{param}: {value}")
	best_hp = tuner.get_best_hyperparameters()[0]
	best_metrics = tuner.get_best_models(num_models=1)[0].evaluate(data_eval)
	
	save_data = {
		"hyperparameters": best_hp.values,
		"metrics": dict(zip(tuner.oracle.objective_names, best_metrics))
	}
	
	with open('best_config.json', 'w') as f:
		json.dump(save_data, f, indent=2)
		

if "__main__" == __name__:
	tuner_main()
	