from typing import Union
import os
import tensorflow as tf
from Model.Configs import Config
from Model.protbert import ProbertModel
from start_and_output.Input_module import get_features as get_dataset
from start_and_output.output_compute import ComputeMetrics, compute_loss
from Model.optimization import create_optimizer
tf.config.run_functions_eagerly(True)
print(tf.__version__)


def initial(input_file: str) -> None:
	with open(input_file, 'w') as f:
		f.write('')
		
		
def main(input_files: Union[dict, bool] = None, train: bool = True) -> None:
	config: Config = Config(input_files=input_files, train=train)
	model: ProbertModel = ProbertModel(config)
	optimizer: tf.keras.optimizers = create_optimizer(config)
	
	if config.train:
		print(f"{'='*40}\n{'Train':^40}\n{'='*40}")
		model.compile(
			optimizer=optimizer,
			loss=compute_loss,
			metrics=ComputeMetrics(
				save_path=config.metrics_save_path,
				return_metrics=config.return_metrics,
				thresholds=config.thresholds)
		)
		data_train: tf.data.Dataset = get_dataset(config)
		model.fit(
			data_train,
			epochs=config.train_epochs,
			steps_per_epoch=config.steps_per_epoch
		)
		model.summary()
		print(f'{model.summary()}:summary')
		model.save_weights(config.save_weights_path, overwrite=True, save_format='h5')  # 仅保存权重
		print('model saved')
		initial('./results/metrics.txt')
	else:
		print(f"{'='*40}\n{'Test':^40}\n{'='*40}")
		model.compile(
			loss=compute_loss,
			metrics=ComputeMetrics(
				save_path=config.metrics_save_path,
				return_metrics=config.return_metrics,
				thresholds=config.thresholds)
		)
		data_eval: tf.data.Dataset = get_dataset(config)
		for batch_size in data_eval.take(1):
			model(batch_size[0])
		initial('./results/metrics.txt')
		model.load_weights(config.save_weights_path)
		model.evaluate(data_eval, steps=config.per_eval_steps)
		print('model is evaluated')
		
		
def reduce() -> None:
	target_name: str = 'na'
	reduce_dir: str = f'./data_make/data/target/{target_name}/reduce'
	all_types: list = os.listdir(reduce_dir)
	for reduce_type in all_types:
		type_dir: str = f'{reduce_dir}/{reduce_type}'
		train_path: str = f"{type_dir}/train.tfrecord"
		eval_path: str = f"{type_dir}/test.tfrecord"
		metrics_path: str = f"./results/{reduce_type}.txt"
		dataset_dict: dict = {
			'train_file': train_path,
			'test_file': eval_path,
			'metrics_save_path': metrics_path
		}
		main(input_files=dataset_dict, train=True)
		initial(metrics_path)
		main(input_files=dataset_dict, train=False)
		

if __name__ == '__main__':
	"""
	the python file runing the T5 model to exert all tasks
	"""
	# main()
	reduce()
	
