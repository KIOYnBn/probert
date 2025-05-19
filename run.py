from typing import Union
import tensorflow as tf
from Model.Configs import Config
from Model.protbert import ProbertModel
from start_and_output.Input_module import get_features as get_dataset
from start_and_output.output_compute import ComputeMetrics, compute_loss
from Model.optimization import create_optimizer
tf.config.run_functions_eagerly(True)
tf.random.set_seed(42)
print(tf.__version__)


def initial() -> None:
	with open('./results/metrics.txt', 'w') as f:
		f.write('')
		

@tf.function
def main(input_files: Union[dict, bool] = None, train: bool = True) -> None:
	config: Config = Config(input_files=input_files, train=train)
	model: ProbertModel = ProbertModel(config)
	optimizer: tf.keras.optimizers = create_optimizer(config)
	model.compile(
		optimizer=optimizer,
		loss=compute_loss,
		metrics=ComputeMetrics(
			save_path=config.metrics_save_path,
			return_metrics=config.return_metrics,
			thresholds=config.thresholds)
	)
	if config.train:
		print(f"{'='*40}\n{'Train':^40}\n{'='*40}")
		data_train: tf.data.Dataset = get_dataset(config)
		model.fit(
			data_train,
			epochs=config.train_epochs,
			steps_per_epoch=config.steps_per_epoch
		)
		model.summary()
		print(f'{model.summary()}:summary')
		model.save_weights(config.save_weights_path, overwrite=True, save_format='tf')  # 仅保存权重
		print('model saved')
		initial()
	else:
		print(f"{'='*40}\n{'Test':^40}\n{'='*40}")
		data_eval: tf.data.Dataset = get_dataset(config)
		for batch_size in data_eval.take(1):
			model(batch_size[0])
		initial()
		model.load_weights(config.save_weights_path, )
		model.evaluate(data_eval, steps=config.per_eval_steps)
		print('model is evaluated')
		
		
def reduced_main() -> None:
	import os
	reduce_path: str = './data_make/data/target/k/reduce'
	all_types: list = os.listdir(f'{reduce_path}/train')
	for reduced_num in range(len(all_types)):
		reduced_type: str = all_types[reduced_num]
		print(f'{reduced_type=}')
		train_file: str = f'{reduce_path}/train/{reduced_type}/train_fragment.tfrecord'
		test_file: str = f'{reduce_path}/test/{reduced_type}/test_fragment.tfrecord'
		metrics_save_dir: str = f'results/reduced/{reduced_type}'
		os.makedirs(metrics_save_dir, exist_ok=True)
		metrics_save_path: str = f'{metrics_save_dir}/metrics.txt'
		input_files: dict = {'train_file': train_file, 'test_file': test_file, 'metrics_save_path': metrics_save_path}
		main(input_files, train=True)
		with open(metrics_save_path, 'w') as f:
			pass
		main(input_files, train=False)
		

if __name__ == '__main__':
	"""
	the python file runing the T5 model to exert all tasks
	"""
	main()
	# reduced_main()
