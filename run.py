import tensorflow as tf
tf.config.run_functions_eagerly(True)


def loss_fn(y_true, y_pred):
	return y_pred


def main():
	print(tf.__version__)
	from Model.Configs import Config
	from Model.protbert import ProbertModel
	from start_and_output.Input_module import get_features
	from Model.optimization import create_optimizer
	config = Config()
	model = ProbertModel(config)
	
	optimizer = create_optimizer(config)
	model.compile(optimizer=optimizer, loss=loss_fn)
	
	if config.train:
		print('pretrain train')
		data_train = get_features(config)
		print('dataset is ok')
		model.fit(data_train, epochs=config.train_epochs, steps_per_epoch=config.train_epoch_steps)
		print('model is ok')
		model.summary()
		print(f'{model.summary()}:summary')
		model.save_weights(config.save_weights_path, overwrite=True, save_format='tf')  # 仅保存权重
		print('model saved')
	else:
		print('pretrain test')
		data_eval = get_features(config)
		print('dataset is ok')
		for batch_size in data_eval.take(1):
			model(batch_size[0])
		model.load_weights(config.save_weights_path)
		print('model loaded')
		model.evaluate(data_eval, steps=config.per_eval_steps)
		print('model is evaluated')


if __name__ == '__main__':
	"""
	the python file runing the T5 model to exert all tasks
	"""
	main()
