from tensorflow.keras.optimizers import AdamW
import tensorflow as tf


def create_optimizer(
		config
):
	"""
	DESCRIPTION:
		model optimizer used in reducing loss
	:param config:
	:return:
	"""
	initial_learning_rate = config.initial_learning_rate
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		initial_learning_rate,
		decay_steps=config.learning_rate_decay_steps,  # 每多少步衰减一次
		decay_rate=config.learning_rate_decay_factor,  # 衰减系数（例如 0.96 表示每次衰减为之前的 96%）
		staircase=config.staircase  # True：阶梯式衰减；False：连续衰减
	)
	optimizer = AdamW(
		learning_rate=1e-4,
		weight_decay=1e-5,
		clipnorm=1
	)
	return optimizer
