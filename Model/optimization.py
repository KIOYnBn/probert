from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import time


def create_optimizer(
		config
) -> tf.keras.optimizers.Optimizer:
	"""
	DESCRIPTION:
		model optimizer used in reducing loss
	:param config:
	:return:
	"""
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		config.initial_learning_rate,
		decay_steps=config.learning_rate_decay_steps,  # 每多少步衰减一次
		decay_rate=config.learning_rate_decay_factor,  # 衰减系数（例如 0.96 表示每次衰减为之前的 96%）
		staircase=config.staircase  # True：阶梯式衰减；False：连续衰减
	)
	optimizer = Nadam(
		learning_rate=lr_schedule,
	)
	return optimizer


class StepTrackerCallback(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs=None):
		self.current_step = 0  # 初始化步数
	
	def on_batch_end(self, batch, logs=None):
		self.current_step += 1  # 每完成一个batch，步数+1
		print(f"当前步数: {self.current_step}")
		with open('./steps.txt', 'w') as f:
			f.write(str(self.current_step))


class NewStepTrackerCallback(tf.keras.callbacks.Callback):
	def __init__(self, log_interval=100):
		super().__init__()
		self.log_interval = log_interval  # 日志打印间隔
		self.current_step = 0  # 跨epoch的全局步骤计数器

	def on_train_begin(self, logs=None):
		print(f"训练开始，总epoch数：{self.params['epochs']}，每epoch步数：{self.params['steps']}")

	def on_epoch_begin(self, epoch, logs=None):
		self.epoch_start_time = time.time()
		print(f"Epoch {epoch+1}/{self.params['epochs']} 开始")

	def on_train_batch_end(self, batch, logs=None):
		self.current_step += 1
		if self.current_step % self.log_interval == 0:
			loss = logs.get('loss', 'N/A')
			lr = self.model.optimizer.lr.numpy() if hasattr(self.model.optimizer, 'lr') else 'N/A'
			print(f"Step {self.current_step}: 损失={loss:.4f}, 学习率={lr:.6f}")

	def on_epoch_end(self, epoch, logs=None):
		epoch_time = time.time() - self.epoch_start_time
		avg_loss = logs.get('loss', 'N/A')
		print(f"Epoch {epoch+1} 完成 | 耗时：{epoch_time:.2f}s | 平均损失：{avg_loss:.4f}")
		