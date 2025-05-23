import keras_tuner as kt
from typing import Union


class Config:
	def __init__(
			self,
			input_files: Union[dict, bool] = None,
			train: bool = True,
			hp: Union[bool, kt.HyperParameters] = None) -> None:
		
		# 1): Hp select
		"""only used when exert keras-tuner to get the best hyperparameters"""
		self._hp = hp
		self._init_tunable_params()
		
		# 2): my test operation
		"""operation"""
		self.data_operation: str = 'sites'
		self.mask_operation: bool = True
		self.focus_label: int = 1
		self.ratio: int = 4
		
		# 3): Input Module
		if input_files is None:
			file_dir = './data_make/data/target'
			self.target_name: str = 'ca'
			self.train_file: str = f'{file_dir}/{self.target_name}/{self.data_operation}/train.tfrecord'
			self.test_file: str = f'{file_dir}/{self.target_name}/{self.data_operation}/test.tfrecord'
		else:
			self.train_file: str = input_files.get('train_file')
			self.test_file: str = input_files.get('test_file')
		print(f"{self.train_file=}, {self.test_file=}")
		self.batch_size: int = 4
		self.buffer_size: int = 1000000
		self.max_seq_length: int = 800
		self.input_label_size: int = self.max_seq_length
		
		# 4): Run Set
		self.train: bool = train
		self.train_epochs: int = 2   # 5
		self.steps_per_epoch: int = 20
		self.per_eval_steps: int = 2
		
		# 5): Mask
		self.mask_prob: float = 0.15
		self.max_prediction_per_seq: float = 50
		self.proposal_distribution: float = 1
		self.max_position_embeddings: int = self.max_seq_length
		self.vocab_file: str = './data_make/vocab.txt'
		
		# 6): ProBert
		self.vocab_size: int = 31

		self.have_cross_attention: bool = False
		
		# 7): Save
		if input_files is None:
			self.metrics_save_path: str = './results/metrics.txt'
			
		else:
			self.metrics_save_path: str = input_files.get('metrics_save_path')
		self.return_metrics: str = 'auc'
		self.thresholds: float = 0.5
		self.save_weights_path: str = './Models_hub/save_model.h5'
		
	def _init_tunable_params(self):
		if self._hp is None:
			# 1): ProBert
			self.hidden_dim: int = 1024
			self.num_layers: int = 6
			self.n_heads: int = 8
			self.size_per_head: int = self.hidden_dim // self.n_heads
			self.intermediate_dim: int = 2048
			self.initializer_range: float = 0.02
			self.activation: str = 'swish'
			
			# 2): Optimizer
			self.initial_learning_rate: float = 1e-3
			self.learning_rate_decay_steps: int = 5
			self.learning_rate_decay_factor: float = 0.85
			self.staircase: bool = False
		else:
			# 1): ProBert
			self.hidden_dim: int = self._hp.Int(
				name='hidden_dim',
				min_value=256,
				max_value=2048,
				step=256
			)
			self.num_layers: int = self._hp.Int(
				name='num_layers',
				min_value=3, max_value=10, step=1
			)
			possible_n_heads: list = [
				2 * selected_n for selected_n in range(2, 32, 2)
				if self.hidden_dim % selected_n == 0
			]
			self.n_heads: int = self._hp.Choice(
				name='n_heads',
				values=possible_n_heads, default=8
			)
			self.size_per_head: int = self.hidden_dim // self.n_heads
			self._intermediate_radio: int = self._hp.Int(
				name='intermediate_dim',
				min_value=1,
				max_value=3,
				step=1
			)
			self.intermediate_dim: int = self.hidden_dim * self._intermediate_radio
			self.initializer_range: float = self._hp.Float(
				name='initializer_range',
				min_value=0.0,
				max_value=0.10,
			)
			self.activation: str = self._hp.Choice('gate_activation', ['gelu', 'relu', 'tanh', 'swish'], default='gelu')
			
			# 2): opitimizer
			self.initial_learning_rate: float = self._hp.Float(
				name='initial_learning_rate',
				min_value=1e-7,
				max_value=1e-3,
				step=1e-1
			)
			self.learning_rate_decay_steps: int = self._hp.Int(
				name='learning_rate_decay_steps',
				min_value=1,
				max_value=10,
				step=1
			)
			self.learning_rate_decay_factor: float = self._hp.Float(
				name='learning_rate_decay_factor',
				min_value=0.70,
				max_value=0.95,
			)
			self.staircase: bool = self._hp.Choice(
				name='staircase',
				values=[True, False],
				default=False
			)
		