import tensorflow as tf


class Config:
	def __init__(self):
		"""start set"""
		self.train = True
		self.train_epochs = 5
		self.train_epoch_steps = 20
		self.per_eval_steps = 100
		
		"""mask"""
		self.mask_prob = 0.15
		self.max_prediction_per_seq = 10
		self.proposal_distribution = 1
		self.max_position_embeddings = 27
		self.dropout_prob = 1
		self.vocab_file = './data_mask/vocab.txt'
		
		"""ProtBert"""
		self.vocab_size = 31
		self.max_position_embeddings = 25
		self.max_seq_length = 27
		
		self.hidden_dim = 1024
		self.num_layers = 6
		self.n_heads = 12
		self.size_per_head = self.hidden_dim // self.n_heads
		self.intermediate_size = 1024
		
		self.have_cross_attention = False
		
		self.initializer_range = 0.02
		self.activation = tf.keras.activations.gelu
		
		"""save"""
		self.metrics_save_path = './results/metrics.txt'
		self.save_weights_path = './weights/save_model'
		