import tensorflow as tf
import tensorflow.keras as keras
from Model.mask import mask as mask_function
from Model.dataclassutils import MaskInputs, Output, MaskedInputs
from data_make.convert_tfrecord import FullTokenizer


@tf.function
class ProbertModel(tf.keras.Model):
	def __init__(self, config) -> None:
		super().__init__(name='ProbertModel')
		
		# 1): the base layer to forward
		self.config = config
		self.embedding = TFBertEmbedding(self.config)
		self.encoder = TFBertcoder(self.config, name='Encoder')
		self.mlm_layer = MaskedLmOutput(config=self.config)
		self.decoder = TFBertcoder(self.config, name='Decoder')
		self.sites_loss = GetLoss(self.config)
		
		# 2): to used in custom step
		self.generator_loss: tf.Tensor = tf.constant(0.0)
	
	def build(self, input_shape) -> None:
		pass
	
	def train_step(self, data):
		inputs: dict
		inputs, _ = data
		
		with (tf.GradientTape() as tape):
			model_output: Output = self(inputs, training=True)
			loss: tf.Tensor = self.compiled_loss(model_output.labels, model_output.loss, regularization_losses=self.losses)
			total_loss: tf.Tensor = self.generator_loss*0.1 + loss
		
		trainable_variables = self.trainable_variables
		gradients = tape.gradient(total_loss, trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, trainable_variables))
		self.compiled_metrics.update_state(model_output.labels, model_output.probs)
		return {m.name: m.result() for m in self.metrics}
	
	def test_step(self, data):
		inputs: dict
		inputs, _ = data
		model_output: Output
		model_output = self(inputs, training=False)
		loss: tf.Tensor = self.compiled_loss(model_output.labels, model_output.loss, regularization_losses=self.losses)
		total_loss: tf.Tensor = self.generator_loss * 0.1 + loss
		self.compiled_metrics.update_state(model_output.labels, model_output.probs)
		return {m.name: m.result() for m in self.metrics}
		
	def call(self, features: dict, training=None, mask=None) -> Output:
		# 1): mask
		inputs: MaskInputs = MaskInputs(
			protein_name=features['protein_name'],
			input_ids=features['input_ids'],
			input_mask=features['input_mask'],
			labels=features['labels'],
		)
		"""must have `input_ids` and `input_mask` when using mask_function"""
		masked_inputs: MaskedInputs = mask_function(self.config, inputs)
		
		# 2): Embedding
		hidden: tf.Tensor
		"""this embed_weight is the share layer represented the residue features"""
		embed_weight: tf.Tensor
		hidden, embed_weight = self.embedding(masked_inputs.input_ids)
		
		# 3): generator
		encoder_output: tf.Tensor = self.encoder((hidden, masked_inputs.input_mask))
		masked_inputs.hidden_states = encoder_output
		
		# 4): Masked Output Loss Compute
		mlm_output: Output = self.mlm_layer((masked_inputs, embed_weight))
		self.generator_loss = mlm_output.loss
		
		# 5): discriminator
		decoder_output: tf.Tensor = self.decoder((encoder_output, masked_inputs.input_mask))
		masked_inputs.hidden_states = decoder_output
		
		# 6): Site Output Loss Compute
		sites_output: Output = self.sites_loss((masked_inputs, embed_weight))
		return sites_output
		
		
class TFBertEmbedding(keras.layers.Layer):
	def __init__(self, config) -> None:
		super(TFBertEmbedding, self).__init__(name='TFBertEmbedding')
		self.config = config
		self.weight = self.add_weight(
			name='weight_embeddings',
			shape=[config.vocab_size, config.hidden_dim],
			initializer=keras.initializers.TruncatedNormal(stddev=self.config.initializer_range),
		)
		self.position_embedding = self.add_weight(
			shape=[self.config.max_position_embeddings, self.config.hidden_dim],
			name='position_embeddings',
			initializer=keras.initializers.TruncatedNormal(stddev=self.config.initializer_range)
		)
		self.layer_norm = keras.layers.LayerNormalization(name='layer_norm')
	
	def call(self, input_ids: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		input_ids: tf.Tensor = tf.cast(input_ids, tf.int32)
		weight_embed: tf.Tensor = tf.gather(params=self.weight, indices=input_ids)
		position_embed: tf.Tensor = tf.gather(self.position_embedding, indices=tf.range(input_ids.shape[1]))
		position_embed: tf.Tensor = tf.broadcast_to(position_embed[tf.newaxis, ...], weight_embed.shape)
		output: tf.Tensor = self.layer_norm(weight_embed+position_embed)
		return output, self.weight
		
		
class TFBertcoder(keras.layers.Layer):
	def __init__(self, config, name: str) -> None:
		super().__init__(name=name)
		self.layers_list: list = [
			TFBertLayer(config, name=f'Bert_layer_{num}') for num in range(config.num_layers)
		]

	def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
		# inputs = (hidden_states, attn_mask
		hidden_states, attn_mask = inputs
		del inputs
		for layer in self.layers_list:
			layer_inputs = (hidden_states, attn_mask)
			hidden_states = layer(layer_inputs)
		return hidden_states
	
	
class TFBertLayer(keras.layers.Layer):
	def __init__(self, config, name: str) -> None:
		super(TFBertLayer, self).__init__(name=f'TFBertLayer{name}')
		self.attention = TFBertAttention(config, name='self_attention')
		self.have_cross_attention: str = config.have_cross_attention
		if self.have_cross_attention:
			self.cross_attention = TFBertAttention(config, name='cross_attention')
		self.intermidiate_dense = keras.layers.Dense(
			units=config.intermediate_dim,
			kernel_initializer=keras.initializers.TruncatedNormal(stddev=config.initializer_range),
			activation=config.activation,
			name='intermediate_dense'
		)
		self.output_dense = keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=keras.initializers.TruncatedNormal(stddev=config.initializer_range),
			name='bert_layer_output'
		)
		self.norm_layer_one = keras.layers.LayerNormalization(name='layer_norm_one')
		self.norm_layer_two = keras.layers.LayerNormalization(name='layer_norm_two')
	
	def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
		# inputs = tuple(hidden_states, attn_mask)
		hidden: tf.Tensor = inputs[0]
		attn_output: tf.Tensor = self.attention(inputs)
		attn_output: tf.Tensor = self.norm_layer_one(attn_output+hidden)
		if self.have_cross_attention:
			attn_output: tf.Tensor = self.cross_attention(attn_output)
		intermediate_output: tf.Tensor = self.intermidiate_dense(attn_output)
		output: tf.Tensor = self.output_dense(intermediate_output)
		output: tf.Tensor = self.norm_layer_two(output)
		return output
		
	
class TFBertAttention(keras.layers.Layer):
	def __init__(self, config, name: str) -> None:
		super().__init__(name=name)
		self.n_heads = config.n_heads
		self.head_dim = config.hidden_dim // self.n_heads
		self.scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
		
		self.query_layer = tf.keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			name=f'query_layer',
		)
		self.key_layer = tf.keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
			name=f'key_layer',
		)
		self.value_layer = tf.keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
			name=f'value_layer',
		)
		self.softmax_layer = keras.layers.Softmax()
		
		self.output_dense = keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=keras.initializers.TruncatedNormal(stddev=config.initializer_range),
			name='attention_dense'
		)
	
	def _transpose_for_scores(self, input_tensor: tf.Tensor) -> tf.Tensor:
		batch, length, dim = input_tensor.shape
		assert dim == self.n_heads*self.head_dim
		output_tensor: tf.Tensor = tf.reshape(input_tensor, [batch, length, self.n_heads, self.head_dim])
		output_tensor: tf.Tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
		return output_tensor
	
	def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
		hidden_states, attn_mask = inputs
		query: tf.Tensor = self._transpose_for_scores(self.query_layer(hidden_states))
		key: tf.Tensor = self._transpose_for_scores(self.key_layer(hidden_states))
		value: tf.Tensor = self._transpose_for_scores(self.value_layer(hidden_states))
		
		attn_scores: tf.Tensor = tf.matmul(query, key, transpose_b=True)
		attn_scores: tf.Tensor = tf.divide(attn_scores, self.scale)
		
		attn_mask: tf.Tensor = tf.cast(attn_mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
		attn_scores: tf.Tensor = (1.0 - attn_mask) * -1e9 + attn_scores
		
		attn_probs: tf.Tensor = self.softmax_layer(attn_scores)
		
		attn_output: tf.Tensor = tf.matmul(attn_probs, value)
		attn_output: tf.Tensor = tf.transpose(attn_output, [0, 2, 1, 3])
		attn_output: tf.Tensor = tf.reshape(attn_output, hidden_states.shape)
		
		output: tf.Tensor = self.output_dense(attn_output)
		return output


class MaskedLmOutput(keras.layers.Layer):
	def __init__(self, config) -> None:
		super().__init__(name='MaskedLMOutput')
		self.thresholds = config.thresholds
		self.dense_layer = keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
			bias_initializer=tf.keras.initializers.zeros,
			activation=config.activation,
			name='lm_output'
		)
		self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1)
			
	@staticmethod
	def _gather_position(seq: tf.Tensor, position: tf.Tensor) -> tf.Tensor:
		batch, length, dimension = seq.shape
		position_shift = tf.expand_dims(length * tf.range(batch), -1)
		flat_positions = tf.reshape(position + position_shift, [-1])
		flat_sequence = tf.reshape(seq, [-1, dimension])
		gathered = tf.gather(flat_sequence, flat_positions)
		return tf.reshape(gathered, [batch, -1, dimension])
	
	def call(self, input_tuple: tuple[MaskedInputs, tf.Tensor]) -> Output:
		inputs, embedding_table = input_tuple
		relevant_hidden: tf.Tensor = self._gather_position(
			seq=inputs.hidden_states,
			position=inputs.masked_lm_positions
		)
		hidden: tf.Tensor = self.dense_layer(relevant_hidden)
		hidden: tf.Tensor = self.norm_layer(hidden)
		#   get_embedding_table.shape = [vocab_size, hidden_size]
		#   logits.shape = [batch, num_masked, vocab_size]
		logits: tf.Tensor = tf.matmul(hidden, embedding_table, transpose_b=True)
		#   shape = [batch, num_masked, vocab_size]
		labels: tf.Tensor = tf.one_hot(inputs.masked_lm_ids, depth=logits.shape[-1], dtype=tf.float32)
		loss: tf.Tensor = tf.nn.weighted_cross_entropy_with_logits(
			labels=labels,
			logits=logits,
			pos_weight=1  # 正样本权重
		)
		loss: tf.Tensor = tf.reduce_mean(loss)
		probs: tf.Tensor = tf.nn.sigmoid(logits)
		preds: tf.Tensor = tf.cast(probs > self.thresholds, tf.int32)
		return Output(logits=logits, probs=probs, loss=loss, preds=preds, labels=labels)


class GetLoss(keras.layers.Layer):
	def __init__(self, config) -> None:
		super(GetLoss, self).__init__(name='GetLoss')
		self.config = config
		self.hidden_layer = keras.layers.Dense(
			config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			use_bias=False,
			name='dis_output',
			activation=keras.activations.gelu
		)
		self.bias = self.add_weight(name='bias', shape=(config.vocab_size,), initializer='zeros', trainable=True)
		self.logits_dense = keras.layers.Dense(
			1,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			use_bias=False, name='logits_dense',
		)
		self.norm_logits = keras.layers.LayerNormalization(axis=-1, name='norm_logits')
		
	def call(self, inputs_tuple: tuple[MaskedInputs, tf.Tensor]) -> Output:
		"""结果输入"""
		inputs: MaskedInputs
		embed_weight: tf.Tensor
		# embedding_wieght:[vocab_size, hidden_dim]
		inputs, embed_weight = inputs_tuple

		# input_ids:[batch, seq, hidden_dim]
		input_ids: tf.Tensor = inputs.input_ids
		hidden: tf.Tensor = inputs.hidden_states
		labels: tf.Tensor = tf.cast(inputs.labels, tf.float32)
		input_mask_2d: tf.Tensor = tf.cast(inputs.input_mask, tf.float32)
		
		"""分类层处理"""
		hidden: tf.Tensor = self.hidden_layer(hidden)
		# hidden:[batch, seq, vocab_size]
		hidden: tf.Tensor = tf.matmul(hidden, embed_weight, transpose_b=True)
		hidden: tf.Tensor = tf.nn.bias_add(hidden, self.bias)
		# hidden:[batch, seq]
		hidden: tf.Tensor = tf.squeeze(self.logits_dense(hidden), axis=-1)
		logits: tf.Tensor = self.norm_logits(hidden)
		logits: tf.Tensor = tf.boolean_mask(logits, input_mask_2d)
		labels: tf.Tensor = tf.boolean_mask(labels, input_mask_2d)
		if self.config.train:
			if self.config.mask_operation == "mask sample":
				logits, labels = self._mask_sample(labels=labels, logits=logits, config=self.config)
			elif self.config.mask_operation == "mask focus":
				candidate: tf.Tensor = self._mask_focus(input_ids=input_ids, focus=self.config.focus)
				candidate: tf.Tensor = tf.boolean_mask(candidate, input_mask_2d)
				logits: tf.Tensor = tf.boolean_mask(logits, candidate)
				labels: tf.Tensor = tf.boolean_mask(labels, candidate)
		loss = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
		loss = tf.reduce_mean(loss)
		probs = tf.keras.activations.sigmoid(logits)
			
		"""0.65提升precision"""
		preds = tf.cast(probs > self.config.thresholds, tf.float32)
		# print(f"{probs=}\n{labels=}")
		"""反应模型预测倾向"""
		return Output(logits=logits, probs=probs, loss=loss, preds=preds, labels=labels)
	
	@staticmethod
	def _mask_sample(
			config,
			logits: tf.Tensor,
			labels: tf.Tensor,) -> tuple[tf.Tensor, tf.Tensor]:
		import random
		positive: tf.Tensor = labels == config.focus_label
		positive: tf.Tensor = tf.where(positive)
		negative: tf.Tensor = labels != config.focus_label
		negative: tf.Tensor = tf.where(negative)
		# ratio: float = random.uniform(1.0, 5.0) * random.choice([0, 1])
		ratio: float = config.ratio
		if len(positive) < len(negative):
			selected_position: tf.Tensor = negative
			length: float = int(len(positive) * ratio)
			output: tf.Tensor = positive
		else:
			selected_position: tf.Tensor = positive
			output: tf.Tensor = negative
			length: float = int(len(negative) * ratio)
		minority: tf.Tensor = tf.random.shuffle(selected_position)
		mask: tf.Tensor = tf.zeros_like(labels, dtype=tf.bool)
		# 使用索引更新掩码
		all_position: tf.Tensor = tf.concat([output, minority[:length]], axis=0)
		mask: tf.Tensor = tf.tensor_scatter_nd_update(
			mask,
			all_position,  # 需要置为True的坐标（形状为[N, D]）
			tf.ones([tf.shape(all_position)[0]], dtype=tf.bool),  # 填充True值
		)
		positions: tf.Tensor = tf.cast(mask, tf.float32)
		logits: tf.Tensor = tf.boolean_mask(logits, positions)
		labels: tf.Tensor = tf.boolean_mask(labels, positions)
		return logits, labels
	
	@staticmethod
	def _mask_focus(input_ids: tf.Tensor, focus: list[str]) -> tf.Tensor:
		token: dict = FullTokenizer().get_vocab()
		focus_token: list = []
		for res in focus:
			assert res in token.keys(), f"this {res} is not in the vocabulary"
			focus_token.append(token[res.upper()])
		focus_token: list = tf.expand_dims(tf.cast(list(map(int, focus_token)), tf.float32), axis=0)
		output: tf.Tensor = tf.equal(tf.expand_dims(input_ids, axis=-1), focus_token)
		output: tf.Tensor = tf.reduce_any(output, axis=-1)
		return tf.cast(output, tf.int32)
	