import tensorflow as tf
import tensorflow.keras as keras


class ProbertModel(keras.Model):
	def __init__(self, config):
		super(ProbertModel, self).__init__(name='ProbertModel')
		self.config = config
		self.embedding = TFBertEmbedding(self.config)
		self.encoder = TFBertcoder(self.config, name='Encoder')
		self.mlm_layer = MaskedLmOutput(config=self.config)
		self.decoder = TFBertcoder(self.config, name='Decoder')
		self.pool_dense = keras.layers.Dense()
		self.sites_loss = GetLoss(self.config)
		
	@staticmethod
	def create_attention_mask_from_input_mask(input_mask):
		shape = input_mask.shape
		batch = shape[0]
		length = shape[1]
		
		#   便于广播
		mask = tf.cast(tf.reshape(input_mask, [batch, 1, length]), tf.float32)
		
		broadcast_ones = tf.ones(shape=[batch, length, 1], dtype=tf.float32)
		
		#   #type:list   type:[batch, length, length]
		mask = broadcast_ones * mask
		return mask
	
	def build(self, input_shape):
		pass
	
	def call(self, features, training=None, mask=None):
		from Model.mask import mask
		from Model.dataclassutils import MaskedInputs
		input_mask_2d = features['input_mask']
		input_mask_3d = self.create_attention_mask_from_input_mask(input_mask_2d)
		
		inputs = MaskedInputs(
			protein_name=features['protein_name'],
			input_ids=features['input_ids'],
			input_mask=input_mask_2d,
			labels=features['input_ids'],
			masked_lm_ids=None,
			masked_lm_weights=None,
			masked_lm_positions=None
		)
		masked_inputs = mask(self.config, inputs)
		
		masked_inputs.input_mask = input_mask_3d
		
		embedding_output = self.embedding(masked_inputs)
		embedding_table = self.embedding.weight
		
		encoder_output = self.encoder(embedding_output)
		
		mlm_output = self.mlm_layer((encoder_output, embedding_table))
		
		decoder_output = self.decoder(encoder_output)
		
		pooled_output = decoder_output.input_ids[:, 0]
		pooled_output = self.pooler(pooled_output)
		pooled_output = self.pool_dense(pooled_output)
		encoder_output.input_ids = pooled_output
		encoder_output.input_mask = input_mask_2d
		sites_loss = self.sites_loss((encoder_output, embedding_table))
		total_loss = mlm_output.loss + sites_loss
		return total_loss
		
		
class TFBertEmbedding(keras.layers.Layer):
	def __init__(self, config):
		super(TFBertEmbedding, self).__init__(name='TFBertEmbedding')
		self.config = config
		self.weight = self.add_weight(
			name='weight_embeddings',
			shape=[config.vocab_size, config.hidden_dim],
			initializer=keras.initializers.RandomNormal(stddev=self.config.initializer_range),
		)
		self.token_type_embedding = self.add_weight(
			shape=[self.config.vocab_size, self.config.hidden_dim],
			name='token_embeddings',
			initializer=keras.initializers.TruncatedNormal(stddev=self.config.initializer_range)
		)
		self.position_embedding = self.add_weight(
			shape=[self.config.max_position_embeddings, self.config.hidden_dim],
			name='position_embeddings',
			initializer=keras.initializers.TruncatedNormal(stddev=self.config.initializer_range)
		)
		self.layer_norm = keras.layers.LayerNormalization(name='layer_norm')
	
	def call(self, inputs):
		input_ids = inputs.input_ids
		inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
		position_ids = tf.expand_dims(tf.range(start=0, limit=input_ids.shape[1]), axis=0)
		position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
		finall_embedding = inputs_embeds + position_embeds
		finall_embedding = self.layer_norm(finall_embedding)
		inputs.input_ids = finall_embedding
		return inputs
		
		
class TFBertcoder(keras.layers.Layer):
	def __init__(self, config, name):
		super().__init__(name=name)
		self.config = config
		self.layers = keras.Sequential([
			TFBertLayer(config, name=f'_layer{i}') for i in range(config.num_layers)
		])

	def call(self, inputs):
		output = self.layers(inputs)
		return output
	
	
class TFBertLayer(keras.layers.Layer):
	def __init__(self, config, name):
		super(TFBertLayer, self).__init__(name=f'TFBertLayer{name}')
		self.config = config
		self.attention = TFBertAttention(config, name='self_attention')
		if self.config.have_cross_attention:
			self.cross_attention = TFBertAttention(config, name='cross_attention')
		self.intermidiate_dense = keras.layers.Dense(
			units=self.config.intermediate_size,
			kernel_initializer=keras.initializers.TruncatedNormal(stddev=self.config.initializer_range),
			activation=self.config.activation,
			name='intermediate_dense'
		)
		self.output_dense = keras.layers.Dense(
			units=self.config.hidden_dim,
			kernel_initializer=keras.initializers.TruncatedNormal(stddev=self.config.initializer_range),
			name='bert_layer_output'
		)
		self.norm_layer = keras.layers.LayerNormalization(name='layer_norm')
	
	def call(self, inputs):
		attention_output = self.attention(inputs)
		if self.config.have_cross_attention:
			attention_output = self.cross_attention(attention_output)
		intermediate_output = self.intermidiate_dense(attention_output.input_ids)
		output = self.output_dense(intermediate_output)
		output = self.norm_layer(output)
		attention_output.input_ids = output
		return attention_output
		
	
class TFBertAttention(keras.layers.Layer):
	def __init__(self, config, name):
		super().__init__(name=name)
		self.config = config
		self.sqrt_atten_head_size = tf.sqrt(self.config.hidden_dim // self.config.n_heads)

		self.query_layer = tf.keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			name=f'query_layer',
		)
		self.key_layer = tf.keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			name=f'key_layer',
		)
		self.value_layer = tf.keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			name=f'value_layer',
		)
		self.attention_output_layer = tf.keras.layers.Dense(
			units=config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			name=f'attention_output_layer',
		)
		
		self.softmax_layer = keras.layers.Softmax()
		
		self.output_dense = keras.layers.Dense(
			units=self.config.hidden_dim,
			kernel_initializer=keras.initializers.RandomNormal(stddev=self.config.initializer_range),
			name='attention_dense'
		)
		
		self.norm_layer = keras.layers.LayerNormalization(name='attention_layer_norm')
	
	def transpose_for_scores(self, input_tensor, batch, length):
		output_tensor = tf.reshape(input_tensor, [batch, length, self.config.n_heads, self.config.size_per_head])
		output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
		return output_tensor
	
	def call(self, inputs):
		input_ids = inputs.input_ids
		input_mask = inputs.input_mask
		batch, length, dimension = input_ids.shape
		query = self.transpose_for_scores(self.query_layer(input_ids), batch, length)
		key = self.transpose_for_scores(self.query_layer(input_ids), batch, length)
		value = self.transpose_for_scores(self.key_layer(input_ids), batch, length)
		
		attention_scores = tf.matmul(query, key, transpose_b=True)
		attention_scores = tf.divide(attention_scores, self.sqrt_atten_head_size)
		
		attention_mask_expand = tf.expand_dims(input_mask, axis=1)
		adder = (1.0 - attention_mask_expand) * -10000.0
		
		attention_scores = tf.add(attention_scores, adder)
		
		attention_probs = self.softmax_layer(attention_scores)
		
		attention_output = tf.matmul(attention_probs, value)
		attention_output = self.transpose_for_scores(attention_output, batch, length)
		attention_output = tf.reshape(attention_output, [batch, length, dimension])
		hidden_states = self.output_dense(attention_output)
		hidden_states += hidden_states
		inputs.input_ids = self.norm_layer(hidden_states)
		return inputs


class MaskedLmOutput(keras.layers.Layer):
	def __init__(self, config):
		super().__init__()
		with tf.name_scope("lm-output"):
			self.config = config
			self.dense_layer = keras.layers.Dense(
				units=config.hidden_dim,
				kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
				bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
				activation=keras.activations.gelu,
				name='lm_output'
			)
			self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1)
			
	@staticmethod
	def gather_position(seq, position):
		shape = seq.shape
		batch = shape[0]
		length = seq.shape[1]
		dimension = seq.shape[2]
		position_shift = tf.expand_dims(length * tf.range(batch), -1)
		flat_positions = tf.reshape(position + position_shift, [-1])
		flat_sequence = tf.reshape(seq, [batch * length, dimension])
		gathered = tf.gather(flat_sequence, flat_positions)
		return tf.reshape(gathered, [batch, -1, dimension])
	
	def call(self, input_tuple):
		from Model.dataclassutils import MLMOutput
		inputs, embedding_table = input_tuple
		with tf.name_scope("lm-output"):
			relevant_hidden = self.gather_position(
				seq=inputs.input_ids,
				position=inputs.masked_lm_positions
			)
			hidden = self.dense_layer(relevant_hidden)
			
			hidden = self.norm_layer(hidden)
			
			#   shape = [batch, num_masked, vocab_size]
			labels = tf.one_hot(inputs.masked_lm_ids, depth=self.config.vocab_size, dtype=tf.float32)
			
			#   get_embedding_table.shape = [vocab_size, hidden_size]
			#   logits.shape = [batch, num_masked, vocab_size]
			logits = tf.matmul(hidden, embedding_table, transpose_b=True)
			logits = self.norm_layer(logits)
			
			per_loss = tf.nn.weighted_cross_entropy_with_logits(
				labels=labels,
				logits=logits,
				pos_weight=1  # 正样本权重
			)
			per_loss = per_loss * inputs.masked_lm_weights
			loss = tf.reduce_mean(per_loss)
			probs = tf.nn.sigmoid(logits)
			probs = tf.multiply(probs, inputs.masked_lm_weights)
			preds = tf.cast(probs > 0.5, tf.int32)
			return MLMOutput(logits=logits, probs=probs, per_example_loss=per_loss, loss=loss, preds=preds)


class GetLoss(keras.layers.Layer):
	def __init__(self, config):
		super(GetLoss, self).__init__(name='GetLoss')
		self.config = config
		self.hidden_layer = keras.layers.Dense(
			config.hidden_dim,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			use_bias=False,
			name='dis_output',
			activation=keras.activations.gelu
		)
		self.bias = self.add_weight(name='bias', shape=(self.config.vocab_size,), initializer='zeros', trainable=True)
		self.logits_dense = keras.layers.Dense(
			1,
			kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			use_bias=False, name='logits_dense',
		)
		self.norm_layer = keras.layers.LayerNormalization(axis=-1, name='norm_output')
		
	def call(self, inputs_tuple):
		from start_and_output.output_compute import compute_metrics
		inputs, embeding_weight = inputs_tuple
		input_ids = inputs.input_ids
		input_mask_2d = inputs.input_mask
		labels = tf.cast(inputs.labels, tf.float32)
		hidden = self.hidden_layer(input_ids)
		hidden = self.norm_layer(hidden)
		hidden = tf.matul(hidden, embeding_weight)
		hidden = tf.nn.bias_add(hidden, self.bias)
		logits = tf.squeeze(self.logits_dense(hidden), axis=-1)
		logits = self.norm_layer(logits)
		input_mask_2d = tf.cast(input_mask_2d, tf.float32)
		logits = tf.multiply(logits, input_mask_2d)
		
		test = False
		if test:
			loss = tf.losses.binary_focal_crossentropy(y_true=labels, y_pred=logits, from_logits=True, alpha=0.8, gamma=5.0)
		else:
			loss = tf.nn.weighted_cross_entropy_with_logits(
				labels=labels,
				logits=logits,
				pos_weight=1  # 正样本权重
			)
			loss = tf.reduce_mean(loss)
		probs = tf.nn.sigmoid(logits)
		probs = tf.multiply(probs, input_mask_2d)
		preds = tf.cast(probs > 0.6, tf.int32)
		compute_metrics(y_true=labels, y_pred=preds, save_path=self.config.metrics_save_path)
		
		return loss
