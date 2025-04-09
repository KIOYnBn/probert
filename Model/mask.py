
import tensorflow as tfv1
from dataclasses import fields


def heading(emphasize):
	split_line = '=' * 80
	print(split_line)
	print(emphasize)
	print(split_line)
	

def mask(config, inputs):
	heading(inputs.input_ids.shape)
	if tfv1.executing_eagerly():
		heading('eager execution')
	else:
		heading('disable eager execution')
	#   最大掩码数量  type:int
	max_mask_number = config.max_prediction_per_seq
	#   掩码概率    type:float
	mask_prob = config.mask_prob
	#   位置掩码分布概率    type:float/int
	proposal_distribution = config.proposal_distribution
	#   输入数据的特征 type:int
	batch, length = inputs.input_ids.shape
	"""基本信息输入"""
	
	#   有效渔猎， 可掩码为True  type:list_bool, shape=[batch, length]
	candidates_mask = tfv1.cast(get_candidates_mask(config, inputs), tfv1.int32)
	#   最大有效掩码数量    type:int
	number_tokens = tfv1.cast(tfv1.reduce_sum(candidates_mask, -1), tfv1.float32)
	#   可以掩码数   type:int
	number_to_token = tfv1.maximum(1, tfv1.minimum(max_mask_number, tfv1.cast(tfv1.round(number_tokens * mask_prob), tfv1.int32)))
	#   设置被掩码的数量不超过序列有效长度 type:list, e.g.:[1, 1, 1, 0, 0, 0]
	masked_lm_weights = tfv1.cast(tfv1.sequence_mask(number_to_token, max_mask_number), tfv1.float32)
	"""这一部分获取了限制掩码数的权重矩阵"""
	
	#   候选掩码布尔矩阵转化为数值矩阵 type:list_float shape = [batch, length]
	candidates_mask_float = tfv1.cast(candidates_mask, tfv1.float32)
	#   加上掩码位置概率    type:list_float shape = [batch, length]
	sample_prob = (proposal_distribution * candidates_mask_float)
	#   概率归一    type:list_float shape = [batch, length]
	sample_prob = sample_prob / tfv1.reduce_sum(sample_prob, -1, keepdims=True)
	#   掩码概率不参与梯度传播
	sample_prob = tfv1.stop_gradient(sample_prob)
	#   对数  type:llist_float    shape = [batch, length]
	sample_logtis = tfv1.math.log(sample_prob)
	#   随机选择被掩码位置   type:list_int   $$\red{shape = [batch, max_mask_number]}$$  $$\red{大概率有重复项， 值为sequence的位置索引}
	mask_lm_positions = tfv1.random.categorical(sample_logtis, max_mask_number)
	mask_lm_positions = tfv1.cast(mask_lm_positions, tfv1.int32)
	#   #type:list_int   shape = [batch, max_mask_number]
	mask_lm_positions *= tfv1.cast(masked_lm_weights, tfv1.int32)
	"""这一部分计算了位置掩码的概率， 获得被掩码的位置"""
	
	#   获取偏移位置向量    type:list   shape = [batch, 1]
	shift = tfv1.expand_dims(length * tfv1.range(batch), -1)
	#   偏移后被掩码的位置   type:list   shape = [batch * max_mask_number, 1]
	token_positions = tfv1.reshape(mask_lm_positions + shift, [-1, 1])
	#   获取掩码位置对应id  type:list   shape = [batch * max_mask_number]
	masked_lm_ids = tfv1.gather(tfv1.reshape(inputs.input_ids, [-1]), token_positions)
	masked_lm_ids = tfv1.reshape(masked_lm_ids, [batch, -1])
	#   #type:list   #shape = [batch, None]
	masked_lm_ids = tfv1.cast(masked_lm_ids, tfv1.int32)
	"""这一部分提取到被掩码位置的数值id"""
	
	#   #type:list   #shape = [batch, max_mask_number]
	replace_with_mask_positions = mask_lm_positions * tfv1.cast(
		tfv1.less(tfv1.random.uniform([batch, max_mask_number]), 0.85), tfv1.int32)
	vocab = get_vocab(config)
	inputs_ids, _ = scatter_update(
		config,
		inputs.input_ids,  # #type:list   #shape = [batch, length]
		tfv1.fill([batch, max_mask_number], vocab["[MASK]"]),  # #type:list  #shape = [batch, max_mask_number]
		replace_with_mask_positions  # #type:list  #shape = [batch, max_mask_number]
	)
	return get_updated_inputs(
		inputs,
		masked_lm_positions=mask_lm_positions,
		input_ids=tfv1.stop_gradient(inputs_ids),
		masked_lm_ids=masked_lm_ids,
		masked_lm_weights=masked_lm_weights
	)


def get_candidates_mask(config, inputs):
	vocab = get_vocab(config)
	heading(inputs.input_ids.shape)
	ignore_ids = [vocab["[SEP]"], vocab["[CLS]"], vocab["[MASK]"]]
	candidates_mask = tfv1.ones_like(inputs.input_ids, dtype=tfv1.bool)
	heading(candidates_mask.shape)
	for ignore_id in ignore_ids:
		candidates_mask &= tfv1.not_equal(inputs.input_ids, ignore_id)
	candidates_mask &= tfv1.cast(inputs.input_mask, dtype=tfv1.bool)
	return candidates_mask


def scatter_update(config, sequence, updates, positions):
	#   shape = [batch, length]
	shape = sequence.shape
	if shape.ndims == 2:
		batch, length = shape
		dimension = 1
		sequence = tfv1.expand_dims(sequence, axis=-1)
	else:
		batch, length, dimension = shape
	max_mask_number = config.max_prediction_per_seq
	"""基本信息获取"""
	
	#   平展偏移量   type:list   shape = [batch, 1]
	shift = tfv1.expand_dims(length * tfv1.range(batch), -1)
	#   掩码的位置矩阵平展   type:list   shape = [batch, max_mask_number]
	flat_positions = tfv1.reshape(positions + shift, [-1, 1])
	"""将position平展"""
	
	#   将全mask矩阵平展  type:list   shape = [batch*max_mask_number, 1]
	flat_updates = tfv1.reshape(updates, [-1, dimension])
	#   形成类似excel表格的矩阵， 初始标记全为mask  type:list   shape = [batch*length, dimension]
	updates = tfv1.scatter_nd(flat_positions, flat_updates, [batch * length, dimension])
	#   #type:list   shape = [batch, length, dimension]
	updates = tfv1.reshape(updates, [batch, length, dimension])
	updates = tfv1.cast(updates, tfv1.float32)
	"""获得一个初始标记mask的矩阵， 匹配序列形状"""
	
	#   获得一个全1矩阵， 用于计算掩码次数  type=list   shape = [batch * max_mask_number]
	flat_updates_mask = tfv1.ones([batch * max_mask_number], dtype=tfv1.int32)
	updates_mask = tfv1.scatter_nd(flat_positions, flat_updates_mask, [batch * length])
	#   shape = [batch, length]
	updates_mask = tfv1.reshape(updates_mask, [batch, length])
	#   shape = [batch, length, 1]
	not_first_token = tfv1.concat([tfv1.zeros([batch, 1], dtype=tfv1.int32), tfv1.ones((batch, length - 1), dtype=tfv1.int32)], -1)
	updates_mask *= not_first_token
	updates_mask_3d = tfv1.expand_dims(updates_mask, -1)
	"""与上一大步类似， 不过， 这里是计算了每个位置被掩码的次数， 避免多次掩码"""
	
	#   shape = [batch, length, 1]
	sequence = tfv1.cast(sequence, tfv1.float32)
	updates_mask_3d = tfv1.cast(updates_mask_3d, tfv1.float32)
	#   shape = [batch, length, 1]
	updates /= tfv1.maximum(1.0, updates_mask_3d)
	"""获取掩码结果的平均值， 即某个位置如果被多次掩码， 则只保留一次"""
	
	#   #只用记录是否被掩码， 不关系重复   [batch, length]
	updates_mask = tfv1.minimum(updates_mask, 1)
	#   [batch, length, 1]
	updates_mask_3d = tfv1.minimum(updates_mask_3d, 1.0)
	#   [batch, length, 1]
	updated_sequence = (((1.0 - updates_mask_3d) * sequence) + updates_mask_3d * updates)
	if shape.ndims == 2:
		#   舍去最后一个1的维度
		updated_sequence = tfv1.squeeze(updated_sequence, axis=-1)
	"""将更改序列的id序列"""
	
	return updated_sequence, updates_mask


#   updated_sequence记录了掩码的id序列， updates_mask记录了掩码的位置
#   updated_sequence    shape = [batch, length]
#   updates_mask    shape = [batch, length]

def get_dataclass_name(the_dataclass):
	return [field.name for field in fields(the_dataclass)]


def get_updated_inputs(inputs, **kwargs):
	features_name = get_dataclass_name(inputs)
	features = {
		field_name: getattr(inputs, field_name) for field_name in features_name
	}
	for k, v in kwargs.items():
		features[k] = v
	return features_to_inputs(features)


def features_to_inputs(features):
	from Model.dataclassutils import MaskedInputs
	return MaskedInputs(
		protein_name=features["protein_name"],
		input_ids=features["input_ids"],
		input_mask=features["input_mask"],
		masked_lm_positions=(features["masked_lm_positions"]),
		masked_lm_ids=(features["masked_lm_ids"]),
		masked_lm_weights=(features["masked_lm_weights"]),
		labels=(features['labels'])
	)


def get_vocab(config):
	from data_make.convert_tfrecord import FullTokenizer
	vocab = FullTokenizer(config.vocab_file).get_vocab()
	return vocab
