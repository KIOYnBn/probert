import tensorflow as tf


@tf.function
def get_features(config):
	"""
	DESCRIPTION:
		Before the model start, loading the data, which is tensorflow datatype
	:param config: self define object class
	:return:
	"""
	
	def _parse_example(input_example):
		examples = tf.io.parse_single_example(input_example, features=feature_description)
		return {
			'protein_name': examples['protein_name'],
			'input_ids': examples['input_ids'],
			'input_mask': examples['input_mask'],
			'labels': examples['labels']}, examples['labels']
	
	# make sure the file path
	if config.train:
		input_file = config.train_file
	else:
		input_file = config.test_file
	
	# create dataset
	dataset = tf.data.TFRecordDataset(input_file)
	feature_description = {
		'protein_name': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
		'input_ids': tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
		'input_mask': tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
		'labels': tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
	}
	data_set = dataset.map(_parse_example)
	dataset = data_set.shuffle(buffer_size=config.buffer_size).repeat().batch(config.batch_size, drop_remainder=True).cache()
	print(f"def get_features: {dataset=}")
	return dataset
