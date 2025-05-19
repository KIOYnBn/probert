import tensorflow as tf
from Model.Configs import Config
tf.config.run_functions_eagerly(True)


def main() -> None:
	config = Config()
	tfrecord_path = '../data_make/data/target'
	target_name = 'ca'
	file_path: str = f'{tfrecord_path}/{target_name}/length/train.tfrecord'
	tfrecord_files = [file_path]
	dataset = tf.data.TFRecordDataset(tfrecord_files)

	feature_description = {
		'protein_name': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
		'input_ids': tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
		'input_mask': tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
		'labels': tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
	}

	def _parse_example(input_example):
		return tf.io.parse_example(input_example, features=feature_description)

	batch_size = 40
	data_set = dataset.map(_parse_example)
	print(f'{data_set}')
	dataset = data_set.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	print(f'dataset size: {dataset}')
	count = 0
	for batch in dataset:
		tf.print("input_ids: ", batch['input_ids'])
		tf.print("input_mask: ", batch['input_mask'])
		tf.print("protein_name", batch['protein_name'])
		tf.print("labels: ", batch['labels'])


if __name__ == '__main__':
	"""
	don't join the model construction, just view the content is right or not when create tfrecord file
	"""
	main()
