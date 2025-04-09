import tensorflow as tf


class FullTokenizer(object):
	"""
	DESCRIPTION:
		the protien letter alphabet
	return:
		get_vocab: the letter alphabet
	"""
	
	def __init__(self, vocab_file):
		self.vocab = dict()
		with open(vocab_file, 'r') as vocab_file:
			vocab_lines = vocab_file.readlines()
			index = 0
			for vocab_line in vocab_lines:
				token = vocab_line.strip()
				self.vocab[token] = index
				index += 1
	
	def get_vocab(self):
		return self.vocab


def loading_data(input_file):
	"""
	DESCRIPTION:
		acquire protein name, sequence, label from txt file
	:param input_file:
	:return:
		names:list.protein names
		sequences:list.protein sequences
		labels:list.protein binding site with other
	"""
	names = []
	seqs = []
	labels = []
	with open(input_file, 'r') as f:
		count = 0
		lines = iter(f.readlines())
		for line in lines:
			if line.startswith('>'):
				name = line[1:].strip()
				names.append(name)
				seq = next(lines).strip()
				seqs.append(seq)
				label = next(lines).strip()
				label = list(map(int, label))
				labels.append(label)
				count += 1
			if count > 10000000000:
				break
	return names, seqs, labels


def convert_tokens_to_ids(tokens, vocab):
	"""
	DESCRIPTION:
		transfer string to number
	:param tokens:str	protein sequence
	:param vocab:dict	protein alphabet
	:return: ids:list	protein sequence showed by number list
	"""
	ids = []
	for position in range(len(tokens)):
		ids += [vocab[tokens[position].lower()]]
	return ids


def get_features(name, seq, label, vocab, max_length):
	"""
	DESCRIPTION:
		used in tokenize, because the same code use.
	:param name: str, pdb_id
	:param seq: list of number. the protein sequence
	:param label: list or int. binding_site or label
	:param vocab: dict. letter in sequence
	:param max_length: to padding the static list
	:return:
	"""
	"""pdb_id"""
	pdb_id = name.encode('utf-8')
	max_length = max_length -2
	
	pad_id = vocab["[PAD]"]
	start = vocab["[CLS"]
	end = vocab["[SEP]"]
	
	seq_id = convert_tokens_to_ids(seq, vocab)
	seq_len = len(seq)
	seq_id = [start] + seq_id + [pad_id] * (max_length - seq_len) + [end]
	
	mask = [1] + [1] * seq_len + [0] * (max_length - seq_len) + [1]
	
	labels = [0] + label + [0] * (max_length - seq_len) + [0]
	print(f'pdb_id:{pdb_id}\nseq_id:{len(seq_id)}\nmask:{len(mask)}\nlabel:{len(labels)}')
	return pdb_id, seq_id, mask, labels


def tokenize(file_path, output_path, vocab_file, train='train'):
	"""
	DESCRIPTION:
		convert txt or fasta to tfrecord
	:param file_path:
	:param output_path:
	:param vocab_file: the letter vocabulary
	:param train:bool
	:return:
	"""
	from Model.Configs import Config
	config = Config()
	names, seqs, labels = loading_data(file_path)
	vocabs = FullTokenizer(vocab_file).get_vocab()
	if train == 'train':
		name = 'train'
	else:
		name = 'test'
	with tf.io.TFRecordWriter(f'{output_path}/{name}_fragment.tfrecord') as writer:
		for line_index in range(len(names)):
			pdb_id, seq_id, mask, label = get_features(names[line_index], seqs[line_index], labels[line_index], vocabs, config.max_seq_length)
			
			features = {
				'protein_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pdb_id])),
				'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=seq_id)),
				'input_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=mask)),
				'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
			}
			example = tf.train.Example(features=tf.train.Features(feature=features))
			writer.write(example.SerializeToString())
		writer.close()


if __name__ == '__main__':
	"""convert txt or fasta to tfrecord"""
	target_name = 'fe2'
	train = 'test'
	path_dir = f'./data/target/{target_name}'
	input_path = f'{path_dir}/{train}_fragment.txt'
	save_dir = path_dir
	tokenize(file_path=input_path, output_path=save_dir, vocab_file='./data/vocab.txt', train=train)
