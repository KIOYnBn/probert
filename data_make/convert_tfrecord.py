from typing import Union
import tensorflow as tf
from Model.Configs import Config


class FullTokenizer(object):
	"""
	DESCRIPTION:
		the protien letter alphabet
	return:
		get_vocab: the letter alphabet
	"""
	
	def __init__(self, vocab_file: str) -> None:
		self.vocab: dict[str, int] = dict()
		with open(vocab_file, 'r') as vocab_file:
			vocab_lines: list[str] = vocab_file.readlines()
			index: int = 0
			for vocab_line in vocab_lines:
				token: str = vocab_line.strip()
				self.vocab[token] = index
				index += 1
	
	def get_vocab(self) -> dict[str, int]:
		return self.vocab
	

def loading_data(input_file: str) -> tuple[list, list, list]:
	"""
	DESCRIPTION:
		acquire protein name, sequence, site label from txt file
	:param input_file:file path
	:return:
		names:list.protein names
		sequences:list.protein sequences
		labels:list.protein binding site with other
	"""
	names: list = []
	seqs: list = []
	labels: list = []
	with open(input_file, 'r') as f:
		count: int = 0
		lines = iter(f.readlines())
		for line in lines:
			if line.startswith('>'):
				name, seq, label = line.strip().split('\t')
				name: str = name[1:]
				label: list = list(map(int, label))
				
				if len(seq) > 798:
					print(f'{name} > 800')
				else:
					names.append(name)
					seqs.append(seq)
					labels.append(label)
				count += 1
			if count > 10000000:
				break
	return names, seqs, labels


def convert_tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
	"""
	DESCRIPTION:
		transfer string to number
	:param tokens:str	protein sequence
	:param vocab:dict	protein alphabet
	:return: ids:list	protein sequence showed by number list
	"""
	ids: list = []
	for position in range(len(tokens)):
		ids += [vocab[tokens[position].upper()]]
	return ids


def get_features(
		name: str,
		seq: list[str],
		label: Union[list, int],
		vocab: dict[str, int],
		max_length: int) -> tuple[bytes, list, list, list]:
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
	pdb_id: bytes = name.encode('utf-8')
	max_length: int = max_length - 2
	
	pad_id: int = vocab["[PAD]"]
	start: int = vocab["[CLS]"]
	end: int = vocab["[SEP]"]
	
	seq_id: list[int] = convert_tokens_to_ids(seq, vocab)
	
	seq_len: int = len(seq_id)
	assert max_length - seq_len >= 0
	
	seq_id: list[int] = [start] + seq_id + [pad_id] * (max_length - seq_len) + [end]
	
	pad: list[int] = [1] + [1] * seq_len + [0] * (max_length - seq_len) + [1]
	
	labels: list[int] = [0] + label + [0] * (max_length - seq_len) + [0]
	# print(f'pdb_id:{pdb_id}\nseq_id:{len(seq_id)}\npad:{len(pad)}\nlabel:{len(labels)}')
	return pdb_id, seq_id, pad, labels


def tokenize(
		file_path: str,
		output_path: str,
		vocab_file: str,
		train: str = 'train') -> None:
	"""
	DESCRIPTION:
		convert txt or fasta to tfrecord
	:param file_path:
	:param output_path:
	:param vocab_file: the letter vocabulary
	:param train:bool
	:return:
	"""

	config: Config = Config()
	names, seqs, labels = loading_data(file_path)
	vocabs: dict[str, int] = FullTokenizer(vocab_file).get_vocab()
	with tf.io.TFRecordWriter(f'{output_path}/{train}.tfrecord') as writer:
		for line_index in range(len(names)):
			pdb_id, seq_id, pad, label = get_features(
				names[line_index],
				seqs[line_index],
				labels[line_index],
				vocabs,
				config.max_seq_length)
			
			features: dict[str, tf.Tensor] = {
				'protein_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pdb_id])),
				'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=seq_id)),
				'input_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=pad)),
				'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
			}
			example = tf.train.Example(features=tf.train.Features(feature=features))
			writer.write(example.SerializeToString())
		writer.close()


def main() -> None:
	"""convert txt or fasta to tfrecord"""
	target_name: str = 'ca'
	train: str = 'test'
	path_dir: str = f'./data/target/{target_name}/sites'
	input_path: str = f'{path_dir}/{train}.csv'
	save_dir: str = path_dir
	tokenize(file_path=input_path, output_path=save_dir, vocab_file='./vocab.txt', train=train)


if __name__ == '__main__':
	main()
