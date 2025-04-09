from icecream import ic


def creat_fasta(input_file, output_file):
	"""
	DESCRIPTION:
		due to the original txt or fasta include the bind site,
		fail to use cd-hit, so create a new fasta to cd-hit
	:param input_file:
	:param output_file:
	:return:
	"""
	protein_context = []
	with open(input_file, 'r') as fasta_file:
		for line in fasta_file:
			if line.startswith('>'):
				name = line.strip().split("\t")[0]
				seq = next(fasta_file).strip()
				protein_line = name + "\n" + seq + "\n"
				protein_context.append(protein_line)
				ic(seq)
	with open(output_file, "w") as new_file:
		ic(1)
		for protein in protein_context:
			new_file.write(protein)


def read_bind_site(bind_site_file):
	"""
	DESCRIPTION:
		get bind site from original txt or fasta
	:param bind_site_file: the txt or fasta file of protein including bind site
	:return:
	"""
	protein_dict = dict()
	with open(bind_site_file, 'r') as bind_site_file:
		for line in bind_site_file:
			if line.startswith('>'):
				name = line.strip().split("\t")[0]
				seq = next(bind_site_file).strip()
				bind_site = next(bind_site_file).strip()
				protein_dict[name] = bind_site
	return protein_dict


def read_fasta(fasta_file):
	"""
	DESCRIPTION:
		get sequence
	:param fasta_file:
	:return: dict
	"""
	protein_dict = dict()
	with open(fasta_file, 'r') as fasta_file:
		for line in fasta_file:
			if line.startswith('>'):
				name = line.strip().split("\t")[0]
				seq = next(fasta_file).strip()
				protein_dict[name] = seq
	return protein_dict


def creat_ready_fasta(seqs, bind_sites, save_path):
	"""
	DESCRIPTION:
		add bind sites to ready cd-hit fasta
	:param seqs:
	:param bind_sites:
	:param save_path:
	:return:
	"""
	protein_line = ''
	for name, seq in seqs.items():
		bind_site = bind_sites[name]
		ic(name, seq, bind_site)
		protein_line += name + "\n" + seq + "\n" + bind_site + "\n"
	with open(save_path, 'w') as ready_fasta:
		ready_fasta.write(protein_line)


def slice_fragment(input_file, output_file, focus):
	"""
	DESCRIPTION:
		slice length==25 fragemnt of protein sequence.
		the postive fragment is the central residue is bind site and focus
	:param input_file:
	:param output_file:
	:param focus:
	:return:
	"""
	positive_fragment = ''
	negative_fragment = ''
	with open(input_file, 'r') as input_file:
		for line in input_file:
			if line.startswith('>'):
				name = line.strip().split("\t")[0]
				seq = next(input_file).strip()
				site = next(input_file).strip()
				for position in range(len(seq)):
					if seq[position] in focus:
						if max(0, position - 12) >= 0 and max(len(seq), position + 13) <= len(seq):
							if '1' not in site[position - 12:position] and '1' not in site[position + 1:position + 13]:
								label = site[position - 12:position + 13]
								ic(label)
								fragment_line = seq[position - 12:position + 13]
								if len(fragment_line) == 25:
									if site[position] != "0":
										ic("label:1")
										positive_fragment += name + "\n" + fragment_line + "\n" + label + "\n"
									else:
										ic("label:0")
										negative_fragment += name + "\n" + fragment_line + "\n" + label + '\n'
	with open(f'{output_file}/positive_fragment', 'w') as positive_file:
		positive_file.write(positive_fragment)
	with open(f'{output_file}/negative_fragment', 'w') as negative_file:
		negative_file.write(negative_fragment)


def slice_train(input_path, save_dir):
	import random
	with open(input_path, 'r') as f_pro:
		lines = f_pro.readlines()
		length = len(lines) // 3
		numbers = [3 * x for x in range(length)]
		eval_length = len(numbers) * 3 // 10
		selected = random.sample(numbers, eval_length)
		print(selected)
		with open(f'{save_dir}/test_fragment.txt', 'w') as f_test:
			for line_index in selected:
				line = lines[line_index] + lines[line_index + 1] + lines[line_index + 2]
				f_test.write(line)
		with open(f'{save_dir}/train_fragment.txt', 'w') as f_train:
			for line_index in numbers:
				if line_index in selected:
					continue
				else:
					line = lines[line_index] + lines[line_index + 1] + lines[line_index + 2]
					f_train.write(line)


def integrate_train_or_eval(positive_file, negative_file, output_file):
	with open(positive_file, 'r') as positive_file:
		positive_lines = positive_file.readlines()
		positive_examples_num = len(positive_lines)
	with open(negative_file, 'r') as negative_file:
		negative_lines = negative_file.readlines()
	with open(output_file, mode='w') as output_file:
		for line_posotive in positive_lines:
			if line_posotive != '':
				output_file.write(line_posotive)
		for line_index in range(min(positive_examples_num * 3, len(negative_lines))):
			line = negative_lines[line_index]
			if line != '':
				output_file.write(line)


def main():
	operation = 'train or eval'
	file_path = './data/target'
	target = 'fe2'
	residue_path = f'{file_path}/{target}'
	if operation == 'train or eval':
		input_file = f'{residue_path}/fe2_fragment.txt'
		save_dir = residue_path
		slice_train(input_file, save_dir)


if __name__ == '__main__':
	main()
