

def filtration(input_file: str, output_file: str) -> None:
	protein_context = []
	with open(input_file, 'r') as fasta_file:
		for line in fasta_file:
			if line.startswith('>'):
				name = line.strip().split("\t")[0]
				seq = next(fasta_file).strip()
				sites = next(fasta_file).strip()
				protein_line = name + "\n" + seq + "\n" + sites + "\n"
				if len(seq) < 800-2:
					protein_context.append(protein_line)
	with open(output_file, "w") as new_file:
		for protein in protein_context:
			new_file.write(protein)


def creat_fasta(input_file: str, output_file: str) -> None:
	"""
	DESCRIPTION:
		due to the original txt or fasta include the bind site,
		fail to use cd-hit, so create a new fasta to cd-hit
	:param input_file:
	:param output_file:
	:return:
	"""
	protein_context: list[str] = []
	with open(input_file, 'r') as fasta_file:
		for line in fasta_file:
			if line.startswith('>'):
				name: str = line.strip().split("\t")[0]
				seq: str = next(fasta_file).strip()
				protein_line: str = name + "\n" + seq + "\n"
				protein_context.append(protein_line)
	with open(output_file, "w") as new_file:
		for protein in protein_context:
			new_file.write(protein)


def read_bind_site(bind_site_file: str) -> dict:
	"""
	DESCRIPTION:
		get bind site from original txt or fasta
	:param bind_site_file: the txt or fasta file of protein including bind site
	:return:dict. {name:bind site}
	"""
	protein_dict = dict()
	with open(bind_site_file, 'r') as bind_site_file:
		for line in bind_site_file:
			if line.startswith('>'):
				name: str = line.strip().split("\t")[0]
				next(bind_site_file)
				bind_site: str = next(bind_site_file).strip()
				protein_dict[name] = bind_site
	return protein_dict


def read_fasta(fasta_file: str) -> dict:
	"""
	DESCRIPTION:
		get sequence
	:param fasta_file:
	:return: dict, {name:seq}
	"""
	protein_dict = dict()
	with open(fasta_file, 'r') as fasta_file:
		for line in fasta_file:
			if line.startswith('>'):
				name = line.strip().split("\t")[0]
				seq = next(fasta_file).strip()
				protein_dict[name] = seq
	return protein_dict


def create_ready_fasta(seqs: dict, bind_sites: dict, save_path: str) -> None:
	"""
	DESCRIPTION:
		add bind sites to ready cd-hit fasta
	:param seqs:
	:param bind_sites:
	:param save_path:
	:return:
	"""
	protein_line: str = ''
	for name, seq in seqs.items():
		bind_site: str = bind_sites[name]
		protein_line += name + "\n" + seq + "\n" + bind_site + "\n"
	with open(save_path, 'w') as ready_fasta:
		ready_fasta.write(protein_line)


def slice_fragment(input_file: str, focus: list[str]) -> tuple[list[str], list[str]]:
	"""
	DESCRIPTION:
		slice length==25 fragemnt of protein sequence.
		the postive fragment is the central residue is bind site and focus
	:param input_file:
	:param focus:
	:return:
	"""
	positive_fragment: list[str] = []
	negative_fragment: list[str] = []
	with open(input_file, 'r') as input_file:
		for line in input_file:
			if line.startswith('>'):
				name: str = line.strip().split("\t")[0]
				seq: str = next(input_file).strip()
				site: str = next(input_file).strip()
				for position in range(len(seq)):
					# fragment的中心残基存在于focus列表
					if seq[position] in focus:
						# 确保fragment没有超界
						if max(0, position - 12) >= 0 and max(len(seq), position + 13) <= len(seq):
							# fragment只有中心残基可以是结合位点
							if '1' not in site[position - 12:position] and '1' not in site[position + 1:position + 13]:
								fragment_label: str = site[position - 12:position + 13]
								fragment_seq: str = seq[position - 12:position + 13]
								# 窗口长度为25
								if len(fragment_seq) == 25:
									# 根据中心残基是否为结合位点判定正负样本
									if site[position] != "0":
										positive_fragment.append(name + "\n" + fragment_seq + "\n" + fragment_label + "\n")
									else:
										negative_fragment.append(name + "\n" + fragment_seq + "\n" + fragment_label + '\n')
										
	return positive_fragment, negative_fragment


def slice_train_test_by_fragment(inputs: list) -> tuple[list[str], list[str]]:
	import random
	length: int = len(inputs) // 3
	numbers: list = [3 * x for x in range(length)]
	# 训练集： 测试集 = 7： 3
	eval_length: int = len(numbers) * 3 // 10
	selected: list = random.sample(numbers, eval_length)
	print(f'{selected=}')
	test_output: list[str] = []
	for line_index in selected:
		test_output.append(inputs[line_index])
	train_output: list[str] = []
	for line_index in numbers:
		if line_index in selected:
			continue
		else:
			train_output.append(inputs[line_index])
	return train_output, test_output


def integrate_train_or_eval(positive: list[str], negative: list[str], save_dir: str) -> None:
	with open(f'{save_dir}/train.txt', 'w') as positive_file:
		for line in positive:
			positive_file.write(line)
	with open(f'{save_dir}/test.txt', 'r') as negative_file:
		for line in negative[:len(positive)]:
			negative_file.write(line)


def main() -> None:
	operation: str = 'focus'
	file_path: str = './data/target'
	target: str = 'ca'
	residue_path: str = f'{file_path}/{target}'
	if operation == 'filtration':
		input_file: str = f'{residue_path}/length/allsulfate.txt'
		output_file: str = f'{residue_path}/length/all.txt'
		filtration(input_file, output_file)
	if operation == 'focus':
		input_file: str = f'{residue_path}/fe2.fasta'
		positive_fragment: list[str]
		negative_fragment: list[str]
		positive_fragment, negative_fragment = slice_fragment(input_file, focus=["C", "D", "E", "G", "H", "K", "N", "R", "S"])
		positive_train, positive_test = slice_train_test_by_fragment(positive_fragment)
		negative_train, negative_test = slice_train_test_by_fragment(negative_fragment)
		train_output: list[str] = positive_train + negative_train
		test_output: list[str] = positive_test + negative_test
		save_dir: str = f'{residue_path}/fragment'
		integrate_train_or_eval(train_output, test_output, save_dir)
		
		
if __name__ == '__main__':
	main()
