import os

def filtration(input_file: str, output_file: str) -> None:
	protein_context = []
	with open(input_file, 'r') as fasta_file:
		for line in fasta_file:
			if line.startswith('>'):
				name = line.strip().split("\t")[0]
				seq = next(fasta_file).strip()
				sites = next(fasta_file).strip()
				protein_line = name + "\t" + seq + "\t" + sites + "\n"
				if len(seq) < 800 - 2:
					protein_context.append(protein_line)
	with open(output_file, "w") as new_file:
		for protein in protein_context:
			new_file.write(protein)


def slice_train_test(input_file: str,  input_dir: str) -> None:
	import random
	with open(input_file, 'r') as fasta_file:
		lines = fasta_file.readlines()
	length: int = len(lines) // 3
	numbers: list = [3 * x for x in range(length)]
	# 训练集： 测试集 = 7： 3
	eval_length: int = len(numbers) * 3 // 10
	selected: list = random.sample(numbers, eval_length)
	print(f'{selected=}')
	test_output: list[str] = []
	for line_index in selected:
		test_output.append(lines[line_index])
	train_output: list[str] = []
	for line_index in numbers:
		if line_index in selected:
			continue
		else:
			train_output.append(lines[line_index])
	with open(f'{input_dir}/train.csv', 'w') as train_file:
		train_file.write(''.join(train_output))
	with open(f'{input_dir}/test.csv', 'w') as test_file:
		test_file.write(''.join(test_output))


def main() -> None:
	operation: str = 'reduce'
	file_path: str = './data/target'
	target: str = 'na'
	residue_path: str = f'{file_path}/{target}'
	if operation == 'filtration':
		filtration_path = f'{residue_path}/sites'
		input_file: str = f'{filtration_path}/allsulfate.txt'
		output_file: str = f'{filtration_path}/all.csv'
		filtration(input_file, output_file)
		slice_train_test(output_file, filtration_path)
	if operation == 'reduce':
		reduce_path = f'{file_path}/{target}/reduce'
		all_type: list = os.listdir(reduce_path)
		for reduce_type in all_type:
			file_path: str = f'{reduce_path}/{reduce_type}'
			name: str = reduce_type.split('.')[0]
			name_path: str = f'{reduce_path}/{name}'
			os.makedirs(name_path, exist_ok=True)
			input_file: str = file_path
			output_file: str = f"{name_path}/all.csv"
			filtration(input_file, output_file)
			slice_train_test(output_file, name_path)
			

if __name__ == '__main__':
	main()
