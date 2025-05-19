import os
import re
import math

dir_path: str = '../results/reduced'
all_files: list = os.listdir(dir_path)
terminal_output: list = []
for file_name in all_files:
	file_path: str = f'{dir_path}/{file_name}/metrics.txt'
	auc: float = 0
	precision: float = 0
	recall: float = 0
	accuracy: float = 0
	
	with open(file_path, 'r') as f:
		lines: list = f.readlines()
		for line in lines[1:]:
			line = list(map(float, line.strip().split('\t')))

			auc += line[0]
			precision += line[1]
			recall += line[2]
			accuracy += line[3]
	auc /= len(lines)-1
	precision /= len(lines)-1
	recall /= len(lines)-1
	accuracy /= len(lines)-1
	if auc > 0.7:
		print(f'{auc:.4f}, {precision:.4f}, {recall:.4f}, {accuracy:.4f}')
	
	# terminal_output.append([auc, precision, recall, accuracy])
	
print(f'{sorted(terminal_output)}')
	
