import random
with open('./data/target/fe2/fe2.fasta', 'r') as file:
	count = 0
	with open('./data/target/fe2/fe2_fragment.txt', 'w') as writer:
		for line in file:
			if line.startswith('>'):
				protein_name = line.strip()
				print(f'protein_name: {protein_name}')
				seq = next(file).strip()
				print(f'seq: {seq}')
				sites = next(file).strip()
				print(f'sites: {sites}')
				for position in range(len(sites)):
					site = sites[position:position+25]
					
					test = False
					if test:
						if site == '1':
							random_site = random.randint(0, 4)
							random_site = min(random_site, position)
							print(f'random_site: {random_site}')
							start = position-random_site
							end = (10-random_site)+position
							selsect_site = sites[start:end]
							print(f'selected_site: {selsect_site},{len(selsect_site)}')
							selsect_seq = seq[start:end]
							print(f'selected_seq: {selsect_seq}')
							count += 1
					else:
						if '1' in site and position+25 < len(sites):
							select = seq[position:position+25]
							print(f'select: {select}')
							print(f'site: {site}')
							print(f'length: {len(site)}')
							count = count + 1
							line = protein_name + '\n' + select + '\n' + site + '\n'
							writer.write(line)
		print(f'count: {count}')
					
			