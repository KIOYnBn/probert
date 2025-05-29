
def irap(seq, irap_list: list) -> str:
	irap_seq: str = ''
	for res in seq:
		for irap_type in irap_list:
			if res in irap_type:
				irap_seq += irap_type[0]
	return irap_seq


def main(fasta, irap_type: str) -> None:
	irap_list: list = irap_type.split("\t")[-1].split("-")
	with open(fasta, "r") as f:
		for line in f:
			if line.startswith(">"):
				seq = next(f).strip()
				output_seq = irap(seq, irap_list)
				print(output_seq)
				break


if __name__ == '__main__':
	irap_str = "LVIMCAGSTPFYW-EDNQKRH"
	fasta_file: str = './data/target/fe2/fe2.fasta'
	main(fasta_file, irap_str)
	