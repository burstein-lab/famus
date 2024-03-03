from tqdm import tqdm

path = "tmp/2023_11_17_17_36_30_997954/full_hmmsearch.txt"
c = 0
with open(path, "r") as input_handle, open("process_result.tsv", "w+") as output_handle:
    while line := input_handle.readline():
        c += 1
        if c % 1000 == 0:
            print(c)
        if line and not line.startswith("#"):
            line = line.strip("\n").split(" ")
            line = [token for token in line if token]
            non_desc_tokens = line[:22]
            desc_tokens = line[22:]
            description = " ".join(desc_tokens)
            non_desc_tokens.append(description)
            output_handle.write("\t".join(non_desc_tokens) + "\n")
