import json

# Input and output files
input_file = "/home/gajjar/Desktop/Sem VIII/Code/Datasets/Sentence pairs in English-French - 2025-05-01.tsv"   # Replace with your Tatoeba TSV file
output_file = "/home/gajjar/Desktop/Sem VIII/Code/Datasets/tatoeba_pairs.jsonl"

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            _, eng, _, fra = parts
            json_obj = {"input": eng, "output": fra}
            outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
