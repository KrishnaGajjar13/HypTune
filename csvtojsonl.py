import csv
import json

# Input and output file names
csv_file = '/home/gajjar/Desktop/Sem VIII/Code/Datasets/ecommerceDataset.csv'
jsonl_file = '/home/gajjar/Desktop/Sem VIII/Code/Datasets/ecommerceDataset.jsonl'

# Open the CSV and JSONL files
with open(csv_file, mode='r', encoding='utf-8') as infile, open(jsonl_file, mode='w', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    
    for row in reader:
        if len(row) < 2:
            continue  # Skip malformed rows
        label = row[0].strip()
        text = row[1].strip()
        
        json_line = {
            "text": text,
            "label": label
        }
        
        outfile.write(json.dumps(json_line, ensure_ascii=False) + '\n')

print("Conversion complete. JSONL saved to", jsonl_file)
