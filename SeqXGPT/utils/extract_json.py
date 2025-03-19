import json

# Input and output file paths
input_file = "/Users/ruoxinwang/Desktop/Duke/Deep_Learning_and_Applications/Natural_Language_Processing/AI-Text-Detector/SeqXGPT/dataset/SeqXGPT_output/en_gpt2_tr.jsonl"
output_file = "/Users/ruoxinwang/Desktop/Duke/Deep_Learning_and_Applications/Natural_Language_Processing/AI-Text-Detector/SeqXGPT/dataset/SeqXGPT_output/sample_tr.jsonl"

# Read the first 10 elements and write to the output file
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    # Read and write the first 10 lines
    for i, line in enumerate(infile):
        if i >= 100:  # Stop after reading 10 lines
            break
        
        # Verify it's valid JSON before writing
        try:
            json_obj = json.loads(line)
            outfile.write(line)
        except json.JSONDecodeError:
            print(f"Warning: Line {i+1} contains invalid JSON. Skipping.")

print(f"Successfully extracted first 100 elements to {output_file}")