import json
import sys

def format_json_file(input_file, output_file=None):
    
        
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  

    output_path = output_file if output_file else input_file


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)  

if __name__ == "__main__":
    
    input_json_file = "data.json"
    output_json_file = "data_formatted.json"
    
    format_json_file(input_json_file, output_json_file)