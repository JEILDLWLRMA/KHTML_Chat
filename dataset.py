import json
from transformers import AutoTokenizer, TextDataset

def preprocess_data(input_json_path, output_txt_path):
    # Load the JSON data
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare data for fine-tuning
    dialogues = []
    for session_id, session_data in data.items():
        conversation = session_data['conversation']
        for conv in conversation:
            dialogue = ""
            for utterance in conv['utterances']:
                speaker = "상담원" if utterance['speaker_idx'].startswith('T') else "학생"
                dialogue += f"{speaker}: {utterance['utterance']}\n"
            dialogues.append(dialogue.strip())

    # Save the prepared dialogues to a text file (one conversation per line)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for dialogue in dialogues:
            f.write(dialogue + '\n')

def load_dataset(tokenizer_name, file_path, block_size=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

if __name__ == "__main__":
    input_json = "/data/hyeokseung1208/cchat/data/consult_highschool.json"
    output_txt = "/data/hyeokseung1208/cchat/data/consult_highschool.txt"
    preprocess_data(input_json, output_txt)