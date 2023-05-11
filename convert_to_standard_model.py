import torch
import sys
import os

infile = sys.argv[1]
output_dir = sys.argv[2]

model = torch.load(infile)

# create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# save the model weights to pytorch_model.bin
weights_file = os.path.join(output_dir, 'pytorch_model.bin')
torch.save(model.state_dict(), weights_file)

# save the model configuration to config.json
config_file = os.path.join(output_dir, 'config.json')
with open(config_file, 'w') as f:
    f.write(model.config.to_json_string())

# save the vocabulary to vocab.txt (for BERT and Transformer-XL) or vocab.json (for GPT/GPT-2)
if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'save_vocabulary'):
    vocab_file = os.path.join(output_dir, 'vocab.txt' if not model.config.model_type.startswith('gpt') else 'vocab.json')
    model.tokenizer.save_vocabulary(vocab_file)

print('This script actually does not handles the files: tokenizer_config.json, tokenizer.json, vocab.txt . You should handle them manually')
