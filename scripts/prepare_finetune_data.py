from tokenizer import BPETokenizer
import torch

# Create tokenizer and load vocab/merge rules
tokenizer = BPETokenizer()
tokenizer.load("data/tokenizer.json")

answer_token_id = tokenizer.vocab["[ANSWER]"]

with open("data/aistotle_qa2.txt", "r") as file:
    text = file.read()

# Split the file into individual q/a pairs
examples = text.split("[QUESTION]")

all_inputs = []
all_targets = []
padded_inputs = []
padded_targets = []

context_length = 512

for example in examples:
    
    parts = example.split("[ANSWER]")
    
    # Make sure it is split into a question and answer part otherwise continue
    if len(parts) != 2:
        continue
    
    question = parts[0]
    answer = parts[1]
    
    # Convert the question and answer parts into token ids
    full_sequence = tokenizer.encode("[QUESTION]" + question + "[ANSWER]" + answer)
    
    # Make sure the sequence is within our context length
    if len(full_sequence) > context_length:
        continue
    
    # Shift inputs and targets so each position predicts the next token
    inputs = full_sequence[:-1]
    targets = full_sequence[1:]
    
    # Find where answer starts and mask everything before and including it with -100
    answer_position = full_sequence.index(answer_token_id)

    # Ignore everything before the first answer token.
    # The first token we train on should be the token after [ANSWER].
    for i in range(answer_position):
        targets[i] = -100

    all_inputs.append(inputs)
    all_targets.append(targets)

# Find the max length of the sequences in all inputs since we need we need them to be same size for our tensors
if not all_inputs:
    raise ValueError("No usable fine-tuning examples were found. Check the [QUESTION]...[ANSWER] format and context length.")

max_len = max(len(sequence) for sequence in all_inputs)

for i in range(len(all_inputs)):
    
    # Find how off the length of the current sequence is compared to the max length sequence
    padding_needed = max_len - len(all_inputs[i])
    
    # Fill in the padding spaces with 0 for inputs and -100 for targets
    input_pad = all_inputs[i] + [0] * padding_needed
    target_pad = all_targets[i] + [-100] * padding_needed
    
    padded_inputs.append(input_pad)
    padded_targets.append(target_pad)

inputs_tensor = torch.tensor(padded_inputs)
targets_tensor = torch.tensor(padded_targets)

torch.save(inputs_tensor, "data/finetune_inputs.pt")
torch.save(targets_tensor, "data/finetune_targets.pt")

print(f"Saved tensors of shape {inputs_tensor.shape}")
