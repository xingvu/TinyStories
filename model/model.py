
import sys
import torch
import argparse

from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

########################################################
# 0. parse args
########################################################

parser = argparse.ArgumentParser()

parser.add_argument('--model-folder', 
                    type=str,
                    default="/embedding/v-xingwuchen/ts_data/TinyStories/tmp_models")
parser.add_argument('--model-name', 
                    type=str,
                    default="TinyStories")
parser.add_argument('--vocab-size', 
                    type=int,
                    default=50257,
                    help='Vocabulary size of the GPT-2 model')
parser.add_argument('--n-embd', 
                    type=int,
                    default=None,
                    help='Hidden size of the transformer embeddings')
parser.add_argument('--n-layer', 
                    type=int,
                    default=None,
                    help='Number of transformer layers')
parser.add_argument('--n-head', 
                    type=int,
                    default=None,
                    help='Number of attention heads')
parser.add_argument('--n-positions', 
                    type=int,
                    default=2048,
                    help='Maximum sequence length')
args = parser.parse_args()

########################################################
# 1. initialize models and dataset
########################################################

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')



config = GPT2Config(
    vocab_size=args.vocab_size,
    n_embd=args.n_embd,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_positions=args.n_positions,
)

# creating the model
model = GPT2LMHeadModel(config)

print(model)

device = "cuda"
model.to(device)

dataset = load_from_disk("/embedding/v-xingwuchen/ts_data/TinyStories/dataset/TinyStories")
train_loader = DataLoader(dataset['train'], batch_size=16, shuffle=True)
valid_loader = DataLoader(dataset['validation'], batch_size=24, shuffle=True)

########################################################
# 2. train from scarch
########################################################

optim = torch.optim.Adam(model.parameters(), lr=1e-3,betas=(0.9, 0.95)) # default lr ,betas and eps

tokenizer.pad_token = tokenizer.eos_token
best_loss = 10000000
updates = 0
for epoch in range(1):
    print(f"epoch : {epoch}")
    print(f"{'*'*45}-train-{epoch:02}-{'*'*45}")
    for batch in tqdm(train_loader):
        optim.zero_grad()
        tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt',max_length = 512,truncation = True)['input_ids'].to(device)
        loss = model(tokenized,labels = tokenized)["loss"]
        loss.backward()
        optim.step()
        updates += 1
        # print(f"train-{epoch:02}-{updates} : {loss.item()}")
        if updates % 50 == 0 :
            sys.stdout.flush()
            tqdm.write(f"train-{epoch+1:02}-{updates} : {loss.item()}")
            # print(f"train-{epoch:02}-{updates} : {loss.item()}")
        del tokenized,loss
        torch.cuda.empty_cache()
    print("*"*100)
    with torch.no_grad():
        print(f"{'-'*45}-valid-{epoch+1:02}-{'-'*45}")
        loss_valid = 0
        for batch in tqdm(valid_loader):
            tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt',max_length = 512,truncation = True)['input_ids'].to(device)
            loss_valid += model(tokenized,labels = tokenized)["loss"].item()
            del tokenized
            torch.cuda.empty_cache()
        print(f"train-{epoch+1:02}- : {loss_valid}")
        print("-"*100)
        if best_loss > loss_valid or epoch == 0:
            model.save_pretrained(f"{args.model_folder}/{args.model_name}_best")
            best_loss = loss_valid
