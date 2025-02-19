import os
print("pwd:", os.getcwd())
os.environ["HF_HOME"] = os.getcwd()

import sys
sys.path.insert(0, "./esm")

from typing import Callable
from esm.models.esm3 import ESM3
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence
from esm.sdk.api import (
    ESM3InferenceClient, 
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
    GenerationConfig
)

from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=True
)

# ----------------------------- quick test to initialize the model --------------------
# model = ESM3(
#             d_model=1536,
#             n_heads=24,
#             v_heads=256,
#             n_layers=48,
#             structure_encoder_fn=ESM3_structure_encoder_v0,
#             structure_decoder_fn=ESM3_structure_decoder_v0,
#             function_decoder_fn=ESM3_function_decoder_v0,
#             tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
#         )

# prompt = "Li_Li___HeH2CH"
# protein = ESMProtein(sequence=prompt)
# print("assembler:", protein)

# protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
# print("generate:", protein)

# protein_tensor = model.encode(protein)
# output = model.logits(protein_tensor, EMBEDDING_CONFIG)
# print("encoded protein:", getattr(output.logits, "sequence").shape)

# ----------------- main training loop --------------------------------
def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=0)
    target_batch = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=0)
    return input_batch, target_batch

def train_model(model, criterion, optimizer, dataset, epochs=10, batch_size=3, device="cpu"):
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for input_sequence, target_sequence in dataloader:
            print("input_seq:", input_sequence)
            print("target_seq:", target_sequence)
            
            # target_logits = embed_sequence(model, target_sequence)
            # print("target_logits requires_grad:", target_logits)

            tokenized_targets = []
            for smiles in target_sequence:
                protein = ESMProtein(sequence=smiles)
                print("protein target check grad:", protein)
                protein_tensor = model.encode(protein)
                print("protein_tensor target check grad:", protein_tensor.sequence.shape)
                # indices = protein_tensor.sequence
                # output = model.logits(protein_tensor, EMBEDDING_CONFIG)
                # print("target output check grad:", output.logits.sequence.shape)
                # s1, s2 = output.logits.sequence.shape[1], output.logits.sequence.shape[2]
                tokenized_targets.append(protein_tensor.sequence)

            # print("tokenized_targets:", tokenized_targets)
            # input_sequence, target_sequence = input_sequence.to(device), target_sequence.to(device)
            
            optimizer.zero_grad()

            output_sequence = generate_sequence(model, input_sequence)
            print("output seq after generation:", output_sequence)

            tokenized_output = []
            for smiles in output_sequence:
                protein = ESMProtein(sequence=smiles)
                protein_tensor = model.encode(protein)
                # indices = protein_tensor.sequence  
                output = model.logits(protein_tensor, EMBEDDING_CONFIG)
                print("seq output check grad:", output.logits.sequence.shape)
                tokenized_output.append(output.logits.sequence.squeeze(0))
            
            vocab_size = output.logits.sequence.shape[2]

            padded_target = pad_sequence(tokenized_targets, batch_first=True, padding_value=-100)
            padded_target = padded_target.to(device)
            padded_target = padded_target.view(-1)

            padded_output = pad_sequence(tokenized_output, batch_first=True, padding_value=-100)
            padded_output = padded_output.to(device)
            padded_output = padded_output.view(-1, vocab_size)

            print("Output logits requires_grad:", padded_output.shape)
            print("Target logits requires_grad:", padded_target.shape)

            loss = criterion(padded_output, padded_target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    print("Training complete.") 


model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
        )

# print("model:",  model)
# print("submodule model:", model.children())

for param in model.parameters():
    # print("param before:", param.requires_grad)
    param.requires_grad = True
    # print("param after:", param.requires_grad)

target_sequence = ["HeCCCO", "HeCCCO", "HeCCCO"]

tokenized_targets = []
for smiles in target_sequence:
    protein = ESMProtein(sequence=smiles)
    protein_tensor = model.encode(protein)
    print("before epoch protein_tensor check grad:", protein_tensor)
    # indices = protein_tensor.sequence
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    print("after epoch target output check grad:", output.logits.sequence.requires_grad)
    tokenized_targets.append(output.logits.sequence)

criterion = nn.CrossEntropyLoss(ignore_index=-100) 
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

dataset = SMILESDataset(smiles_pairs=[("HeC__O", "HeCCCO"), ("Li__CN", "LiCCCN"), ("P__LiN", "PCCLiN")], tokenizer=model.tokenizers)
train_model(model, criterion, optimizer, dataset)
