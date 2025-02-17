import random
import torch
from torch.utils.data import Dataset

class MaskedProteinDataset(Dataset):
    def __init__(self, sequences, tokenizer, mask_prob=0.15, max_length=512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        encoded = self.tokenizer(seq, 
                                 padding="max_length", 
                                 truncation=True, 
                                 max_length=self.max_length, 
                                 return_tensors="pt")

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels (same as input_ids)
        labels = input_ids.clone()

        # Apply random masking (15% probability)
        mask_token_id = self.tokenizer.mask_token_id
        for i in range(len(input_ids)):
            if random.random() < self.mask_prob:
                input_ids[i] = mask_token_id  # Replace token with [MASK]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
