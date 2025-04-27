"""
Training pipeline for M1 Hybrid Reasoning Model
Includes: distillation, supervised fine-tuning (SFT), RL, and inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from model import M1Model
from config import MODEL_CONFIG, TRAINING_CONFIG
from datasets import load_dataset
from transformers import AutoTokenizer

class GSM8KDataset(Dataset):
    def __init__(self, split="train", tokenizer_name="bert-base-uncased", max_length=32):
        self.dataset = load_dataset("gsm8k", "main", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.inputs = []
        self.labels = []
        for item in self.dataset:
            question = item["question"]
            answer = item["answer"]
            enc = self.tokenizer(question, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            dec = self.tokenizer(answer, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            self.inputs.append(enc["input_ids"].squeeze(0))
            self.labels.append(dec["input_ids"].squeeze(0))
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def distillation_train():
    print("Starting distillation training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M1Model(**MODEL_CONFIG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    dataset = GSM8KDataset(split="train")
    loader = DataLoader(dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
    model.train()
    for epoch in range(TRAINING_CONFIG["distillation_epochs"]):
        total_loss = 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['distillation_epochs']} - Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "m1_distilled.pt")
    print("Distillation training complete. Model saved as m1_distilled.pt.")

def sft_train():
    print("Starting supervised fine-tuning (SFT)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M1Model(**MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load("m1_distilled.pt", map_location=device))
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    dataset = GSM8KDataset(split="train")
    loader = DataLoader(dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
    model.train()
    for epoch in range(TRAINING_CONFIG["sft_epochs"]):
        total_loss = 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['sft_epochs']} - Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "m1_sft.pt")
    print("SFT complete. Model saved as m1_sft.pt.")

def rl_train():
    print("Starting reinforcement learning (RL) stage...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M1Model(**MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load("m1_sft.pt", map_location=device))
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    dataset = GSM8KDataset(split="train")
    loader = DataLoader(dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
    model.train()
    for epoch in range(TRAINING_CONFIG["rl_epochs"]):
        total_loss = 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            # Policy gradient RL loss (REINFORCE):
            log_probs = torch.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            reward = (logits.argmax(-1) == labels).float()
            loss = -(chosen_log_probs * reward).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['rl_epochs']} - RL Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "m1_rl.pt")
    print("RL training complete. Model saved as m1_rl.pt.")

def run_inference(input_text, checkpoint="m1_rl.pt", tokenizer_name="bert-base-uncased", max_length=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M1Model(**MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    with torch.no_grad():
        inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        logits = model(input_ids)
        preds = logits.argmax(-1)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
    return decoded

if __name__ == "__main__":
    distillation_train()
    sft_train()
    rl_train()
    # Example inference after training
    test_question = "If you have 3 apples and you get 2 more, how many apples do you have?"
    prediction = run_inference(test_question)
    print("Inference output:", prediction)
