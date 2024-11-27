import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm
from peft import LoraConfig, PeftModel, TaskType
import json
import pandas as pd

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

NUM_NEG_PER_SAMPLE = 1  # 每个正样本的负样本数量


class LinkPredHead(nn.Module):
    def __init__(self, config):
        super(LinkPredHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        dropout = config.header_dropout_prob if config.header_dropout_prob is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x_i, x_j):
        x = x_i * x_j
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return torch.sigmoid(x)

# 原link_lm_trainerpy：Dataset class
class TextPairDataset(Dataset):
    def __init__(
        self,
        text1_list: List[str],
        text2_list: List[str], 
        labels: Optional[List[int]] = None,
        tokenizer_name: str = "intfloat/e5-large",
        max_length: int = 512 #原512
    ):
        assert len(text1_list) == len(text2_list)
        self.text1_list = text1_list
        self.text2_list = text2_list
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.num_samples = len(text1_list)
        
        logger.info("Tokenizing all texts...")
        self.encodings1 = self.tokenizer(
            text1_list,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.encodings2 = self.tokenizer(
            text2_list,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        if self.labels is None:  
            return {
                'input_ids1': self.encodings1['input_ids'][idx],
                'attention_mask1': self.encodings1['attention_mask'][idx],
                'input_ids2': self.encodings2['input_ids'][idx],
                'attention_mask2': self.encodings2['attention_mask'][idx]
            }
        
        src_idx = idx
        pos_idx = idx
        neg_indices = torch.randint(self.num_samples, (NUM_NEG_PER_SAMPLE,))
        
        input_ids1 = self.encodings1['input_ids'][src_idx]
        attention_mask1 = self.encodings1['attention_mask'][src_idx]
        
        # 组合正样本和负样本
        pos_input_ids2 = self.encodings2['input_ids'][pos_idx]
        pos_attention_mask2 = self.encodings2['attention_mask'][pos_idx]
        neg_input_ids2 = self.encodings2['input_ids'][neg_indices]
        neg_attention_mask2 = self.encodings2['attention_mask'][neg_indices]
        
        input_ids2 = torch.cat([pos_input_ids2.unsqueeze(0), neg_input_ids2])
        attention_mask2 = torch.cat([pos_attention_mask2.unsqueeze(0), neg_attention_mask2])
        
        return {
            'input_ids1': input_ids1.repeat(NUM_NEG_PER_SAMPLE + 1, 1),
            'attention_mask1': attention_mask1.repeat(NUM_NEG_PER_SAMPLE + 1, 1),
            'input_ids2': input_ids2,
            'attention_mask2': attention_mask2,
            'label': torch.tensor([1] + [0] * NUM_NEG_PER_SAMPLE, dtype=torch.float)
        }

# 原 link_lm_modeling.py : Link_E5_model class
class E5LinkModel(nn.Module):
    def __init__(
        self, 
        model_name: str = "intfloat/e5-large",
        use_peft: bool = True,
        peft_r: int = 8,
        peft_lora_alpha: int = 32,
        peft_lora_dropout: float = 0.1,
        header_dropout_prob: float = 0.1
    ):
        super().__init__()
       
        config = AutoConfig.from_pretrained(model_name)
        config.header_dropout_prob = header_dropout_prob
        
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = self.encoder.config.hidden_size
        
        # LoRA
        if use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=peft_r,
                lora_alpha=peft_lora_alpha,
                lora_dropout=peft_lora_dropout,
            )
            self.encoder = PeftModel(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        
        # Link prediction head
        self.link_head = LinkPredHead(config)
        

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_text(self, input_ids, attention_mask):
        if input_ids.dim() > 2:
            batch_size, num_samples, seq_len = input_ids.size()
            input_ids = input_ids.view(-1, seq_len)
            attention_mask = attention_mask.view(-1, seq_len)
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        
        emb1 = self.encode_text(input_ids1, attention_mask1)
        emb2 = self.encode_text(input_ids2, attention_mask2)
        

        scores = self.link_head(emb1, emb2)
        return scores.squeeze(-1), emb1, emb2

def train_model(
    train_dataset: TextPairDataset,
    val_dataset: Optional[TextPairDataset] = None,
    model_name: str = "intfloat/e5-large",
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    output_dir: str = "outputs",
    use_peft: bool = True,
    peft_r: int = 8,
    peft_lora_alpha: int = 32,
    peft_lora_dropout: float = 0.1,
    header_dropout_prob: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    
    model = E5LinkModel(
        model_name=model_name,
        use_peft=use_peft,
        peft_r=peft_r,
        peft_lora_alpha=peft_lora_alpha,
        peft_lora_dropout=peft_lora_dropout,
        header_dropout_prob=header_dropout_prob
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            scores, _, _ = model(
                batch['input_ids1'],
                batch['attention_mask1'],
                batch['input_ids2'],
                batch['attention_mask2']
            )
            
            labels = batch['label'].view(-1) 
            scores = scores.view(-1) 
            
            loss = criterion(scores, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # validation
        if val_dataset:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    scores, _, _ = model(
                        batch['input_ids1'],
                        batch['attention_mask1'],
                        batch['input_ids2'],
                        batch['attention_mask2']
                    )
                    preds = (scores > 0).float()
                    correct += (preds == batch['label']).sum().item()
                    total += len(batch['label'])
            
            val_acc = correct / total
            logger.info(f"Validation Accuracy: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
    
    return model

def generate_embeddings(model: E5LinkModel, dataset: TextPairDataset, device: str):
    model.eval()
    loader = DataLoader(dataset, batch_size=8)
    all_emb1, all_emb2 = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            input_ids1 = batch['input_ids1']
            attention_mask1 = batch['attention_mask1']
            input_ids2 = batch['input_ids2']
            attention_mask2 = batch['attention_mask2']
            
            if len(input_ids1.shape) == 3:
                input_ids1 = input_ids1.reshape(-1, input_ids1.size(-1))
                attention_mask1 = attention_mask1.reshape(-1, attention_mask1.size(-1))
            if len(input_ids2.shape) == 3:
                input_ids2 = input_ids2.reshape(-1, input_ids2.size(-1))
                attention_mask2 = attention_mask2.reshape(-1, attention_mask2.size(-1))
            
            _, emb1, emb2 = model(
                input_ids1,
                attention_mask1,
                input_ids2,
                attention_mask2
            )
            all_emb1.append(emb1.cpu())
            all_emb2.append(emb2.cpu())
    
    embeddings1 = torch.cat(all_emb1)
    embeddings2 = torch.cat(all_emb2)
    
    embeddings1 = embeddings1[::2]  # 每隔一个取样，只保留正样本
    embeddings2 = embeddings2[::2]
    
    print(f"Generated embeddings shapes: {embeddings1.shape}, {embeddings2.shape}")
    return embeddings1, embeddings2

def save_positive_embeddings(embeddings, save_path):

    # 每隔一个样本取一次,只取正样本
    positive_embeddings = embeddings[::2]  
    
    # 保存正样本嵌入
    torch.save(positive_embeddings, save_path)
    
    print(f"Original embeddings shape: {embeddings.shape}")
    print(f"Positive embeddings shape: {positive_embeddings.shape}")
    
    return positive_embeddings

def save_embeddings(embeddings1, embeddings2, drug_ids, target_ids, save_path):

    # 处理drug的去重
    unique_drug_dict = {}  # {drug_id: embedding}
    for idx, drug_id in enumerate(drug_ids):
        if drug_id not in unique_drug_dict:
            unique_drug_dict[drug_id] = embeddings1[idx]
        else:
            unique_drug_dict[drug_id] = (unique_drug_dict[drug_id] + embeddings1[idx]) / 2
    
    # 处理target的去重
    unique_target_dict = {}  # {target_id: embedding}
    for idx, target_id in enumerate(target_ids):
        if target_id not in unique_target_dict:
            unique_target_dict[target_id] = embeddings2[idx]
        else:
            unique_target_dict[target_id] = (unique_target_dict[target_id] + embeddings2[idx]) / 2
    
    unique_drug_ids = list(unique_drug_dict.keys())
    unique_target_ids = list(unique_target_dict.keys())
    unique_drug_embeddings = torch.stack([unique_drug_dict[id_] for id_ in unique_drug_ids])
    unique_target_embeddings = torch.stack([unique_target_dict[id_] for id_ in unique_target_ids])
    
    embeddings_dict = {
        'drug': {
            'embeddings': unique_drug_embeddings.cpu().numpy(),
            'ids': unique_drug_ids
        },
        'target': {
            'embeddings': unique_target_embeddings.cpu().numpy(),
            'ids': unique_target_ids
        }
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings_dict, save_path)
    
    print("\nAfter deduplication:")
    print(f"Original drug embeddings: {embeddings1.shape} -> Unique: {unique_drug_embeddings.shape}")
    print(f"Original target embeddings: {embeddings2.shape} -> Unique: {unique_target_embeddings.shape}")

    
def main():

    with open('paired_data.json', 'r', encoding='utf-8') as f:
        paired_data = json.load(f)
    
    logger.info("\nProcessing  pairs from train dataset...")

    train_data = paired_data['train']  
    drug_texts = [item['drug_text'] for item in train_data]
    target_texts = [item['target_text'] for item in train_data]
    drug_ids = [item['drug_id'] for item in train_data]
    target_ids = [item['target_id'] for item in train_data]
    labels = [1] * len(train_data) 

    dataset = TextPairDataset(
        text1_list=drug_texts,
        text2_list=target_texts,
        labels=labels
    )

    model = train_model(
        train_dataset=dataset,
        batch_size=16,
        num_epochs=15
    )

    embeddings1, embeddings2 = generate_embeddings(
        model=model,
        dataset=dataset,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    save_path = 'embeddings/train_embeddings.pt'
    save_embeddings(embeddings1, embeddings2, drug_ids, target_ids, save_path)
    
    # 打印统计信息
    logger.info("Train dataset statistics:")
    logger.info(f"Number of pairs: {len(train_data)}")
    
    logger.info("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
