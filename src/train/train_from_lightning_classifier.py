import os
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from models import RwkvModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import argparse
from transformers import AutoTokenizer, DefaultDataCollator
import datasets
from torch.utils.data import DataLoader
import torch
# torch.set_float32_matmul_precision('high')

class PeftModelForSequenceClassificationLightning(pl.LightningModule):
    def __init__(self,peft_model,is_variable_len = False):
        super().__init__()
        self.peft_model = peft_model
        self.is_variable_len = is_variable_len
    

    def training_step(self, batch, batch_idx):
        input_ids = batch['inputs_ids']
        labels = batch['labels']
        output = self.peft_model.forward(input_ids,labels=labels,return_dict=False)
        loss = output[0]
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['inputs_ids']
        labels = batch['labels']
        output = self.peft_model.forward(input_ids,labels=labels,return_dict=False)
        loss = output[0]
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        # return optimizer
        return optim.AdamW(self.trainer.model.parameters(), lr=1e-5)
        # return DeepSpeedCPUAdam(self.parameters(), lr=1e-5)
        # return FusedAdam(self.parameters())

def main():
    print('train from lightning')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/ChnSentiCorp_htl_all_train.csv')
    parser.add_argument('--test_file', type=str, default='data/ChnSentiCorp_htl_all_test.csv')
    parser.add_argument('--model_path', type=str, default='/Volumes/TOUROS/models/rwkv/raven-0.4b-world')
    parser.add_argument('--tokenizer_file', type=str, default='data/rwkv_vocab_v20230424.txt')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=2048)

    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file
    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    num_devices = args.num_devices
    max_length = args.max_length

    is_world = 'world' in model_path
    print(args)

    if is_world:
        from models import RWKV_TOKENIZER
        tokenizer = RWKV_TOKENIZER(args.tokenizer_file)
        pad_token_id = 0
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        print(tokenizer)
        pad_token_id = tokenizer.pad_token_id

    
    
    def tokenization_rwkv(example):
        inputs_ids = [tokenizer.encode(d) for d in example["review"]]
        #pad the inputs_ids with pad_token_id to max_length or truncate the inputs_ids to max_length
        inputs_ids = [ids + [pad_token_id] * (max_length - len(ids)) if len(ids) < max_length else ids[:max_length] for ids in inputs_ids]
        labels = example['label'].copy()
        example['inputs_ids']=inputs_ids
        example['labels']=labels
        return example
    
    def tokenization(example):
        inputs_ids = tokenizer(example["review"],return_tensors = 'pt', padding='max_length',max_length=max_length,truncation=True)['input_ids']
        labels = example['label'].copy()
        example['inputs_ids']=inputs_ids
        example['labels']=labels
        return example
    model = RwkvModelForSequenceClassification.from_pretrained(model_path,num_labels=2,pad_token_id=pad_token_id)
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,target_modules=["key","value"])
    model = get_peft_model(model, peft_config)
    print(model)
    model.print_trainable_parameters()
    model = PeftModelForSequenceClassificationLightning(model)
    print('------------')
    print(model)
    ds = datasets.load_dataset('csv', data_files={"train":train_file,"test":test_file})
    print(ds)
    ds = ds.map(tokenization_rwkv if is_world else tokenization,remove_columns=['review','length'],batched=True)
    train_ds = ds['train']
    test_ds = ds['test']
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=1,collate_fn=DefaultDataCollator(return_tensors='pt'),shuffle=True)
    print(train_dataloader)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=1,collate_fn=DefaultDataCollator(return_tensors='pt'))
    print(test_dataloader)
    # ckpt = 'lightning_logs/version_0/checkpoints/epoch=29-step=510.ckpt'
    trainer = pl.Trainer( max_epochs=1,accelerator=device,devices=num_devices)
    trainer.fit(model=model, train_dataloaders=train_dataloader)
    peft_model = model.peft_model
    # print(peft_model.peft_config)
    peft_model.save_pretrained('peft_model_0.4b_world_lora/')

if __name__ == '__main__':
    main()