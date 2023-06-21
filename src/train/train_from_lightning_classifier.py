import os
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from models import RwkvModelForSequenceClassification
from peft import LoraConfig, get_peft_model, AdaLoraConfig
import argparse
from transformers import AutoTokenizer, DefaultDataCollator
import datasets
from torch.utils.data import DataLoader
import torch
torch.set_float32_matmul_precision('high')
from lightning.pytorch.callbacks import EarlyStopping,StochasticWeightAveraging
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning.pytorch.strategies import DeepSpeedStrategy
import transformers

class PeftModelForSequenceClassificationLightning(pl.LightningModule):
    def __init__(self,peft_model,is_variable_len = False):
        super().__init__()
        self.peft_model = peft_model
        self.is_variable_len = is_variable_len
    

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        output = self.peft_model.forward(input_ids,labels=labels,return_dict=False)
        loss = output[0]
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        output = self.peft_model.forward(input_ids,labels=labels,return_dict=False)
        loss = output[0]
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        # return optimizer
        # return optim.AdamW(self.trainer.model.parameters(), lr=1e-5)
        return DeepSpeedCPUAdam(self.parameters(), lr=1e-5)
        # return FusedAdam(self.parameters())

def main():
    print('train from lightning')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/ChnSentiCorp_htl_all_train.csv')
    parser.add_argument('--test_file', type=str, default='data/ChnSentiCorp_htl_all_test.csv')
    parser.add_argument('--model_path', type=str, default='/media/yueyulin/KINGSTON/pretrained_models/rwkv/raven-0.4b-world')
    parser.add_argument('--tokenizer_file', type=str, default='data/rwkv_vocab_v20230424.txt')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--max_epoches', type=int, default=30)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--strategy',type=str,default='deepspeed_stage_2_offload')
    parser.add_argument('--is_adalora',action='store_true',default=False)
    parser.add_argument('--is_qlora',action='store_true',default=False)
    parser.add_argument('--output_dir',type=str,default='output_dir')

    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file
    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    num_devices = args.num_devices
    max_length = args.max_length
    max_epoches = args.max_epoches
    accumulate_grad_batches = args.accumulate_grad_batches
    ckpt_path = args.ckpt_path
    strategy = args.strategy
    is_world = 'world' in model_path
    is_adalora = args.is_adalora
    is_q_lora = args.is_qlora
    output_dir = args.output_dir
    num_classes = args.num_classes
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
        inputs_ids = [tokenizer.encode(d) for d in example["news"]]
        #pad the inputs_ids with pad_token_id to max_length or truncate the inputs_ids to max_length
        inputs_ids = [ids + [pad_token_id] * (max_length - len(ids)) if len(ids) < max_length else ids[:max_length] for ids in inputs_ids]
        labels = example['label'].copy()
        example['input_ids']=inputs_ids
        example['labels']=labels
        return example
    
    def tokenization(example):
        inputs_ids = tokenizer(example["news"],return_tensors = 'pt', padding='max_length',max_length=max_length,truncation=True)['input_ids']
        labels = example['label'].copy()
        example['input_ids']=inputs_ids
        example['labels']=labels
        return example
    if is_q_lora:
        model = RwkvModelForSequenceClassification.from_pretrained(model_path,load_in_4bit=True, device_map="auto",num_labels=num_classes,pad_token_id=pad_token_id)
    else:
        model = RwkvModelForSequenceClassification.from_pretrained(model_path, num_labels=num_classes,pad_token_id=pad_token_id)

    if is_adalora:
        peft_config = AdaLoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,target_modules=["key","value"])
    else:
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,target_modules=["key","value"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if not is_q_lora:
        model = PeftModelForSequenceClassificationLightning(model)
    print('------------')
    ds = datasets.load_dataset('csv', data_files={"train":train_file,"test":test_file})
    print(ds)
    ds = ds.map(tokenization_rwkv if is_world else tokenization,remove_columns=['news','length','label'],batched=True)
    train_ds = ds['train']
    test_ds = ds['test']
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=1,collate_fn=DefaultDataCollator(return_tensors='pt'),shuffle=True)
    print(train_dataloader)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=1,collate_fn=DefaultDataCollator(return_tensors='pt'))
    print(test_dataloader)
    if not is_q_lora:
        callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
        trainer = pl.Trainer( max_epochs=max_epoches,accelerator=device,
                            strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        ),
                            devices=num_devices,accumulate_grad_batches=accumulate_grad_batches,
                            check_val_every_n_epoch=10,precision="bf16",              
                            callbacks=callbacks)    
        trainer.fit(model=model, train_dataloaders=train_dataloader,val_dataloaders=test_dataloader,ckpt_path=ckpt_path)
    else:
        print(train_ds[0])
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-4,
            output_dir="outputs",
            fp16=False,
            optim="paged_adamw_8bit")
        trainer = transformers.Trainer(model=model,
                                    args=training_args,
                                    train_dataset=train_ds,
                                    eval_dataset=test_ds,
        )
        trainer.train()
    peft_model = model.peft_model
    # print(peft_model.peft_config)
    peft_model.save_pretrained(output_dir)

if __name__ == '__main__':
    main()