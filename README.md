# RWKV-Classification
This is a project to train classification model using RWKV model from Huggingface transformers library

## Dependancies
- transformers :4.30.2
- torch >= 1.13.1+cu117
- pandas : 2.0.2
- peft : 0.3.0


## Data preparation

I prepared ChnSentiCorp_htl data for testing. It's a Chinese hotel review data which is labeled as positive and negative. 

It's a good start to test the classification model.

## Base model download

- Download the ckpt from https://huggingface.co/BlinkDL/rwkv-4-world/tree/main 
- Call the train/convert.py to convert the ckpt to Huggingface model format
- Modify the converted model config.json, change the vocab_size to 65536 and context_length to 4096

## Train the model

- Call the train.py to train the model using your own data and base model.

## Enjoy !