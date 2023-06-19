import os
import argparse
import pandas as pd
from models import RWKV_TOKENIZER
import matplotlib.pyplot as plt
import random

def cal_str_len(row,tokenizer):
    str_input = row['review']
    if str_input is None or not isinstance(str_input,str) or str_input == '':
        return 0
    inputs = tokenizer.encode(str_input)
    return len(inputs)

def analyze_corpus(corpus_file,tokenizer_file):
    df = pd.read_csv(corpus_file,encoding='utf-8')
    print(df.head())
    tokenizer = RWKV_TOKENIZER(tokenizer_file)
    labels = df['label'].unique()
    df['length'] = df.apply(cal_str_len,axis=1,args=(tokenizer,))
    hist = df.plot.hist(column='length',by='label',bins=3)
    plt.show()

def split_corpus(corpus_file,output_dir,train_ratio,tokenizer_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.basename(corpus_file)
    train_file = os.path.join(output_dir,base_name.replace('.csv','_train.csv'))
    test_file = os.path.join(output_dir,base_name.replace('.csv','_test.csv'))
    df = pd.read_csv(corpus_file,encoding='utf-8')
    print(df.head())
    tokenizer = RWKV_TOKENIZER(tokenizer_file)
    labels = df['label'].unique()
    df['length'] = df.apply(cal_str_len,axis=1,args=(tokenizer,))
    #seperate data by label
    train_data = pd.DataFrame(columns=['label','review'])
    test_data = pd.DataFrame(columns=['label','review'])
    df_by_labels = [df[df['label']==label] for label in labels]
    for df_by_label in df_by_labels:
        for index,row in df_by_label.iterrows():
            if row['length'] < 1:
                continue
            is_train = random.random() < train_ratio
            if is_train:
                train_data = train_data._append(row)
            else:
                test_data = test_data._append(row)
    
    print(f'train_data:{train_data.shape}')
    print(f'test_data:{test_data.shape}')
    
    train_data.to_csv(train_file,encoding='utf-8',index=False)
    test_data.to_csv(test_file,encoding='utf-8',index=False)
    


    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['analyze_corpus', 'split_corpus'],default='split_corpus')
    parser.add_argument('--corpus_file', type=str, default='data/ChnSentiCorp_htl_all.csv')
    parser.add_argument('--tokenizer_file', type=str, default='data/rwkv_vocab_v20230424.txt')
    parser.add_argument('--output_dir', type=str, default='data/')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()

    task = args.task
    tokenizer_file = args.tokenizer_file
    
    if task == 'analyze_corpus':
        corpus_file = args.corpus_file
        analyze_corpus(corpus_file,tokenizer_file)
    elif task == 'split_corpus':
        output_dir = args.output_dir
        corpus_file = args.corpus_file
        train_ratio = args.train_ratio
        split_corpus(corpus_file,output_dir,train_ratio,tokenizer_file)
if __name__ == '__main__':
    main()