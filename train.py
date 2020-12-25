import pandas as pd
import numpy as np
import random
from tqdm import tqdm, tqdm_notebook

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# KoBERT
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

# etc
import gluonnlp as nlp
from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule

# model
from model.encoder import *
from model.classifier import *
from model.bertsum import *

from utils import *
from utils.data import *



def save_checkpoint(epoch, model, optimizer, scheduler, loss_list, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_list': loss_list
        }, PATH)


###### hyper parameter #####
max_sentence_num = 64
max_word_num = 64
num_workers = 5
batch_size = 2
learning_rate = 1e-5
num_epochs = 1000
embedding_vector_size = 128

config = Config({
    "n_enc_vocab": len(vocab),
    "n_dec_vocab": len(vocab),
    "n_enc_seq":embedding_vector_size,
    "n_layer":6,
    "d_hidn":embedding_vector_size,
    "i_pad":1,
    "d_ff":1024,
    "n_head":4,
    "d_head":64,
    "dropout":0.1,
    "layer_norm_epsilon":1e-12
})


if __name__ == '__main__':


    ###### setting device ######
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ####### load KoBERT ########
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    ######### load data ########
    df = pd.read_json("data/train.jsonl", lines=True)

    train_dataset = ArticleLineDataset(df[:500], "article_original", "extractive", tok, max_sentence_num, max_word_num, True, False)

    train_loader = MyDataLoader(train_dataset, batch_size=batch_size)


    ######## model init ########
    encoder = BertEncoder(bertmodel)

    reducer = DimensionReducer(768, embedding_vector_size)
    second = SecondEncoder(config=config, n_layer=1)
    classifier = BinaryClassifier(embedding_vector_size)

    model = BERTSummarizer(config, reducer, second, classifier, device)

    # bert model freezing
    for name, param in encoder.named_parameters():
        param.requires_grad = False

    # to device
    encoder = encoder.to(device)
    model = model.to(device)
        

    #### logging model info ####
    print("## model layer info ##")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print("\n## model params num ##")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameter number > {pytorch_total_params}")
    print("\n======================")


    ######## optimizer #########
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # init optimizer, loss
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = nn.BCELoss().to(device)

    ## linear scheduler 초기화
    t_total = len(train_loader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)


    ########## train ###########
    loss_list = []
    print('\n## Start training ##')
    ## checkpoint 불러와서 학습시킬 경우 range 범위 바꿔주기
    for epoch in range(0, num_epochs):

        model.train()

        for i, (batch, num, label) in enumerate(tqdm_notebook(train_loader)):

            optimizer.zero_grad()

            batch, num, label = batch.long().to(device), num.to(device), label.long().to(device)
            
            # bert embedding
            new_batch = embedding(encoder, batch, num)

            #calculate output
            output = model(new_batch, num)
            
            loss = criterion(output.long().reshape(-1,1).to(device), label.reshape(-1,1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step() 
            scheduler.step()

            ## accuracy 계산
            _, topi = outputs.permute(0,2,1).topk(1)
            topi = topi.squeeze()
            accuracy = (label == topi).float().mean()

            if (i+1) % print_every == 0: ## 상태 출력
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                    epoch+1, num_epochs, i+1, len(train_loader), loss.item(), accuracy.item() * 100))
                loss_list.append(loss.item())
                translate_example(token_ids, valid_length, outputs.permute(0,2,1), label, label_length)

            
        ## checkpoint 저장
        if (epoch+1) % save_every == 0:
        if not os.path.exists("model"):
            os.makedirs("model")
        save_checkpoint(epoch+1, model, optimizer, scheduler, loss_list, "model/checkpoint_2_"+str(epoch+1)+".tar")
        print("checkpoint saved!")













    save_checkpoint(epoch, model, optimizer, scheduler, loss_list, PATH)