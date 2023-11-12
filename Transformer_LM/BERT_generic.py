
'''
This is a generic BERT training process. if you have a training data, and testing data with binary labels, you 
should be able to use this for your BERT fine tuning purpose
'''


#training the holistic model
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import pickle
import time
import random
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
from transformers import BertTokenizer
from tqdm import tqdm
import torch.nn.functional as F
import nltk
import re
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings 
warnings.filterwarnings('ignore') 


start = time.time()

def tokenization_for_BERT(df, path="/media2/special/Sadat/Brexit/", filename="put_the_filename_here", saveit="No"):

    if "tokenized" not in df.columns:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        df["tokenized"] = df["text"].apply(lambda sent:tokenizer.encode(sent, add_special_tokens=True, 
                                                                                max_length=512, truncation=True,
                                                                                padding='max_length', 
                                                                                return_attention_mask=False))

        if saveit!="No":
            df.to_pickle(path + filename)
        

    return df




# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, hidden_size=32, dropout=0): 
        #The freeze_bert is set to false to make sure our model DOES do some fine tuning
        #on the BERT layers
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, hidden_size, 2 #Just one hidden layer with 50 units in it
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # We'd like to build a fully connected neural network for classification task. We choose to keep the droput to 
        #zero for now. Later on, we will see if the dropouts can be adjusted to avoid overfitting. 

        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        '''
        This function takes input as the training set and attention mask and 
        gives the output as porbability values.
        Inputs-->
        input_ids: the training set tensor. MUST be of size [batch_size, tokenization_length]
        attention_mask: The 1/0 indication of input_ids. MUST be of size [batch_size, tokenization_length]
        output-->
        logits: Output values of shape [batch_size, number_of_labels]. Now keep it in mind, this is NOT
        softmax, it is only logits.
        '''
        # Feed input to BERT
        bert_cls_outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)[0][:, 0, :]
        

        # Feed input to classifier to compute logits
        out1 = self.fc1(bert_cls_outputs)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        logits = self.fc2(out1)
        return logits

def format_input_for_BERT_input(df):

    tokenized = np.array(list(df["tokenized"]))
    attention_masks = np.where(tokenized>0, 1, 0)
    labels = np.array(list(df["label"]))
    tokenized, attention_masks, labels = torch.tensor(tokenized), torch.tensor(attention_masks), torch.tensor(labels)
    return tokenized, attention_masks, labels


def initialize_model(epochs, train_dataloader, device, H, D_in=768, dropout=0.25, classes=2):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(hidden_size=H, dropout=dropout)
    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def create_dataloader(features, attention_masks, labels, batch_size, mode="Train"):
    # Create the DataLoader for our training set
    '''
    This function will create a dataloader for our training set. The dataloader will help to feed the randomly 
    sampled data on each batch. The batch size is selected to be 16, is simply as instructed in the original
    paper. 
    '''
    data = TensorDataset(features, attention_masks, labels)
    if mode=="Train":
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train_model(model, train_dataloader, val_dataloader, epochs, device, optimizer, scheduler):
    """Model Training
    Inputs:

    model: The BertClassifier model
    train_dataloader: will contain tokenized values, attention masks and labels in batches
                    for trianing
    val_dataloader:will contain tokenized values, attention masks and labels in batches
                    for validation
    epochs: How many epochs
    device: CPU or GPU?
    optimizer: Check the initialize_model function
    scheduler: Check the initialize_model function

    """
    loss_fn = nn.CrossEntropyLoss()
    # Start training loop
    print("...Training Process Started...\n")
    loss_record = pd.DataFrame()
    train_loss = []
    valid_loss = []
    val_acc = []
    val_f1 = []
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate((train_dataloader)):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 10 batches
            if (step % 10 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================

        # After the completion of each training epoch, measure the model's performance
        # on our validation set.
        avg_loss, acc, f1, prec, rec, prediction_df = evaluate(model, val_dataloader, device)

        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch
        
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_loss:^10.6f} | {acc:^9.2f} | {time_elapsed:^9.2f}")
        print("-"*70)

        with open(args.log_dir+ "/" + args.name_prefix + "_results.txt", 'a') as r:
            res = "\n#epoch: " + str(epoch_i) + "==>\n" +"  #train_loss " + str(round(avg_train_loss,4)) +  "\n#test_loss "  + str(round(avg_loss ,4)) \
             +  "\n#test_acc "  + str(round(acc ,4)) +  "\n#test_f1 "  + str(round(f1 ,4)) + "\n#test_rec "  + str(round(rec ,4)) + "\n#test_prec " + str(round(prec ,4))
            r.write(res)
            r.write("\n--------------------------------------------------------------------------------\n")

        print("\n")
    if args.save_model!="No":
        model_name = args.name_prefix + "_epoch_" + str(epoch_i) + ".pth"
        path = args.model_path_dir + model_name
        torch.save(model.state_dict(), path)

    if args.save_prediction=="Yes":
        prediction_df.to_csv(args.log_dir + args.name_prefix + "_Prediction.csv")


def evaluate(model, dataloader, device):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    all_probs = []
    all_labels = []

    # For each batch in our test set...
    total_loss = 0
    all_logits = []

    for batch in (dataloader):
        # Load batch to GPU
        b_input_ids, b_attn_mask, labels = tuple(t.to(device) for t in batch)
        all_labels = all_labels + (labels.tolist())
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy() 

    df = pd.DataFrame()
    df["GT"] = all_labels
    df["probs_0"] = probs[:,0]
    df["prediction"] = df["probs_0"].apply(lambda x:0 if x>0.5 else 1)
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(df["GT"], df["prediction"])
    f1 = f1_score(df["GT"], df["prediction"], average=args.metric_avg)
    prec =  precision_score(df["GT"], df["prediction"], average=args.metric_avg)
    rec = recall_score(df["GT"], df["prediction"], average=args.metric_avg)

    print("Avg loss is: ", avg_loss)
    print("acc is: ", acc)
    print("f1 is", f1)
    print("precision is", prec)
    print("recall is", rec)


    return avg_loss, acc, f1, prec, rec, df


# Is not in use. But can be, if needed
def sample_as_instructed(df, instruction="none"):
  if instruction=="Over":
    n = RandomOverSampler(random_state=42)
    dfo = n.fit_resample(df, df["Hard_lab"])[0]
    return dfo
  
  elif instruction=="Under":
    n = RandomUnderSampler(random_state=42)
    dfu = n.fit_resample(df, df["Hard_lab"])[0]
    return dfu
  else:
    return df

set_seed(42)

##
t = time.localtime()
current_time = time.strftime("%Y-%m-%d:%H:%M:%S", t)
result_filename_default_prefix = str(current_time)

parser = argparse.ArgumentParser(description='BERT model arguments')


parser.add_argument("--data_dir", 
                    type=str, 
                    default="/media2/sadat/Sadat/GenericLM/",
                     help="Input data path.")
parser.add_argument("--log_dir",
                     type=str, 
                     default="/media2/sadat/Sadat/GenericLM/",
                     help="Store result path.")
parser.add_argument("--model_path_dir",
                     type=str, 
                     default="/media2/sadat/Sadat/GenericLM/",
                     help="Store result path.")
parser.add_argument("--batch_size", type=int, default=8, help="what is the batch size?")
parser.add_argument("--device", type=str, default="cuda:0", help="what is the device?")
parser.add_argument("--dropout", type=np.float32, default=0.1, help="what is the dropout in FC layer?")
parser.add_argument("--epochs", type=int, default=4, help="what is the epoch size?")
#parser.add_argument("--classes", type=int, default=2, help="what is the number of classes?")
parser.add_argument("--hidden_size", type=int, default=32, help="what is the hidden layer size?")
#parser.add_argument("--sample", type=str, default="none", help="what kind of sampling you want? Over, Under or none")
parser.add_argument("--save_model", type=str, default="No", help="Do you want to save the model")
parser.add_argument("--fast_track", type=np.float32, default=1.00, help="Do you want a fast track train ?")
parser.add_argument("--save_prediction", type=str, default="Yes", help="Do you want to save the predictions?")
parser.add_argument("--name_prefix", type=str, default=result_filename_default_prefix, help="What is the name prefix you want to use for your models and logfile?")
parser.add_argument("--metric_avg", type=str, default="binary", help="Which metric averaing you want to use? binary, macro, micro ?")


args = parser.parse_args()





device = args.device
# first, we'll see if we have CUDA available
if torch.cuda.is_available():       
    device = torch.device(device)
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

try:
    # If you have pickle, grat ! Otherwise, csv works as well 
    train = pd.read_pickle(args.data_dir + "train.pkl")
    test = pd.read_pickle(args.data_dir + "test.pkl")
except:
    train = pd.read_csv(args.data_dir + "train.csv")
    test = pd.read_csv(args.data_dir + "test.csv")

train = tokenization_for_BERT(train, path=args.data_dir, filename="train.pkl", saveit="Yes")
test = tokenization_for_BERT(train, path=args.data_dir, filename="test.pkl", saveit="Yes")


tok_tr, mask_tr, lab_tr = format_input_for_BERT_input(train)
train_dataloader = create_dataloader(tok_tr, mask_tr, lab_tr, batch_size=args.batch_size, mode="Train")
print("\n\n >>>>  Total number of batches: {}  <<<<\n".format(len(train_dataloader)))

tok_ts, mask_ts, lab_ts = format_input_for_BERT_input(test)
test_dataloader = create_dataloader(tok_ts, mask_ts, lab_ts, batch_size=args.batch_size)

## Model Specification
with open(args.log_dir+ "/" + args.name_prefix + "_results.txt", 'a') as r:
    r.write("====Model Specification====\n")
    details = "device: " + str(args.device) + "\nDropout: " + str(args.dropout)+ "\nEpochs: " + str(args.epochs)+ "\nHidden Size: " + str(args.hidden_size) + "\n"
    r.write(details)
bert_classifier, optimizer, scheduler = initialize_model(epochs=args.epochs, train_dataloader=train_dataloader, \
device=args.device, H=args.hidden_size,  D_in=768, dropout=args.dropout, classes=args.classes)


train_model(bert_classifier, train_dataloader, test_dataloader, epochs=args.epochs, device=args.device,
        optimizer=optimizer, scheduler=scheduler)


end = time.time()

with open(args.log_dir+ "/" + args.name_prefix + "_results.txt", 'a') as r:
    timerq = str(end-start)
    r.write("################ TIME ###############\n")
    r.write(timerq)
    r.write("\n\n\n")
