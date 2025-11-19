import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

# Set torch home directory
os.environ["TORCH_HOME"] = "./torch"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BERT Fine-tuning for IMDB Sentiment Analysis')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/home/sliufo/ML/HA6_BERT/IMDB Dataset.csv',
                        help='Path to the IMDB dataset CSV file')
    parser.add_argument('--max_seq_len', type=int, default=80,
                        help='Maximum sequence length for BERT input')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Proportion of dataset to use for validation and test (will be split in half)')
    parser.add_argument('--random_state', type=int, default=2018,
                        help='Random state for reproducibility')
    
    # Model parameters
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='Pre-trained BERT model name')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for the first FC layer')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--freeze_bert', action='store_true', default=True,
                        help='Freeze BERT parameters during training')
    parser.add_argument('--unfreeze_bert', action='store_false', dest='freeze_bert',
                        help='Unfreeze BERT parameters during training')
    parser.add_argument('--unfreeze_layers', type=int, default=0,
                        help='Number of last encoder layers to unfreeze (0=freeze all BERT layers, -1=unfreeze all)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced dataset')
    parser.add_argument('--no_class_weights', action='store_false', dest='use_class_weights',
                        help='Do not use class weights')
    
    # Output parameters
    parser.add_argument('--save_path', type=str, default='saved_weights.pt',
                        help='Path to save the best model')
    parser.add_argument('--print_every', type=int, default=50,
                        help='Print progress every N batches')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cuda, or cpu')
    
    args = parser.parse_args()
    return args


# Define BERT Architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert, hidden_dim=512, dropout=0.1):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768, hidden_dim)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(hidden_dim, 2)  # positive vs negative (2 LABELS)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        outputs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(outputs.pooler_output)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x


# Training function
def train(model, train_dataloader, optimizer, cross_entropy, device, max_grad_norm=1.0, print_every=50):
    model.train()
    total_loss = 0
    total_preds = []
    total_labels = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        # progress update after every N batches
        if step % print_every == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the gradients to prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)
        total_labels.append(labels.detach().cpu().numpy())

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    
    # compute accuracy
    pred_labels = np.argmax(total_preds, axis=1)
    accuracy = np.sum(pred_labels == total_labels) / len(total_labels)

    # returns the loss, predictions, and accuracy
    return avg_loss, total_preds, accuracy


# Evaluation function
def evaluate(model, val_dataloader, cross_entropy, device, print_every=50):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()
    total_loss = 0
    total_preds = []
    total_labels = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):
        # Progress update every N batches
        if step % print_every == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
            total_labels.append(labels.detach().cpu().numpy())

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    
    # compute accuracy
    pred_labels = np.argmax(total_preds, axis=1)
    accuracy = np.sum(pred_labels == total_labels) / len(total_labels)

    return avg_loss, total_preds, accuracy


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    print("\n" + "="*50)
    print("Hyperparameters:")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    print("\n" + "="*50)
    print("Loading IMDB Dataset...")
    print("="*50)
    
    # Load Dataset
    df = pd.read_csv(args.data_path)
    print(f"Number of sentences = {df.shape[0]}")
    
    # Convert sentiment labels to numeric
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    print(f"Label positive Num: {(df['sentiment']=='positive').sum()}")
    print(f"Label negative Num: {(df['sentiment']=='negative').sum()}")
    print("\nClass distribution:")
    print(df['sentiment'].value_counts(normalize=True))
    
    # Split dataset into train, validation and test sets
    print("\n" + "="*50)
    print("Splitting dataset...")
    print("="*50)
    
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        df['review'], df['label'],
        random_state=args.random_state,
        test_size=args.test_size,
        stratify=df['label']
    )
    
    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels,
        random_state=args.random_state,
        test_size=0.5,
        stratify=temp_labels
    )
    
    print(f"Train size: {len(train_text)}")
    print(f"Validation size: {len(val_text)}")
    print(f"Test size: {len(test_text)}")
    
    # Import BERT Model and Tokenizer
    print("\n" + "="*50)
    print("Loading BERT model and tokenizer...")
    print("="*50)
    
    bert = AutoModel.from_pretrained(args.bert_model)
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model)
    
    # Tokenization
    print("\n" + "="*50)
    print("Tokenizing sequences...")
    print("="*50)
    
    max_seq_len = args.max_seq_len
    
    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )
    
    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )
    
    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )
    
    # Convert to tensors
    print("Converting to tensors...")
    
    # for train set
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())
    
    # for validation set
    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())
    
    # for test set
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())
    
    # Create DataLoaders
    print("Creating DataLoaders...")
    
    # wrap tensors for train set
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    # wrap tensors for validation set
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)
    
    # Freeze BERT parameters
    print("\n" + "="*50)
    if args.freeze_bert:
        if args.unfreeze_layers == -1:
            print("All BERT parameters will be fine-tuned...")
        elif args.unfreeze_layers == 0:
            print("Freezing all BERT parameters...")
            for param in bert.parameters():
                param.requires_grad = False
        else:
            print(f"Freezing BERT parameters except last {args.unfreeze_layers} encoder layer(s)...")
            # First freeze all parameters
            for param in bert.parameters():
                param.requires_grad = False
            
            # Then unfreeze the last N encoder layers
            total_layers = 12  # BERT-base has 12 encoder layers
            layers_to_unfreeze = list(range(total_layers - args.unfreeze_layers, total_layers))
            
            for name, param in bert.named_parameters():
                for layer_num in layers_to_unfreeze:
                    if name.startswith(f"encoder.layer.{layer_num}"):
                        param.requires_grad = True
                        print(f"  Unfreezing: {name}")
                        break
    else:
        print("All BERT parameters will be fine-tuned...")
    print("="*50)
    
    # Create model
    model = BERT_Arch(bert, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Compute class weights
    print("\n" + "="*50)
    print("Computing class weights...")
    print("="*50)
    
    if args.use_class_weights:
        class_wts = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        print(f"Class weights: {class_wts}")
        
        # Convert class weights to tensor
        weights = torch.tensor(class_wts, dtype=torch.float)
        weights = weights.to(device)
        
        # Define loss function
        cross_entropy = nn.NLLLoss(weight=weights)
    else:
        print("Not using class weights")
        cross_entropy = nn.NLLLoss()
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1} / {args.epochs}')
        print('-' * 50)
        
        # Train model
        train_loss, _, train_acc = train(model, train_dataloader, optimizer, cross_entropy, device, 
                                        max_grad_norm=args.max_grad_norm, print_every=args.print_every)
        
        # Evaluate model
        valid_loss, _, valid_acc = evaluate(model, val_dataloader, cross_entropy, device, 
                                           print_every=args.print_every)
        
        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"SAVING MODEL to {args.save_path}")
            torch.save(model.state_dict(), args.save_path)
        
        # Append training and validation metrics
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
        
        print(f'\nTraining Loss: {train_loss:.3f} | Training Accuracy: {train_acc:.3f}')
        print(f'Validation Loss: {valid_loss:.3f} | Validation Accuracy: {valid_acc:.3f}')
    
    # Load best model for testing
    print("\n" + "="*50)
    print("Loading best model and testing...")
    print("="*50)
    
    model.load_state_dict(torch.load(args.save_path))
    
    # Get predictions for test data
    with torch.no_grad():
        model.eval()
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
    
    # Convert predictions to class labels
    preds = np.argmax(preds, axis=1)
    
    # Print classification report
    print("\n" + "="*50)
    print("Test Set Performance:")
    print("="*50)
    print(classification_report(test_y, preds, target_names=['negative', 'positive']))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(pd.crosstab(test_y, preds, rownames=['Actual'], colnames=['Predicted']))
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


if __name__ == "__main__":
    main()