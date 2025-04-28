import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import json
import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, DistilBertConfig, DistilBertForSequenceClassification

label_map = {'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4}
reverse_map = {0:'business', 1:'entertainment', 2:'politics', 3:'sport', 4:'tech'}
# Load model in half precision
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom config (5 output classes + FP16 if CUDA)

class NewDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = data.reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]['Text'])  # Ensure string type
        label = int(self.data.iloc[index]['Category'])  # Ensure integer
        return {'text': text, 'label': label}

def data_loader(data,batch_size=32):
    """
    Load the data and create DataLoader objects for training and validation.
    """
    # Load your dataset here
    train = data.sample(frac=0.8, random_state=42)
    val = data.drop(train.index)
    train_dataset = NewDataset(train)
    val_dataset = NewDataset(val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_model(model,data,lr,batch_size,num_epoch):
    os.makedirs('metrics', exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = data_loader(data=data,batch_size=batch_size)
    total_cnt = 0
    correct_cnt = 0
    metrics_history,best_labels,best_preds = [],[],[]
    for epoch in range(num_epoch):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            labels = torch.tensor(batch['label'].tolist()).to(device)
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels = labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            correct_cnt += (torch.argmax(outputs.logits, dim=1) == labels).sum().item()
            total_cnt += labels.size(0)
        print(f"Epoch {epoch+1}/{num_epoch}, Loss: {loss.item()}, accuracy: {correct_cnt / total_cnt}")
        with torch.no_grad():
            model.eval()
            val_loss = 0
            total_cnt = 0
            correct_cnt = 0
            epoch_preds = []
            epoch_labels = []
            best_val_accuracy = 0.0
            for batch in val_loader:
                inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True).to(device)
                labels = torch.tensor(batch['label'].tolist()).to(device)
                outputs = model(**inputs, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                
                correct_cnt += (preds == labels).sum().item()
                total_cnt += labels.size(0)

                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
            if len(val_loader) != 0:
                val_loss /= len(val_loader)
            if total_cnt != 0:
                accuracy = correct_cnt / total_cnt
            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                best_preds = epoch_preds.copy()  # Store best epoch's predictions
                best_labels = epoch_labels.copy()
            epoch_metrics = {
                'epoch': epoch+1,
                'val_loss': val_loss,
                'accuracy': accuracy
            }
            metrics_history.append(epoch_metrics)
            print(f"Epoch {epoch+1}/{num_epoch}, Validation Loss: {val_loss}, Accuracy: {accuracy}")
    save_metrics(best_labels, best_preds, metrics_history)
    return accuracy
    
def predict_helper(model,news:list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(news, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    model.eval()
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    label = torch.argmax(outputs.logits, dim=1)
    return [reverse_map[label.data[i].item()] for i in range(len(label))]

def save_metrics(true_labels, pred_labels, metrics_history):
    # Ensure we always get all 5 classes in the matrix/report
    labels = list(label_map.values())      # [0, 1, 2, 3, 4]
    class_names = list(label_map.keys())   # ['business', 'entertainment', 'politics', 'sport', 'tech']

    # 1. Confusion Matrix (explicitly pass labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    pd.DataFrame(cm,
                 index=class_names,
                 columns=class_names
                ).to_csv('metrics/confusion_matrix.csv')

    # 2. Classification Report (also pass labels + target_names)
    report = classification_report(
        true_labels,
        pred_labels,
        labels=labels,
        target_names=class_names,
        output_dict=True
    )
    with open('metrics/classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # 3. Training history
    with open('metrics/training_history.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)


def build_model_train_model(num_neurons = 512, lr=1e-2, batch_size = 32, num_epochs= 5):
    config = DistilBertConfig.from_pretrained(
        model_name,
        num_labels=5,  # Your 5 classes
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )

    # Load pretrained model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    ).to(device)

    # ====== KEY MODIFICATION ======
    # 1. Add a new 256-unit layer before the classifier
    model.classifier = nn.Sequential(
        nn.Linear(768, num_neurons),  # New layer (768 -> 256)
        nn.GELU(),            # Activation
        nn.Linear(num_neurons, 5)     # Final classifier (256 -> 5 classes)
    ).to(device)

    # 2. Freeze all layers EXCEPT the new classifier
    for param in model.parameters():
        param.requires_grad = False  # Freeze entire model

    # Unfreeze only the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    data = pd.read_csv("data/BBC News Train.csv")
    data = pd.concat([data["Text"],data["Category"]],axis=1)
    data['Category'] = data['Category'].map(label_map)
    predictions = train_model(model,data=data,lr=lr,batch_size=batch_size,num_epoch=num_epochs)
    return predictions,model

if __name__ =="__main__":
    config = DistilBertConfig.from_pretrained(
        model_name,
        num_labels=5,  # Your 5 classes
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )

    # Load pretrained model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    ).to(device)

    # ====== KEY MODIFICATION ======
    # 1. Add a new 256-unit layer before the classifier
    model.classifier = nn.Sequential(
        nn.Linear(768, 512),  # New layer (768 -> 256)
        nn.GELU(),            # Activation
        nn.Linear(512, 5)     # Final classifier (256 -> 5 classes)
    ).to(device)

    # 2. Freeze all layers EXCEPT the new classifier
    for param in model.parameters():
        param.requires_grad = False  # Freeze entire model

    # Unfreeze only the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    data = pd.read_csv("data/BBC News Train.csv")
    data = pd.concat([data["Text"],data["Category"]],axis=1)
    data['Category'] = data['Category'].map(label_map)
    train_model(model,data=data,lr=1e-2,batch_size=32,num_epoch=5)