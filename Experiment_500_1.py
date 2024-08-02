# Import necessary libraries
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from Transformer_200_Last import PangenomeTransformerModel
import pickle

# Load and prepare data
data_file = "best_500_dataset.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(data_file, 'rb') as f:
    df_raw = pickle.load(f)
print("Data loaded")

df_raw = df_raw.sample(n=317141, random_state=42)  # Sample 317141 rows for quick training
df_X = df_raw.drop(columns=[' serotype_encoded'])
df_y = df_raw[' serotype_encoded']

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.long), torch.tensor(y_train.values, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.long), torch.tensor(y_test.values, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

print("Dataloaders created")
print(f"Size of training data: {len(X_train)}")
print(f"Size of testing data: {len(X_test)}")
print(f"Shape of training data: {X_train.shape}")
print(f"Shape of testing data: {X_test.shape}")

# Model setup
model = PangenomeTransformerModel(vocab_size=21, n_embd=32, n_head=2, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 100
best_eval_f1 = 0.0
print("Starting training for the last block")
for epoch in range(epochs):
    model.train()
    total_train_loss = []
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X, return_all_blocks=True)  
        output = outputs[3].mean(dim=1)  
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Batch {i} - Loss: {loss.item()}")

    model.eval()
    total_val_loss = []
    all_y_val_true = []
    all_y_val_pred = []
    with torch.no_grad():
        for X_val, y_val in test_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val, return_all_blocks=True)  
            output = outputs[3].mean(dim=1)  
            eval_loss = criterion(output, y_val)
            total_val_loss.append(eval_loss.item())
            all_y_val_true.append(y_val.cpu())
            all_y_val_pred.append(output.cpu())

    y_val_true = torch.cat(all_y_val_true)
    y_val_pred = torch.cat(all_y_val_pred)
    eval_f1 = f1_score(y_val_true, y_val_pred.argmax(dim=1), average='macro')
    eval_loss = sum(total_val_loss) / len(total_val_loss)
    if eval_f1 > best_eval_f1:
        best_eval_f1 = eval_f1
        torch.save(model.state_dict(), 'best_model_last_block_500.pth')

    print(f"Epoch {epoch} - Eval loss: {eval_loss} - Eval F1: {eval_f1}")

print("Training for the 500 first block complete")
