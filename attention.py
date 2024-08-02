import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformer_model import PangenomeTransformerModel
import pickle
import os

def load_data():
    print("Loading data...")
    with open('best_200_dataset.pkl', 'rb') as f:
        df_raw = pickle.load(f)
    df_raw.columns = df_raw.columns.str.strip()
    df_X = df_raw.drop(columns=['serotype_encoded'])
    df_y = df_raw['serotype_encoded']
    print("Data loaded successfully.")
    return train_test_split(df_X, df_y, test_size=0.2, random_state=42)

def prepare_dataloaders(X_test, y_test):
    print("Preparing dataloaders...")
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.long), torch.tensor(y_test.values, dtype=torch.long))
    print("Dataloaders prepared.")
    return DataLoader(test_dataset, batch_size=1, shuffle=False)

def load_model():
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PangenomeTransformerModel(21, 32, 2, 4, 0.1).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    print("Model loaded and set to evaluation mode.")
    return model, device

def check_memory(stage):
    print(f"{stage} - Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"{stage} - Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB")
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory metrics after logging

def compute_attention_matrices(loader, model, device):
    print("Computing attention matrices...")
    serotype_attention = {}
    for i, (inputs, labels) in enumerate(loader):
        serotype = labels.item()
        if i % 100 == 0:  # Log less frequently to avoid clutter
            print(f"Processing serotype: {serotype} at iteration {i}")
        
        inputs = inputs.to(device)
        check_memory("Before processing")
        
        with torch.no_grad():
            attention_matrix = model.get_attention_matrix(inputs)
        attention_matrix_np = attention_matrix.cpu().numpy()
        
        check_memory("After processing")
        
        if serotype not in serotype_attention:
            serotype_attention[serotype] = []
        serotype_attention[serotype].append(attention_matrix_np)

    print("Finished computing attention matrices.")
    return serotype_attention

def average_attention_matrices(serotype_attention):
    print("Averaging attention matrices...")
    averaged_attention = {}
    for serotype, matrices in serotype_attention.items():
        averaged_attention[serotype] = np.mean(matrices, axis=0)
    print("Attention matrices averaged.")
    return averaged_attention

def save_data(data, filename):
    print(f"Saving data to {filename}...")
    np.save(filename, data)
    print("Data saved successfully.")

def main():
    X_train, X_test, y_train, y_test = load_data()
    loader = prepare_dataloaders(X_test, y_test)
    model, device = load_model()
    serotype_attention = compute_attention_matrices(loader, model, device)
    averaged_attention = average_attention_matrices(serotype_attention)
    save_data(averaged_attention, 'averaged_attention_matrices.npy')
    print("All attention matrices have been calculated and saved.")

if __name__ == '__main__':
    main()
