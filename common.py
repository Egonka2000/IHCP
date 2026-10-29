import numpy as np
import pandas as pd
import torch
from typing import Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def create_kan_dataset(train_inputs, train_labels, val_inputs, val_labels, device):
    dataset = {}
    dataset['train_input'] = train_inputs.detach().to(device)
    dataset['test_input'] = val_inputs.detach().to(device)
    dataset['train_label'] = train_labels.detach().to(device)
    dataset['test_label'] = val_labels.detach().to(device)

    return dataset

def data_to_tensor(input_path: str, output_path: str, input_scaler: Union[MinMaxScaler, StandardScaler], output_scaler: Union[MinMaxScaler, StandardScaler]):
    input_df = pd.DataFrame(create_dataset(input_path, 120))
    output_df = pd.DataFrame(create_dataset(output_path, 86))

    scaled_input = input_scaler.fit_transform(input_df.copy())
    scaled_output = output_scaler.fit_transform(output_df.copy())

    input_normalized_df = pd.DataFrame(data=scaled_input, columns=input_df.columns)
    output_normalized_df = pd.DataFrame(data=scaled_output, columns=output_df.columns)

    input_tensor = torch.tensor(input_normalized_df.values, dtype=torch.float32)
    output_tensor = torch.tensor(output_normalized_df.values, dtype=torch.float32)

    return input_tensor, output_tensor

def create_dataset(path: str, column: int):
    with open(path, 'rb') as file:
        binary_content = file.read()

    data = np.frombuffer(binary_content, dtype=np.float32)

    return data.reshape(-1, column)