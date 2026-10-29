from torch.utils.data import DataLoader, TensorDataset
from neural_network_mlp import NeuralNetwork
import visualisation as visual
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, mean_squared_error
import yaml
import os
import common
import torch, time

if __name__ == '__main__':
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))

    with open(f"{ABS_PATH}/config.yml", encoding='utf-8') as cfg:
        config = yaml.safe_load(cfg)

    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    train_input_tensor, train_output_tensor = common.data_to_tensor(config['train']['input'], config['train']['output'], input_scaler, output_scaler)
    val_input_tensor, val_output_tensor = common.data_to_tensor(config['validate']['input'], config['validate']['output'], input_scaler, output_scaler)
    test_input_tensor, test_output_tensor = common.data_to_tensor(config['test']['input'], config['test']['output'], input_scaler, output_scaler)

    train_dataset =  TensorDataset(train_input_tensor, train_output_tensor)
    val_dataset =  TensorDataset(val_input_tensor, val_output_tensor)
    test_dataset =  TensorDataset(test_input_tensor, test_output_tensor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    network = NeuralNetwork()
    network.build_network(input_size=train_input_tensor.shape[1], hidden_layers=config['hidden_layers'], output_size=train_output_tensor.shape[1], learning_rate=config['learning_rate'])

    if config['train_mode']:
        epochs = 2
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            network.train_loop(train_dataloader)
            network.test_loop(val_dataloader)
        print("Done!")

        network.save_model(f"./mlp_models/model_{epochs}_epochs_standard.pth")
    else:
        network.load_model(f'{ABS_PATH}/mlp_models/model_2_epochs.pth')

        for i in range(5):
            pred = network.test_one(test_input_tensor[i])
            actual = output_scaler.inverse_transform(test_output_tensor[i].reshape(1, -1))[0]
            predicted = output_scaler.inverse_transform(pred)[0]

            visual.draw_htc_diagram(actual, predicted)

        subset_size = 100000
        torch.manual_seed(int(time.time()))
        indices = torch.randperm(len(test_input_tensor))[:subset_size]
        test_input_subset = test_input_tensor[indices]
        test_output_subset = test_output_tensor[indices]

        network.model.eval()
        with torch.no_grad():
            test_input_subset = test_input_subset.to(network.device)
            preds = network.model(test_input_subset)

        preds_np = preds.cpu().detach().numpy()
        actual_np = test_output_subset.cpu().detach().numpy()

        predicted = output_scaler.inverse_transform(preds_np)
        actual = output_scaler.inverse_transform(actual_np)

        print(mean_squared_error(actual, predicted))
        print(root_mean_squared_error(actual, predicted))
        print(mean_absolute_percentage_error(actual, predicted))
