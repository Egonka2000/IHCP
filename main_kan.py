from kan import *
import torch
import visualisation as visual
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yaml
import os
import random
import common
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

if __name__ == '__main__':
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))

    with open(f"{ABS_PATH}/config.yml", encoding='utf-8') as cfg:
        config = yaml.safe_load(cfg)
    
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    print("Loading IHCP temperature and heat transfer coefficient data...")
    if config['train_mode']:
        train_input_tensor, train_output_tensor = common.data_to_tensor(config['train']['input'], config['train']['output'], input_scaler, output_scaler)
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dataset = create_dataset_from_data(inputs=train_input_tensor, labels=train_output_tensor, train_ratio=0.9, device=device)
        
        steps = config['steps']
        loss_fn = nn.MSELoss()
        hidden_layers = config['hidden_layers']
        grid = config.get('grid', 10)
        k = config.get('k', 3)
        lamb = config.get('lamb', 0.001)
        lamb_entropy = config.get('lamb_entropy', 0.0)
        
        print(f"Training KAN with: {hidden_layers} hidden layers, grid={grid}, k={k}")
        print(f"Using device: {device}")

        model = KAN(width=[120, *hidden_layers, 86], grid=grid, k=k, seed=0, 
                   device=device, ckpt_path=f'./kan_models/{steps}_steps_standard')
        
        results = model.fit(dataset, 
                            opt="Adam", 
                            batch=config['batch_size'], 
                            steps=config['steps'],
                            lr=config['learning_rate'], 
                            loss_fn=loss_fn,
                            lamb=config['lamb'],
                            lamb_entropy=config['lamb_entropy'])
        visual.draw_loss_curve(results, step=steps)
    else:
        #val_input_tensor, val_output_tensor = common.data_to_tensor(config['validate']['input'], config['validate']['output'], input_scaler, output_scaler)
        test_input_tensor, test_output_tensor = common.data_to_tensor(config['test']['input'], config['test']['output'], input_scaler, output_scaler)

        model = KAN.loadckpt(f'{ABS_PATH}/kan_models/10000_steps_standard/0.1')
        
        model.train(mode=False)
        for i in range(5):
            pred = model(test_input_tensor[i].reshape(1, -1))
            actual = output_scaler.inverse_transform(test_output_tensor[i].reshape(1, -1))[0]
            predicted = output_scaler.inverse_transform(pred.detach().numpy())[0]

            visual.draw_htc_diagram(actual, predicted)

        subset_size = 100000
        torch.manual_seed(int(time.time()))
        indices = torch.randperm(len(test_input_tensor))[:subset_size]
        test_input_subset = test_input_tensor[indices]
        test_output_subset = test_output_tensor[indices]

        model.eval()
        with torch.no_grad():
            preds = model(test_input_subset)

        preds_np = preds.detach().numpy()
        actual_np = test_output_subset.detach().numpy()

        predicted = output_scaler.inverse_transform(preds_np)
        actual = output_scaler.inverse_transform(actual_np)

        print(root_mean_squared_error(actual, predicted))
        print(mean_absolute_percentage_error(actual, predicted))
