#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


#Data loading
conc_comp_str_data = pd.read_excel('Data/Concrete_Data.xls')
conc_comp_str_data.columns = ["Cement", "Furnace_Slag", "Fly_Ash", "Water", 
                              "Admixture", "C_Agg", "F_Agg", "Age", "Strength"]
#Preprocessing
#Slicing for 7-28 days data
wd = conc_comp_str_data[(conc_comp_str_data.Age>6.9) & (conc_comp_str_data.Age<28.5)].reset_index(drop = True)
#Scaling
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaled_wd = scaler1.fit_transform(wd)
scaled_wd_pd = pd.DataFrame(scaled_wd)
#Slicing for x,y data
x_wd = wd.iloc[:,[7,8]]
y_wd = wd.iloc[:,:7]
#Scaling x and y & converting to Dataframe
scaled_wd_x = scaler1.fit_transform(x_wd)
scaled_wd_y = scaler2.fit_transform(y_wd)
scaled_wd_y_pd = pd.DataFrame(scaled_wd_y)
scaled_wd_x_pd = pd.DataFrame(scaled_wd_x)
x = scaled_wd_x_pd.values
y = scaled_wd_y_pd.values
x_tensor = torch.tensor(x, dtype = torch.float32)
y_tensor = torch.tensor(y, dtype = torch.float32)
x_train, x_test,y_train, y_test = train_test_split(x_tensor,y_tensor, 
                                                   test_size = 0.2, 
                                                   random_state = 1234)
# Model Architecture
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class CrossConnectedMTL(nn.Module):
    def __init__(self, input_dim, output_names, cross_groups):
        super().__init__()
        self.output_names = output_names
        self.cross_groups = cross_groups
        
        # 1. Shared bottom
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            #nn.BNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.LayerNorm(64),
            #nn.ReLU(),
            #nn.Linear(128,64),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.Linear(64, 64),
            nn.ReLU(),
            #ResidualBlock(64),
            ResidualBlock(64)
        )
        
        self.cross_towers = nn.ModuleDict()
        self.cross_final_layers = nn.ModuleDict()
        self.independent_towers = nn.ModuleDict()
        
        # 2. Build towers for cross-connected groups
        for group in cross_groups:
            for output in group:
                self.cross_towers[output] = nn.Sequential(
                    nn.Linear(64, 64),
                   #nn.ReLU(),
                    #nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU()
                )
            group_output_dim = 64 * len(group)
            for output in group:
                self.cross_final_layers[output] = nn.Linear(group_output_dim, 1)
        
        # 3. Independent towers
        all_cross_outputs = {i for group in cross_groups for i in group}
        independent_outputs = [i for i in output_names if i not in all_cross_outputs]
        for output in independent_outputs:
            self.independent_towers[output] = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                #nn.Linear(128, 64),
                #nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def forward(self, x: Tensor):
        shared_features = self.shared_net(x)
        outputs = {}

        # Cross-connected outputs
        for group in self.cross_groups:
            tower_outs = [self.cross_towers[o](shared_features) for o in group]
            concat = torch.cat(tower_outs, dim=1)
            for i, o in enumerate(group):
                outputs[o] = self.cross_final_layers[o](concat)

        # Independent outputs
        for o, tower in self.independent_towers.items():
            outputs[o] = tower(shared_features)

        return outputs
#important dictionaries for model class and class instance

output_names = ['Cement','Furnace_Slag', 'Fly_ash', 'Water_content', 'Admixture_content',
                'Coarse_agg', 'Fine_agg']
cross_groups = [['Cement', 'Furnace_Slag','Fly_ash'],
                ['Water_content', 'Admixture_content']]
model = CrossConnectedMTL(input_dim=2, output_names=output_names, cross_groups=cross_groups)

#class for data conversion to dictionary
class ConcreteDataset(Dataset):
    def __init__(self, X, Y_dict):
        self.X = X  # Tensor of shape (n_samples, 2)
        self.Y_dict = Y_dict  # Dict of {target_name: Tensor of shape (n_samples,)}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = {k: v[idx] for k, v in self.Y_dict.items()}
        return x, y

target_cols = ['Cement','Furnace_Slag', 'Fly_ash', 'Water_content', 'Admixture_content',
                'Coarse_agg', 'Fine_agg']
Y_train_dict = {
    name: y_train[:, i]
    for i, name in enumerate(target_cols)
}

Y_test_dict = {
    name: y_test[:, i]
    for i, name in enumerate(target_cols)
}

#
#Final data loading and hyperparameters
train_dataset = ConcreteDataset(x_train, Y_train_dict)
test_dataset = ConcreteDataset(x_test, Y_test_dict)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#Loss function
def multitask_mse_loss(preds, targets):
    loss = 0.0
    for key in preds:
        loss += F.mse_loss(preds[key], targets[key])
    return loss / len(preds)
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#Training loop
torch.manual_seed(42)
num_epochs = 1500

model.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        preds = model(x_batch)
        
        y_batch = {k: v.float().view(-1, 1) for k, v in y_batch.items()}
        loss = multitask_mse_loss(preds, y_batch)
        train_loss += loss.item()
        

        optimizer.zero_grad()
        loss.backward()
        

        
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_loss = 0.0
        y_true_all = []
        y_pred_all = []
        for x_2, y_2 in test_loader:
            preds_2 = model(x_2)
            y_2 = {k: v.float().view(-1, 1) for k, v in y_2.items()}
            loss_count = multitask_mse_loss(preds_2, y_2)
            test_loss += loss_count.item()

            #y_true_batch = torch.cat([v for v in y_2.values()], dim=1).detach().cpu()
            #y_pred_batch = torch.cat([v for v in preds_2.values()], dim=1).detach().cpu()
            y_true_batch = torch.cat([y_2[name] for name in output_names], dim=1)
            y_pred_batch = torch.cat([preds_2[name] for name in output_names], dim=1)

            y_true_all.append(y_true_batch)
            y_pred_all.append(y_pred_batch)

        y_true_concat = torch.cat(y_true_all, dim=0).numpy()
        y_pred_concat = torch.cat(y_pred_all, dim=0).numpy()
        acc = r2_score(y_true_concat, y_pred_concat)

#if epoch % 200 == 0:
#print("Sample preds:", y_pred_concat[0])
#print("Sample true:", y_true_concat[0])

    if epoch%100 ==0:
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {acc:.4f}")

#Evaluation loop
def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               accuracy_function):
   
    model.eval()
    
    with torch.inference_mode():
        test_losses, acc = 0,0
        y_true_all = []
        y_pred_all = []
        
        for x,y in tqdm(test_loader):
            preds_2 = model(x)
            y = {k: v.float().view(-1, 1) for k, v in y.items()}
            loss_count = multitask_mse_loss(preds_2, y)
            test_losses += loss_count.item()
    
            y_true_batch = torch.cat([v for v in y.values()], dim=1).detach().cpu()
            y_pred_batch = torch.cat([v for v in preds_2.values()], dim=1).detach().cpu()
    
            y_true_all.append(y_true_batch)
            y_pred_all.append(y_pred_batch)
    
            y_true_concat = torch.cat(y_true_all, dim=0).numpy()
            y_pred_concat = torch.cat(y_pred_all, dim=0).numpy()
            
            acc += r2_score(y_true_concat, y_pred_concat)
        test_losses /= len(test_loader)
        acc /= len(test_loader)
    return {"model_name":model.__class__.__name__,
           "model_loss": test_losses,
           "model_acc": acc}


model_1_results = eval_model(model = model,
                           dataloader = test_loader,
                           criterion = multitask_mse_loss,
                           accuracy_function = r2_score)
model_1_results 

