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

#RMSE 

def calculate_mtl_rmse(model, test_loader, target_cols):
    """
    Calculates the RMSE for each task in the CrossConnectedMTL architecture.
    """
    model.eval()
    all_preds = {name: [] for name in target_cols}
    all_trues = {name: [] for name in target_cols}

    with torch.no_grad():
        for x_batch, y_batch_dict in test_loader:
            # Model returns dictionary: {'Cement': tensor, ...}
            preds_dict = model(x_batch) 
            
            for name in target_cols:
                all_preds[name].append(preds_dict[name].cpu().numpy())
                all_trues[name].append(y_batch_dict[name].cpu().numpy())

    print(f"\n{'Output Task':<20} | {'RMSE Score':<15}")
    print("-" * 40)

    total_rmse = 0
    for name in target_cols:
        # Concatenate batches and flatten to 1D
        y_true = np.concatenate(all_trues[name]).flatten()
        y_pred = np.concatenate(all_preds[name]).flatten()
        
        # Calculate RMSE: sqrt(mean_squared_error)
        rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"{name:<20} | {rmse_score:.4f}")
        total_rmse += rmse_score

    print("-" * 40)
    print(f"{'Average RMSE':<20} | {total_rmse / len(target_cols):.4f}")

# --- EXECUTION ---
target_names = ['Cement', 'Furnace_Slag', 'Fly_ash', 'Water_content', 
                'Admixture_content', 'Coarse_agg', 'Fine_agg']

calculate_mtl_rmse(model, test_loader, target_names)

#MAE
def calculate_mtl_mae_unscaled(model, test_loader, target_cols, scaler):
    """
    Calculates MAE in original units (kg/m³) for the MTL model.
    Fixes the 3D array error by ensuring 2D shapes before inverse_transform.
    """
    model.eval()
    
    all_preds_scaled = []
    all_trues_scaled = []

    with torch.no_grad():
        for x_batch, y_batch_dict in test_loader:
            preds_dict = model(x_batch)
            
            # Stack and ensure it is 2D (Batch Size, Number of Tasks)
            # .squeeze() or .view(-1, len(target_cols)) handles extra dimensions
            batch_preds = torch.stack([preds_dict[name].flatten() for name in target_cols], dim=1)
            batch_trues = torch.stack([y_batch_dict[name].flatten() for name in target_cols], dim=1)
            
            all_preds_scaled.append(batch_preds.cpu().numpy())
            all_trues_scaled.append(batch_trues.cpu().numpy())

    # 2. Concatenate into (Total_Samples, 7)
    y_pred_scaled = np.vstack(all_preds_scaled)
    y_true_scaled = np.vstack(all_trues_scaled)

    # Safety check: print shape to confirm it's (Samples, 7)
    print(f"DEBUG: Scaled array shape: {y_pred_scaled.shape}")

    # 3. Inverse Transform to original units (kg/m³)
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)
    y_true_unscaled = scaler.inverse_transform(y_true_scaled)

    # 4. Compute MAE per task
    results = []
    print(f"\n{'Output Task':<20} | {'MAE (kg/m³)':<15}")
    print("-" * 40)

    for i, name in enumerate(target_cols):
        mae_score = mean_absolute_error(y_true_unscaled[:, i], y_pred_unscaled[:, i])
        results.append({"Task": name, "MAE": mae_score})
        print(f"{name:<20} | {mae_score:.4f}")

    mean_mae = np.mean([r["MAE"] for r in results])
    print("-" * 40)
    print(f"{'MTL System Mean':<20} | {mean_mae:.4f}")
    
    return pd.DataFrame(results)

# --- EXECUTION ---
mae_mtl_df = calculate_mtl_mae_unscaled(model, test_loader, target_names, scaler2)
