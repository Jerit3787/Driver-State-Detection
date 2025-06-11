import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
import utils
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model import FaceKeypointModel
from datasets import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm

matplotlib.style.use('ggplot')

os.makedirs('outputs', exist_ok=True)

# Model.
model = FaceKeypointModel(pretrained=True, requires_grad=True).to(config.DEVICE)
# Optimizer.
optimizer = optim.Adam(model.parameters(), lr=config.LR)
# Loss function.
criterion = nn.SmoothL1Loss()

# Training function.
def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_mae = 0.0
    train_mse = 0.0
    counter = 0
    
    for i, data in tqdm(enumerate(dataloader), total=len(train_loader)):
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        
        # Calculate metrics
        outputs_np = outputs.detach().cpu().numpy()
        keypoints_np = keypoints.detach().cpu().numpy()
        train_mae += mean_absolute_error(keypoints_np.flatten(), outputs_np.flatten())
        train_mse += mean_squared_error(keypoints_np.flatten(), outputs_np.flatten())
        
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    train_mae = train_mae/counter
    train_mse = train_mse/counter
    return train_loss, train_mae, train_mse

# Validatioon function.
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    valid_mae = 0.0
    valid_mse = 0.0
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            counter += 1
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
            
            # Calculate metrics
            outputs_np = outputs.detach().cpu().numpy()
            keypoints_np = keypoints.detach().cpu().numpy()
            valid_mae += mean_absolute_error(keypoints_np.flatten(), outputs_np.flatten())
            valid_mse += mean_squared_error(keypoints_np.flatten(), outputs_np.flatten())
            
            # Plot the predicted validation keypoints after every
            # predefined number of epochs.
            if (epoch+1) % 1 == 0 and i == 0:
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    valid_mae = valid_mae/counter
    valid_mse = valid_mse/counter
    return valid_loss, valid_mae, valid_mse

train_loss = []
val_loss = []
train_mae_list = []
val_mae_list = []
train_mse_list = []
val_mse_list = []

for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    train_epoch_loss, train_epoch_mae, train_epoch_mse = fit(model, train_loader, train_data)
    val_epoch_loss, val_epoch_mae, val_epoch_mse = validate(model, valid_loader, valid_data, epoch)
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    train_mae_list.append(train_epoch_mae)
    val_mae_list.append(val_epoch_mae)
    train_mse_list.append(train_epoch_mse)
    val_mse_list.append(val_epoch_mse)
    
    print(f"Train Loss: {train_epoch_loss:.4f}, Train MAE: {train_epoch_mae:.4f}, Train MSE: {train_epoch_mse:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}, Val MAE: {val_epoch_mae:.4f}, Val MSE: {val_epoch_mse:.4f}')

# Loss plots.
plt.figure(figsize=(15, 12))

# Loss plot
plt.subplot(2, 2, 1)
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# MAE plot
plt.subplot(2, 2, 2)
plt.plot(train_mae_list, color='blue', label='train MAE')
plt.plot(val_mae_list, color='green', label='validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Training and Validation MAE')
plt.legend()
plt.grid(True)

# MSE plot
plt.subplot(2, 2, 3)
plt.plot(train_mse_list, color='purple', label='train MSE')
plt.plot(val_mse_list, color='brown', label='validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation MSE')
plt.legend()
plt.grid(True)

# RMSE plot
plt.subplot(2, 2, 4)
train_rmse = [np.sqrt(mse) for mse in train_mse_list]
val_rmse = [np.sqrt(mse) for mse in val_mse_list]
plt.plot(train_rmse, color='cyan', label='train RMSE')
plt.plot(val_rmse, color='magenta', label='validation RMSE')
plt.xlabel('Epochs')
plt.ylabel('Root Mean Squared Error')
plt.title('Training and Validation RMSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{config.OUTPUT_PATH}/performance_metrics.png", dpi=300, bbox_inches='tight')
plt.show()

# Save individual metric plots for detailed analysis
# Loss plot
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.show()

# MAE plot
plt.figure(figsize=(10, 7))
plt.plot(train_mae_list, color='blue', label='train MAE')
plt.plot(val_mae_list, color='green', label='validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Training and Validation MAE')
plt.legend()
plt.grid(True)
plt.savefig(f"{config.OUTPUT_PATH}/mae.png")
plt.show()

# MSE plot
plt.figure(figsize=(10, 7))
plt.plot(train_mse_list, color='purple', label='train MSE')
plt.plot(val_mse_list, color='brown', label='validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation MSE')
plt.legend()
plt.grid(True)
plt.savefig(f"{config.OUTPUT_PATH}/mse.png")
plt.show()

# RMSE plot
plt.figure(figsize=(10, 7))
plt.plot(train_rmse, color='cyan', label='train RMSE')
plt.plot(val_rmse, color='magenta', label='validation RMSE')
plt.xlabel('Epochs')
plt.ylabel('Root Mean Squared Error')
plt.title('Training and Validation RMSE')
plt.legend()
plt.grid(True)
plt.savefig(f"{config.OUTPUT_PATH}/rmse.png")
plt.show()

torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/model.pth")

print('DONE TRAINING')
