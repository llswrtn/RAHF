from RAHFHeatmapModel import RAHFHeatmapModel
from heatmap_predictor import HeatmapPredictor
from RHFDataset import RHFDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, ViTModel
import wandb


#images_dir = '/content/drive/MyDrive/RHF/pick_a_pic/richHF_images_w_captions/train/images'
#metadata_dir = '/content/drive/MyDrive/RHF/RHF_dataset_with_captions/train/'
#images_dir = '/content/drive/MyDrive/RHF/pick_a_pic/richHF_images_w_captions/test/images'
#metadata_dir = '/content/drive/MyDrive/RHF/RHF_dataset_with_captions/test/'

images_dir ='/Users/luisa/RHF/pick_a_pic/richHF_images_w_captions/test/images'
metadata_dir = '/Users/luisa/RHF/RHF_parsed_dataset_with_captions/test'

# Hyperparameters and setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
num_epochs = 1
base_learning_rate = 0.015
#base_learning_rate = 1

wandb_project_name = 'rhf_test_project_0.1'

wandb.init(
    # set the wandb project where this run will be logged
    project= wandb_project_name,

    # track hyperparameters and run metadata
    config={
    "base_learning_rate": base_learning_rate,
    "architecture": "heatmap_pred",
    "dataset": "test",
    "epochs": num_epochs,
    "batch_size": batch_size
    }
)

# Initialize model, loss, and optimizer
heatmap_predictor = HeatmapPredictor()
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
t5_text_encoder = T5EncoderModel.from_pretrained('t5-base')
t5_encoder = T5EncoderModel.from_pretrained('t5-base')  #  12 layers, 12 heads

heatmap_predictor.train()
t5_encoder.train()
vit_model.train()
t5_text_encoder.eval()  # keep frozen? would that prevent catastrophic forgetting?
# t5_text_encoder.train()

rahf_model = RAHFHeatmapModel(heatmap_predictor, t5_text_encoder, t5_encoder, vit_model)
# Training Loop

# Learning rate scheduler
def lr_lambda(current_step):
    warmup_steps = 2000
    if current_step < warmup_steps:
        return current_step / warmup_steps  # Linear warm-up
    return (warmup_steps / current_step) ** 0.5  # Reciprocal square root decay


# Load your dataset
train_dataset = RHFDataset(images_dir, metadata_dir)

# FOR TESTING ONLY, REMOVE NEXT 2 LINES AFTER TESTING
subset_indices = list(range(20))  # Indices of the first 10 samples
train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


criterion = nn.MSELoss()  # Assuming target heatmaps are continuous values

''' 
optimizer = optim.AdamW(list(heatmap_predictor.parameters()) +
                        list(t5_encoder.parameters()) +
                        list(vit_model.parameters()),
                        lr=base_learning_rate)
'''

optimizer = optim.AdamW(rahf_model.parameters(), lr=base_learning_rate)



# LambdaLR scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train(rahf_model, dataloader, criterion, optimizer, scheduler, device):

    rahf_model.to(device)
    ''' 
    heatmap_predictor.to(device)
    t5_encoder.to(device)
    vit_model.to(device)
    t5_text_encoder.to(device)   
    '''

    for iteration, (images, texts, target_heatmaps) in enumerate(dataloader, 1):  # start iteration count at 1

        optimizer.zero_grad()

        images = images.to(device)  # Shape: (batch_size, 3, 224, 224)
        texts = texts.to(device)    # Shape: (batch_size, seq_len)
        target_heatmaps = target_heatmaps.to(device)  # Shape: (batch_size, 1, 224, 224)

        # Step 1 to 6: Heatmap Prediction
        predicted_heatmap= rahf_model(images, texts)

        # Step 7: Compute Loss
        loss = criterion(predicted_heatmap, target_heatmaps)

        # Step 8: Backpropagation and optimization
        loss.backward()
        optimizer.step()
        lr_log = scheduler.get_last_lr()[0]
        #wandb.log({"loss": loss, "learning_rate":lr_log})
        # Update the learning rate
        scheduler.step()

        # Log metrics to Weights & Biases in real-time
        wandb.log({
            "epoch": epoch,
            "train_loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0]
        })

        print(f"Iteration {iteration}, Training Loss: {loss.item()}, Learning Rate: {lr_log}")



# Training Loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(rahf_model, train_loader, criterion, optimizer, scheduler, device)
#wandb.finish()
