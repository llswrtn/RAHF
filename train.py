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
#base_learning_rate = 0.015
base_learning_rate = 1

wandb.init(
    # set the wandb project where this run will be logged
    project="rhf_test_project_0",

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



# Training Loop
def train(model, t5_text_encoder, t5_encoder, vit_model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    t5_encoder.train()
    vit_model.train()
    t5_text_encoder.eval() # keep frozen? would that prevent catastrophic forgetting?
    #t5_text_encoder.train()

    model.to(device)
    t5_encoder.to(device)
    vit_model.to(device)
    t5_text_encoder.to(device)


    for iteration, (images, texts, target_heatmaps) in enumerate(dataloader, 1):  # start iteration count at 1

        optimizer.zero_grad()

        images = images.to(device)  # Shape: (batch_size, 3, 224, 224)
        texts = texts.to(device)    # Shape: (batch_size, seq_len)
        target_heatmaps = target_heatmaps.to(device)  # Shape: (batch_size, 1, 224, 224)

        # Step 1: Extract Image Tokens with ViT
        vit_outputs = vit_model(pixel_values=images)

        image_tokens = vit_outputs.last_hidden_state  # Shape: (batch_size, 196, 768)

        # Step 2: Extract Text Tokens with T5 (FROZEN for now)
        with torch.no_grad():
            text_tokens = t5_text_encoder(input_ids=texts).last_hidden_state  # Shape: (batch_size, n_text_tokens, 768)

        # Step 3: Concatenate Image and Text Tokens
        fused_tokens = torch.cat([image_tokens, text_tokens], dim=1)  # Shape: (batch_size, 196 + n_text_tokens, 768)

        # Step 4: Encode Fused Tokens with T5 Encoder
        fused_encoded_tokens = t5_encoder(inputs_embeds=fused_tokens).last_hidden_state  # Shape: (batch_size, seq_len, 768)

        # Step 5: Extract Image Tokens and Reshape to Feature Map
        image_tokens_encoded = fused_encoded_tokens[:, :196, :]  # Shape: (batch_size, 196, 768)

        image_size = 224
        patch_size = 16

        num_patches_side = image_size // patch_size

        image_feature_map = image_tokens_encoded.view(images.size(0), num_patches_side, num_patches_side, 768).permute(0, 3, 1, 2)  # (batch_size, 768, 14, 14)
        #print(image_feature_map.size())
        # Step 6: Heatmap Prediction
        predicted_heatmap = model(image_feature_map)  # Shape: (batch_size, 1, 224, 224)

        # Step 7: Compute Loss
        loss = criterion(predicted_heatmap, target_heatmaps)

        # Step 8: Backpropagation and optimization

        loss.backward()
        optimizer.step()
        lr_log = scheduler.get_last_lr()[0]
        wandb.log({"loss": loss, "learning_rate":lr_log})
        # Update the learning rate
        scheduler.step()


        print(f"Iteration {iteration}, Training Loss: {loss.item()}, Learning Rate: {lr_log}")

        print(f"Training Loss: {loss.item()}")



# Load your dataset
train_dataset = RHFDataset(images_dir, metadata_dir)

# FOR TESTING ONLY, REMOVE NEXT 2 LINES AFTER TESTING
subset_indices = list(range(20))  # Indices of the first 10 samples
train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


criterion = nn.MSELoss()  # Assuming target heatmaps are continuous values


optimizer = optim.AdamW(list(heatmap_predictor.parameters()) +
                        list(t5_encoder.parameters()) +
                        list(vit_model.parameters()),
                        lr=base_learning_rate)

#optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
def lr_lambda(current_step):
    warmup_steps = 2000
    if current_step < warmup_steps:
        return current_step / warmup_steps  # Linear warm-up
    return (warmup_steps / current_step) ** 0.5  # Reciprocal square root decay

# LambdaLR scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


#

# Training Loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(heatmap_predictor, t5_text_encoder, t5_encoder, vit_model, train_loader, criterion, optimizer, scheduler, device)