from RAHFHeatmapModel import RAHFHeatmapModel
from heatmap_predictor import HeatmapPredictor
from RHFDataset import RHFDataset
from util import lr_lambda, save_checkpoint, load_checkpoint

import argparse
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, ViTModel
import wandb


# INPUTS
IMAGES_DIR = '/Users/luisa/RHF/pick_a_pic/richHF_images_w_captions/test/images'
METADATA_DIR = '/Users/luisa/RHF/RHF_parsed_dataset_with_captions/test'
#IMAGES_DIR = '/content/drive/MyDrive/RHF/pick_a_pic/richHF_images_w_captions/train/images'
#METADATA_DIR = '/content/drive/MyDrive/RHF/RHF_dataset_with_captions/train/'
#IMAGES_DIR = '/content/drive/MyDrive/RHF/pick_a_pic/richHF_images_w_captions/test/images'
#METADATA_DIR = '/content/drive/MyDrive/RHF/RHF_dataset_with_captions/test/'

#CHECKPOINT_LOADPATH = os.path.join(checkpoint_dir, 'rhf_test_project_1/rahf_model_checkpoint_epoch0_iteration10.pth')
CHECKPOINT_LOADPATH= None
OUTPUT_DIR = ''

WANDB_PROJECT_NAME = 'rhf-test-project-wandb-continue-1'
WANDB_ENTITY = 'll_swrtn-heidelberg-university'

# HYPERPARAMETERS
BATCH_SIZE = 1
NUM_EPOCHS = 1 #total number of epochs to train, including start_epoch count
BASE_LEARNING_RATE = 0.015
#BASE_LEARNING_RATE = 1


def train(rahf_model, dataloader, criterion, optimizer, scheduler, device, start_epoch, num_epochs, start_iteration, wandb_run, checkpoint_savedir):

    rahf_model.to(device)
    ''' 
    heatmap_predictor.to(device)
    t5_encoder.to(device)
    vit_model.to(device)
    t5_text_encoder.to(device)   
    '''
    iteration = start_iteration  # Start from the loaded iteration
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")


        for images, texts, target_heatmaps in dataloader:  # start iteration count at 1

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
            iteration += 1

        wandb_run_id = wandb_run.id
        d = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_savename = 'rahf-model-checkpoint-run_' + wandb_run_id + '_epoch'+ str(epoch) + '_iteration' + str(iteration) + "_" + d + '.pth'
        checkpoint_savepath = os.path.join(checkpoint_savedir, checkpoint_savename)

        # Save checkpoint at the end of each epoch
        save_checkpoint(epoch, iteration, rahf_model, optimizer, scheduler, checkpoint_savepath)

def main(args):
    # SETUP

    images_dir = args.images_dir if args.images_dir else IMAGES_DIR
    metadata_dir = args.metadata_dir if args.metadata_dir else METADATA_DIR
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR

    num_epochs = args.num_epochs if args.num_epochs else NUM_EPOCHS
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    base_learning_rate = args.learning_rate if args.learning_rate else BASE_LEARNING_RATE
    wandb_project_name = args.wandb_project_name if args.wandb_project_name else WANDB_PROJECT_NAME

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.wandb_key:
        wandb.login(key=args.wandb_key)



    if args.load:
        # load checkpoint and init wandb from project name parsed from savedir

        checkpoint_loadpath = args.load

        wandb_run_id = os.path.basename(args.load).split('_')[1]
        wandb_project_name = args.load.split('/')[-2]
        wandb_run = wandb.init(entity= WANDB_ENTITY, project=wandb_project_name, id=wandb_run_id, resume="must")



    else:
        # start training from scratch
        checkpoint_loadpath = None
        wandb_run = wandb.init(
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


    # Set paths for checkpoints
    checkpoint_dir = 'checkpoints'
    checkpoint_savedir = os.path.join(output_dir, checkpoint_dir, wandb_project_name)
    os.makedirs(checkpoint_savedir, exist_ok=True)

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


    # Load your dataset
    train_dataset = RHFDataset(images_dir, metadata_dir)

    # FOR DEBUGGING ONLY
    if args.test_run:
        subset_indices = list(range(5))  # Indices of the first 10 samples
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

    # Load checkpoint if available, otherwise start from scratch
    start_epoch, start_iteration = load_checkpoint(rahf_model, optimizer, scheduler, checkpoint_loadpath)
    # if loaded from checkpoint, start training at next epoch, not the one we stopped at
    if args.load:
        start_epoch += 1
        print(f"Checkpoint loaded: starting from epoch {start_epoch}, iteration {start_iteration}.")

    train(rahf_model, train_loader, criterion, optimizer, scheduler, device, start_epoch, num_epochs, start_iteration, wandb_run, checkpoint_savedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # either path to config file or checkpoint required
    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument(
        "--config_file", type=str, help="path to config yaml file")

    # optional
    group.add_argument(
        "--load", type=str, help="path where model to be loaded is stored"
    )

    parser.add_argument("--num_epochs", type=int, help="TOTAL number of epochs to train, if not from config")
    parser.add_argument("--batch_size", type=int, help="batch size, if not from config")
    parser.add_argument("--images_dir", type=str, help="path to training images")
    parser.add_argument("--metadata_dir", type=str, help="path to metadata for training images")
    parser.add_argument("--wandb_project_name", type=str, help="name of Weights&Biases Project")
    parser.add_argument("--learning_rate", type=int, help="name of Weights&Biases Project")
    parser.add_argument("--test_run", action='store_true', help="run training with only 5 instances for debugging")
    parser.add_argument("--output_dir", type=str, help="path to where to save output")
    parser.add_argument("--wandb_key", type=str, help="wandb api key")


    args = parser.parse_args()


    main(args)