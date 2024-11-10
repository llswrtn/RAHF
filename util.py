import os
import torch

# Function to save the model checkpoint
# Learning rate scheduler
def lr_lambda(current_step):
    #warmup_steps = 2   # for debugging
    warmup_steps = 2000
    if current_step < warmup_steps:
        return current_step / warmup_steps  # Linear warm-up
    return (warmup_steps / current_step) ** 0.5  # Reciprocal square root decay
'''
def save_checkpoint(epoch, iteration, model, optimizer, scheduler, path):
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}, iteration {iteration}.")

# Function to load the model checkpoint
def load_checkpoint(model, optimizer, scheduler, path):
    if path is not None and os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        return epoch, iteration
    else:
        print(f"No checkpoint found at {path}. Starting from scratch.")
        return 0, 0
'''