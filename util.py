import numpy as np
import os
import torch
from PIL import Image

def lr_lambda(current_step):
    # warmup_steps = 2   # for debugging
    warmup_steps = 2000
    if current_step < warmup_steps:
        return current_step / warmup_steps  # Linear warm-up
    return (warmup_steps / current_step) ** 0.5  # Reciprocal square root decay

# transforms images tensors after preprocessing back to original
def recover_preprocessed_image(images):
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])

    unnormalized_images = images * std[:, None, None] + mean[:, None, None]
    unnormalized_images= unnormalized_images.permute(0, 2, 3, 1).clamp(0, 1).numpy()  # [H, W, C]
    unnormalized_images = (unnormalized_images * 255).astype('uint8')


    return  unnormalized_images
"""
def recover_preprocessed_image(image):
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])

    unnormalized_images = image * std[:, None, None] + mean[:, None, None]
    unnormalized_images= unnormalized_images.permute(1, 2, 0).clamp(0, 1).numpy()  # [H, W, C]
    unnormalized_images = (unnormalized_images * 255).astype('uint8')


    return  unnormalized_images
"""


# transforms normalized heatmap back to original [0,255]
def recover_heatmaps(heatmaps):
    #heatmaps = (heatmaps * 255).squeeze().numpy().astype('uint8')
    heatmaps = (heatmaps * 255).squeeze(1).numpy().astype('uint8')  # [B, H, W]

    return heatmaps


# Input: recovered images and unnormalized heatmaps
def generate_and_save_visualizations(images, target_heatmaps, orig_names, predicted_heatmaps, output_dir):
    images = recover_preprocessed_image(images)
    predicted_heatmaps = recover_heatmaps(predicted_heatmaps)
    target_heatmaps = recover_heatmaps(target_heatmaps)

    for i in range(len(images)):
        image = Image.fromarray(images[i])
        image.save(os.path.join(output_dir, orig_names[i] + "_image" + '.png'))

        target_heatmap = Image.fromarray(target_heatmaps[i])
        target_heatmap.save(os.path.join(output_dir, orig_names[i] + "_gt" + '.png'))

        pred_heatmap = predicted_heatmaps[i]
        ''''
        # check non-zero values
        non_zero_values = pred_heatmap[pred_heatmap != 0]
        # Find unique non-zero values
        unique_non_zero_values = np.unique(non_zero_values)
        # Print the result
        print("Unique non-zero values:\n", unique_non_zero_values)
        print()
        
        print('num nonzero heatmaps: ', len(nonzero_pred_heatmaps))
        nonzero_pred_heatmaps = np.asarray(nonzero_pred_heatmaps)
        np.savetxt(os.path.join(output_dir, 'nonzero_pred_heatmaps.csv'), nonzero_pred_heatmaps, fmt="%s", delimiter=",")
        '''
        pred_heatmap = Image.fromarray(pred_heatmap)
        pred_heatmap.save(os.path.join(output_dir, orig_names[i] + "_pred" + '.png'))
        #np.savetxt(os.path.join(output_dir, orig_names[i] + "_pred" + '.csv'), pred_heatmap, fmt="%s",delimiter=",")





