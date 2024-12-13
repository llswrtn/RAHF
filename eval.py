import argparse
import numpy as np
import os
from pysaliency.metrics import MIT_KLDiv, SIM, CC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, ViTModel

from RAHFHeatmapModel import RAHFHeatmapModel
from heatmap_predictor import HeatmapPredictor
from RHFDataset import RHFDataset
from util import lr_lambda, generate_and_save_visualizations
from eval_metrics import NSS, AUC_Judd


IMAGES_DIR = '/Users/luisa/RHF/pick_a_pic/richHF_images_w_captions/test/images'
METADATA_DIR = '/Users/luisa/RHF/RHF_parsed_dataset_with_captions/test'
OUTPUT_DIR = 'eval_test_output'
BATCH_SIZE = 4
NUM_WORKERS = 4

def evaluate_model(model, dataloader, device, output_dir, visualizations=False):
    model.eval()
    mse_all = []
    mse_empty = []
    saliency_metrics = {
        "NSS": [],
        "KLD": [],
        "AUC_Judd": [],
        "SIM": [],
        "CC": []
    }

    os.makedirs(output_dir, exist_ok = True)
    with torch.no_grad():

        for batch in dataloader:

            images, texts, target_heatmaps, orig_names = batch

            images.to(device)
            target_heatmaps.to(device)

            # Predict heatmaps
            predicted_heatmaps = model(images, texts)

            #if visualizations set to True, recover preprocessed images and save visualizations
            if visualizations:
                generate_and_save_visualizations(images, target_heatmaps, orig_names, predicted_heatmaps, output_dir)

            # Compute MSE for all samples
            batch_mse = F.mse_loss(predicted_heatmaps, target_heatmaps, reduction='none').mean(dim=[1, 2, 3]).cpu().numpy() # mean over C, H, W (dim=[1, 2, 3])
            mse_all.extend(batch_mse)

            # Separate samples with empty ground truth
            empty_gt_mask = (target_heatmaps.sum(dim=[1, 2, 3]) == 0).cpu().numpy()
            non_empty_gt_mask = ~empty_gt_mask
            mse_empty.extend(batch_mse[empty_gt_mask])

            # Saliency metrics for non-empty ground truth
            np.seterr(divide='ignore') # divide by zero encountered in log from MIT_KLDiv
            for i, is_non_empty in enumerate(non_empty_gt_mask):
                if is_non_empty:
                    gt = target_heatmaps[i].cpu().numpy()
                    pred = predicted_heatmaps[i].cpu().numpy()

                    # Normalize ground truth and predicted heatmaps (Directly in functions of metrics)

                    # Compute saliency metrics
                    saliency_metrics["NSS"].append(NSS(pred, gt))
                    saliency_metrics["KLD"].append(MIT_KLDiv(pred, gt)) #compute image-based KL divergence with same hyperparameters as in Tuebingen/MIT Saliency Benchmark, see source in NSS definition, should be the same as in paper?

                    saliency_metrics["AUC_Judd"].append(AUC_Judd(pred, gt))
                    saliency_metrics["SIM"].append(SIM(pred, gt))
                    saliency_metrics["CC"].append(CC(pred, gt))
            np.seterr(divide='warn')

    # Aggregate results
    results = {
        "MSE_All": np.mean(mse_all),
        "MSE_Empty_GT": np.mean(mse_empty) if mse_empty else None,
        "Saliency_Metrics": {key: np.mean(values) for key, values in saliency_metrics.items()}
    }

    return results


def main(args):
    # Define paths and parameters
    model_checkpoint = args.checkpoint
    images_dir = args.images_dir if args.images_dir else IMAGES_DIR
    metadata_dir = args.metadata_dir if args.metadata_dir else METADATA_DIR
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR

    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    num_workers = args.num_workers if args.num_workers else NUM_WORKERS

    # Load dataset
    test_dataset = RHFDataset(images_dir, metadata_dir, train=False)

    # FOR DEBUGGING ONLY
    if args.test_run:
        subset_indices = list(range(5))  # Indices of the first 10 samples
        test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    heatmap_predictor = HeatmapPredictor()
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    t5_text_encoder = T5EncoderModel.from_pretrained('t5-base')
    t5_encoder = T5EncoderModel.from_pretrained('t5-base')  #  12 layers, 12 heads

    base_learning_rate = 0.015


    criterion = nn.MSELoss()
    '''
    model = RAHFHeatmapModel.load_from_checkpoint(
        model_checkpoint,
        heatmap_predictor=heatmap_predictor,
        t5_text_encoder=t5_text_encoder,
        t5_encoder=t5_encoder,
        vit_model=vit_model,
        criterion=criterion,
        base_learning_rate=base_learning_rate,
        scheduler_lambda=lr_lambda,
                                                    )
    '''
    model = RAHFHeatmapModel(
        heatmap_predictor=heatmap_predictor,
        t5_text_encoder=t5_text_encoder,
        t5_encoder=t5_encoder,
        vit_model=vit_model,
        criterion=criterion,
        base_learning_rate=base_learning_rate,
        scheduler_lambda=lr_lambda,
                                                    )


    model.to(device)

    # Evaluate model
    results = evaluate_model(model, test_dataloader, device, output_dir, args.visualizations)

    # Print results
    print("Evaluation Results:")
    print(f"MSE (All Samples): {results['MSE_All']:.4f}")
    if results["MSE_Empty_GT"] is not None:
        print(f"MSE (Empty Ground Truth): {results['MSE_Empty_GT']:.4f}")
    else:
        print("No samples with empty ground truth.")
    print("Saliency Metrics for Non-Empty Ground Truth:")
    for metric, value in results["Saliency_Metrics"].items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True, type=str, help="path to model checkpoint for evaluation")
    parser.add_argument("--batch_size", type=int, help="batch size, if not from config")
    parser.add_argument("--images_dir", type=str, help="path to training images")
    parser.add_argument("--metadata_dir", type=str, help="path to metadata for training images")
    parser.add_argument("--test_run", action='store_true', help="run training with only 5 instances for debugging")
    parser.add_argument("--visualizations", action='store_true', help="save original images and visualized model output")
    parser.add_argument("--output_dir", type=str, help="path to where to save output")
    parser.add_argument("--num_workers", type=int, help="number of workers for dataloader")


    args = parser.parse_args()


    main(args)

