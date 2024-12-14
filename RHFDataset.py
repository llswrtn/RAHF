import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from transformers import T5Tokenizer,  ViTFeatureExtractor
import random
from PIL import Image
import io


class RHFDataset(Dataset):
    def __init__(self, imgs_dir,  metadata_dir, train=False):
        self.imgs_dir = imgs_dir
        self.metadata_dir = metadata_dir
        self.train = train
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        df = pd.read_csv(os.path.join(metadata_dir, 'uids_prompts_promptlabels.csv'))
        self.images = df['uid'].to_list()
        self.captions = df['caption'].to_list()   # Expecting list of texts (strings)
        self.target_heatmaps = np.load(os.path.join(metadata_dir, 'artifact_maps.npy')).astype(np.float32)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

        # Find maximum length of captions
        #self.max_len = max(len(tokens) for tokens in self.tokenized_captions)
        max_length = 256  # 256 is max length of test set, max length of train set is only 57
        '''
        answer to this issue:
        supplementary material page 2: we filter out any image-text pairs that have a text prompt with less than 3 words or more than 30 words
        but the test set was apparently not filtered!
        '''

        self.tokenized_captions = [self.tokenizer(caption, return_tensors="pt", max_length=256, padding='max_length', truncation=True).input_ids.squeeze() for caption in self.captions]


    def random_crop(self, image, heatmap):
        # Randomly sample crop size (80%-100% of width and height)
        crop_scale = random.uniform(0.8, 1.0)
        crop_width = int(crop_scale * image.width)
        crop_height = int(crop_scale * image.height)

        # Randomly sample crop position
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(crop_height, crop_width))

        # Crop image and heatmap
        image = F.crop(image, i, j, h, w)
        heatmap = F.crop(heatmap, i, j, h, w)

        return image, heatmap

    def random_augmentations(self, image):
        # Random brightness, contrast, hue, saturation, and JPEG noise
        image = T.ColorJitter(brightness=0.05, contrast=(0.8, 1.0), hue=0.025, saturation=(0.8, 1.0))(image)
        if random.random() < 0.5:
            quality = random.randint(70, 100)
            image = self.apply_jpeg_noise(image, quality)
        return image

    def apply_jpeg_noise(self, image, quality):
        # Apply JPEG noise by saving and reloading the image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.imgs_dir, self.images[idx]+ '.png')
        original_img_name = self.images[idx]
        #image = read_image(img_path)
        #resize_img = T.Resize(size = (224,224)) # images are 512x512 up to  768x768, target heatmaps 512x512 (!)
        #image = resize_img(image)

        image = Image.open(img_path).convert("RGB")  # Use PIL for augmentation compatibility


        target_heatmap = self.target_heatmaps[idx]
        target_heatmap = (target_heatmap / 255).squeeze(-1) # normalize target to values between 0 and 1
        target_heatmap = torch.from_numpy(target_heatmap)
        target_heatmap = target_heatmap.unsqueeze(0)

        # Resize both image and heatmap to 224x224
        resize_img = T.Resize(size = (224,224)) # images are 512x512 up to  768x768, target heatmaps 512x512 (!)
        image = resize_img(image)
        resize_heatmap = T.Resize(size = (224,224), interpolation=InterpolationMode.NEAREST)
        target_heatmap = resize_heatmap(target_heatmap)


        if random.random() < 0.5 and self.train:
            image, target_heatmap = self.random_crop(image, target_heatmap)

        # 10% chance to apply random augmentations
        if random.random() < 0.1  and self.train:
            image = self.random_augmentations(image)

        if random.random() < 0.1  and self.train:
            image = T.Grayscale(num_output_channels=3)(image)

        #image = self.feature_extractor(images= read_image(img_path), return_tensors="pt")

        image = self.feature_extractor(images= image, return_tensors="pt")
        image = image['pixel_values'].squeeze(0)

        text_inputs = self.tokenized_captions[idx]

        return image, text_inputs, target_heatmap, original_img_name