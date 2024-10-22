

import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from transformers import T5Tokenizer,  ViTFeatureExtractor

class RHFDataset(Dataset):
    def __init__(self, imgs_dir,  metadata_dir):
        self.imgs_dir = imgs_dir
        self.metadata_dir = metadata_dir
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


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.imgs_dir, self.images[idx]+ '.png')
        #image = read_image(img_path)
        #resize_img = T.Resize(size = (224,224)) # images are 512x512 up to  768x768, target heatmaps 512x512 (!)
        #image = resize_img(image)
        image = self.feature_extractor(images= read_image(img_path), return_tensors="pt")
        image = image['pixel_values'].squeeze(0)



        target_heatmap = self.target_heatmaps[idx]
        target_heatmap = target_heatmap.squeeze(-1)
        target_heatmap = torch.from_numpy(target_heatmap)
        target_heatmap = target_heatmap.unsqueeze(0)
        resize_heatmap = T.Resize(size = (224,224), interpolation=InterpolationMode.NEAREST)
        target_heatmap = resize_heatmap(target_heatmap)


        text_inputs = self.tokenized_captions[idx]

        return image, text_inputs, target_heatmap