import torch
import torch.nn as nn


class RAHFHeatmapModel(nn.Module):
    def __init__(self, heatmap_predictor, t5_text_encoder, t5_encoder, vit_model, heatmap_size=224):
        super(RAHFHeatmapModel, self).__init__()

        self.heatmap_predictor = heatmap_predictor
        self.t5_text_encoder = t5_text_encoder
        self.t5_encoder = t5_encoder
        self.vit_model = vit_model

        self.heatmap_size = heatmap_size


    def forward(self, images, texts):
        # Step 1: Extract Image Tokens with ViT
        vit_outputs = self.vit_model(pixel_values=images)
        image_tokens = vit_outputs.last_hidden_state  # Shape: (batch_size, 196, 768)

        # Step 2: Extract Text Tokens with T5 (FROZEN for now)
        with torch.no_grad():
            text_tokens = self.t5_text_encoder(input_ids=texts).last_hidden_state  # Shape: (batch_size, n_text_tokens, 768)

        # Step 3: Concatenate Image and Text Tokens
        fused_tokens = torch.cat([image_tokens, text_tokens], dim=1)  # Shape: (batch_size, 196 + n_text_tokens, 768)

        # Step 4: Encode Fused Tokens with T5 Encoder
        fused_encoded_tokens = self.t5_encoder(inputs_embeds=fused_tokens).last_hidden_state  # Shape: (batch_size, seq_len, 768)

        # Step 5: Extract Image Tokens and Reshape to Feature Map

        image_tokens_encoded = fused_encoded_tokens[:, :196, :]  # Shape: (batch_size, 196, 768)

        patch_size = 16
        image_size = self.heatmap_size
        num_patches_side = image_size // patch_size

        image_feature_map = image_tokens_encoded.view(images.size(0), num_patches_side, num_patches_side, 768).permute(0, 3, 1, 2)  # (batch_size, 768, 14, 14)
        #print(image_feature_map.size())
        # Step 6: Heatmap Prediction
        predicted_heatmap = self.heatmap_predictor(image_feature_map)  # Shape: (batch_size, 1, 224, 224)


        return predicted_heatmap