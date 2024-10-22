import torch.nn as nn

class HeatmapPredictor(nn.Module):
    def __init__(self):
        super(HeatmapPredictor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)

        # output_padding needed to achieve correct output size
        self.deconv1 = nn.ConvTranspose2d(in_channels=384, out_channels=768, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=768, out_channels=384, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=3, stride=2, padding=1, output_padding=1)


        # Readout layers
        self.readout1_1 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.readout1_2 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)

        self.readout2_1 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.readout2_2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)

        self.readout3_1 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.readout3_2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)

        self.readout4_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.readout4_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)


        # LayerNorm sizes after each layer
        self.layer_norm_conv = nn.LayerNorm([384, 14, 14])  # After conv layers
        self.layer_norm_deconv1 = nn.LayerNorm([768, 28, 28])  # After deconv1
        self.layer_norm_deconv2 = nn.LayerNorm([384, 56, 56])  # After deconv2
        self.layer_norm_deconv3 = nn.LayerNorm([384, 112, 112])  # After deconv3
        self.layer_norm_deconv4 = nn.LayerNorm([192, 224, 224])  # After deconv4
        self.relu = nn.ReLU()


        self.final_readout1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.final_readout2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map):

        x = self.conv1(feature_map)
        x = self.layer_norm_conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.layer_norm_conv(x)
        x = self.relu(x)



        x = self.deconv1(x)
        x = self.layer_norm_deconv1(x)
        x = self.relu(x)
        x = self.readout1_1(x) # layer norm also after readout??
        x = self.relu(x)
        x = self.readout1_2(x)
        x = self.relu(x)



        x = self.deconv2(x)
        x = self.layer_norm_deconv2(x)
        x = self.relu(x)
        x = self.readout2_1(x)
        x = self.relu(x)
        x = self.readout2_2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.layer_norm_deconv3(x)
        x = self.relu(x)
        x = self.readout3_1(x)
        x = self.relu(x)
        x = self.readout3_2(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.layer_norm_deconv4(x)
        x = self.relu(x)
        x = self.readout4_1(x)
        x = self.relu(x)
        x = self.readout4_2(x)
        x = self.relu(x)

        x = self.final_readout1(x)
        x = self.relu(x)
        x = self.final_readout2(x)

        heat_map = self.sigmoid(x)
        return heat_map