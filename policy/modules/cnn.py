import torch
import torch.nn as nn
import torch.nn.functional as F    
        
class Image_Encoder(nn.Module):
    def __init__(self, args):
        super(Image_Encoder, self).__init__()
        if args.merge_vis:
            T_kernel_size = 1
        else:
            T_kernel_size = 2
        
        self.img_width = args.img_width
        self.img_height = args.img_height
        
        self.group = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(T_kernel_size, 2, 2), stride=(T_kernel_size, 2, 2)),
            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(T_kernel_size, 2, 2), stride=(T_kernel_size, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(T_kernel_size, 2, 2), stride=(T_kernel_size, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1))) # 32 * 1 * 8 * 16
        
        self.group_linear = nn.Sequential(
            nn.Linear(32 * 1 * int(self.img_height / 8) * int(self.img_width / 8), 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )

    def forward(self, image):
        """
        input:   image sequence          [batch_size, C, T, H, W]
        output:  latent image info       [batch_size, 256]
        """
        state = self.group(image)
        state = state.view(-1, 32 * int(self.img_height / 8) * int(self.img_width / 8))
        state = self.group_linear(state)

        return state
