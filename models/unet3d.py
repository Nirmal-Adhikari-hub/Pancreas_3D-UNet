from torch import nn
import torch.amp
from torchsummary import summary
import torch
import time



class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False, fix_depth=False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        
        self.fix_depth = fix_depth  # To stop downsampling along the depth axis

        if not bottleneck:
            # Adjust pooling kernel to avoid depth downsampling after the 4th layer
            if fix_depth:
                self.pooling = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # No downsampling along depth
            else:
                self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # Normal downsampling otherwise


    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res




class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None, fix_depth=False) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'

        # Up-convolution without depth upsampling if fix_depth is True
        if fix_depth:
            self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        else:
            self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: 
            out = torch.cat((out, residual), 1)
            print(f"Concat: {out.shape}")
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out
        



class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, config) -> None:
        super(UNet3D, self).__init__()
        self.config = config
        self.in_channels = self.config.in_channels
        self.spatial_downsampling = self.config.spatial_downsampling
        self.depth_downsampling = self.config.depth_downsampling
        self.level_channels = self.config.level_channels
        self.bottleneck_channel = self.config.bottleneck_channel
        self.num_classes = self.config.num_classes

        # Down-Sampling
        self.down_convs = nn.ModuleList()
        previous_channels = self.in_channels
        for i, channels in enumerate(self.level_channels):
            fix_depth = self.depth_downsampling[i] == self.depth_downsampling[i+1]
            self.down_convs.append(Conv3DBlock(previous_channels, channels, fix_depth=fix_depth))
            previous_channels = channels

        # Bottleneck
        self.bottleNeck = Conv3DBlock(in_channels=previous_channels, out_channels=self.bottleneck_channel, bottleneck= True)
        previous_channels = self.bottleneck_channel

        # Up-Sampling layers (decoder)
        self.upconvs = nn.ModuleList()
        for i, channels in enumerate(reversed(self.level_channels)):
            last_layer = i == len(self.level_channels) - 1
            num_classes = self.num_classes if last_layer else None
            # print(f"Last layer: {last_layer}, Num_classes: {num_classes}")
            fix_depth = self.depth_downsampling[-i-1] == self.depth_downsampling[-i-2]
            self.upconvs.append(UpConv3DBlock(in_channels=previous_channels, res_channels=channels, last_layer=last_layer, num_classes=num_classes, fix_depth=fix_depth))
            previous_channels = channels

    
    def forward(self, x):
        # Encoder path
        residual_out = []
        for down in self.down_convs:
            x, res = down(x)
            residual_out.append(res)
            print(f"X: {x.shape}, Res: {res.shape}")

        x, _ = self.bottleNeck(x)
        print(f"Bottleneck: {x.shape}")

        # Decoder path (reverse order)
        for ups in self.upconvs:
            res = residual_out.pop()
            x = ups(x, res)
            print(f"Decoder: {x.shape}")
        return x


if __name__ == '__main__':

    import os
    import sys
    import torch.cuda as cuda
    from torch.cuda.amp import autocast, GradScaler
    scaler = torch.GradScaler("cuda")

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import Config

    config = Config()

    def print_memory_usage(device):
        """Helper function to print memory usage stats."""
        current_memory = cuda.memory_allocated(device)
        max_memory = cuda.max_memory_allocated(device)
        total_memory = cuda.get_device_properties(device).total_memory

        print(f"Current GPU memory usage: {current_memory / (1024**3):.2f} GB")
        print(f"Max GPU memory usage: {max_memory / (1024**3):.2f} GB")
        print(f"Total GPU memory: {total_memory / (1024**3):.2f} GB")

    #Configurations according to the Xenopus kidney dataset
    model = UNet3D(config)
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    torch.cuda.empty_cache()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallelism")
        model = nn.DataParallel(model)
    model = model.to(device)
    input_tensor = torch.randn(1, 1, 64, 512, 512).to(device)

    try:
        with torch.autocast(device_type=device):
            output = model(input_tensor)

        # Print the memory stats after model execution
        print_memory_usage(device)

        print(f"Output shape: {output.shape}")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"Caught out of memory error!")
            print_memory_usage(device)
        else:
            raise e

    # summary(model=model, input_size=(1, 128, 512, 512), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))





