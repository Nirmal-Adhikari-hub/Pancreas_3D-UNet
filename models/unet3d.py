from torch import nn
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
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)
        
        self.fix_depth = fix_depth  # To stop downsampling along the depth axis

        if not bottleneck:
            # Adjust pooling kernel to avoid depth downsampling after the 4th layer
            if fix_depth:
                self.pooling = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # No downsampling along depth
            else:
                self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # Normal downsampling otherwise


    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        # print(f"First Con: {res.shape}")
        res = self.relu(self.bn2(self.conv2(res)))
        # print(f"Second Con: {res.shape}")
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
            # print(f"After Pool: {out.shape}")
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

        # self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        # print(f"First Upconv: {out.shape}")
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        # print(f"Decode Conv: {out.shape}")
        out = self.relu(self.bn(self.conv2(out)))
        # print(f"DEcoder Conv: {out.shape}")
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
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256, 512], bottleneck_channel=1024) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls, level_4_chnls = level_channels[0], level_channels[1], level_channels[2], level_channels[3]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls, fix_depth=True)
        self.a_block4 = Conv3DBlock(in_channels=level_3_chnls, out_channels=level_4_chnls, fix_depth=True)

        self.bottleNeck = Conv3DBlock(in_channels=level_4_chnls, out_channels=bottleneck_channel, bottleneck= True)

        self.s_block4 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_4_chnls, fix_depth=True)
        self.s_block3 = UpConv3DBlock(in_channels=level_4_chnls, res_channels=level_3_chnls, fix_depth=True)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    
    def forward(self, input):
        #Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        # print(out.shape)
        out, residual_level2 = self.a_block2(out)
        # print(out.shape)
        out, residual_level3 = self.a_block3(out)
        # print(out.shape)
        out, residual_level4 = self.a_block4(out)
        # print(out.shape)

        out, _ = self.bottleNeck(out)
        # print(f"Bottlencek: {out.shape}")

        #Synthesis path forward feed
        out = self.s_block4(out, residual_level4)
        # print(out.shape)
        out = self.s_block3(out, residual_level3)
        # print(out.shape)
        out = self.s_block2(out, residual_level2)
        # print(out.shape)
        out = self.s_block1(out, residual_level1)
        # print(out.shape)
        return out


if __name__ == '__main__':
    #Configurations according to the Xenopus kidney dataset
    model = UNet3D(in_channels=1, num_classes=3)
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    torch.cuda.empty_cache()
    model = model.to(device)
    input_tensor = torch.randn(1, 1, 32, 64, 64).to(device)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    # summary(model=model, input_size=(1, 128, 512, 512), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))





