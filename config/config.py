import os
import torch.distributed as dist
import torch

class Config:
    def __init__(self):
        # General
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Default to GPU if available
        self.num_gpus = int(os.getenv('NUM_GPU', torch.cuda.device_count()))  # Number of GPUs to use
        self.in_channels = 1  # Number of input channels (for grayscale)
        self.num_classes = 3  # Number of output classes (segmentation classes)
        self.input_size = (32, 512, 512)  # Fixed size input patches for the model
        self.augmented_samples = 2 # If > 1 then augmnetation is done else original is fed

        # Model Architecture
        self.level_channels = [64, 128, 256, 512]  # Channel sizes for U-Net encoder layers
        self.bottleneck_channel = 1024  # Bottleneck layer channel size
        
        # Training Settings
        self.batch_size = int(os.getenv('BATCH_SIZE', 4))  # Batch size, can be passed via environment variables
        self.epochs = int(os.getenv('EPOCHS', 100))  # Number of epochs for training
        self.learning_rate = float(os.getenv('LR', 1e-3))  # Learning rate
        self.patch_size = 32  # Patch depth for 3D data
        self.patch_overlap = 16  # Overlap between patches along the depth axis
        self.mixed_precision = bool(int(os.getenv('MIXED_PRECISION', 1)))  # Mixed precision training (1=True, 0=False)
        self.optimizer = os.getenv('OPTIMIZER', 'adam')  # Optimizer type
        
        # Data paths (fixed for your project)
        self.preprocessed_dir = '/shared/home/xvoice/nirmal/data/Task07_Pancreas/Preprocessed'
        
        self.dataset_json = os.getenv('DATASET_JSON', '/shared/home/xvoice/nirmal/data/Task07_Pancreas/dataset.json')
        self.dataset_path = os.getenv('DATASET_PATH', '/shared/home/xvoice/nirmal/data/Task07_Pancreas')
        self.checkpoint_dir = os.getenv('CKPT_DIR', '/shared/home/xvoice/nirmal/exp/3d-unet/checkpoints')
        self.log_dir = os.getenv('LOG_DIR', '/shared/home/xvoice/nirmal/exp/3d-unet/log-common')

        # Distributed Training Settings
        self.distributed = dist.is_available() and dist.is_initialized()
        # self.world_size = self.get_world_size()  # Dynamically find world size
        # self.local_rank = self.get_local_rank()  # Dynamically find local rank

        # # Initialize the process group if distributed training is set
        # if self.distributed:
        #     self.init_distributed()

        
    def init_distributed(self):
        """ Initialize the process group for distributed training."""
        dist.init_process_group(backend='nccl', init_method='env://', world_size=self.get_world_size(), rank=self.get_local_rank())

    def update(self, args):
        """ Update config dynamically using parsed arguments (if needed) """
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_world_size(self):
        return dist.get_world_size() if self.distributed else 1

    def get_local_rank(self):
        return dist.get_rank() if self.distributed else 0
