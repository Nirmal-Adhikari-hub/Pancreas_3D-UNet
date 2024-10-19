
# """""
# Dataset configurations:
#     :param DATASET_PATH -> the directory path to dataset .tar files
#     :param TASK_ID -> specifies the the segmentation task ID (see the dict below for hints)
#     :param IN_CHANNELS -> number of input channels
#     :param NUM_CLASSES -> specifies the number of output channels for dispirate classes
#     :param BACKGROUND_AS_CLASS -> if True, the model treats background as a class

# """""
# DATASET_PATH = '/PATH/TO/THE/DATASET'
# TASK_ID = 9
# IN_CHANNELS = 1
# NUM_CLASSES = 3
# BACKGROUND_AS_CLASS = False


# """""
# Training configurations:
#     :param TRAIN_VAL_TEST_SPLIT -> delineates the ratios in which the dataset shoud be splitted. The length of the array should be 3.
#     :param SPLIT_SEED -> the random seed with which the dataset is splitted
#     :param TRAINING_EPOCH -> number of training epochs
#     :param VAL_BATCH_SIZE -> specifies the batch size of the training DataLoader
#     :param TEST_BATCH_SIZE -> specifies the batch size of the test DataLoader
#     :param TRAIN_CUDA -> if True, moves the model and inference onto GPU
#     :param BCE_WEIGHTS -> the class weights for the Binary Cross Entropy loss
# """""
# TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
# SPLIT_SEED = 42
# TRAINING_EPOCH = 100
# TRAIN_BATCH_SIZE = 1
# VAL_BATCH_SIZE = 1
# TEST_BATCH_SIZE = 1
# TRAIN_CUDA = True
# BCE_WEIGHTS = [0.004, 0.996]




import os
import torch.distributed as dist

class Config:
    def __init__(self):
        # General
        self.in_channels = 1  # Number of input channels (for grayscale)
        self.num_classes = 3  # Number of output classes (segmentation classes)
        self.input_size = (32, 512, 512)  # Fixed size input patches for the model

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
        self.dataset_json = os.getenv('DATASET_JSON', 'D:/Nirmal/pancreas/Task07_Pancreas/Task07_Pancreas/dataset.json')
        self.dataset_path = os.getenv('DATASET_PATH', 'D:/Nirmal/pancreas/Task07_Pancreas/Task07_Pancreas')
        self.checkpoint_dir = os.getenv('CKPT_DIR', 'D:/Nirmal/pancreas/3D-UNet/checkpoints')
        self.log_dir = os.getenv('LOG_DIR', 'D:/Nirmal/pancreas/3D-UNet/logs')

        # Distributed Training Settings
        self.world_size = self.get_world_size()  # Dynamically find world size
        self.local_rank = self.get_local_rank()  # Dynamically find local rank

    def update(self, args):
        """ Update config dynamically using parsed arguments (if needed) """
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_world_size(self):
        """ Dynamically find the total number of processes for distributed training """
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1  # If not distributed, return 1 as default

    def get_local_rank(self):
        """ Dynamically find the local rank (process ID within a node) """
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0  # Return 0 if not in distributed mode
