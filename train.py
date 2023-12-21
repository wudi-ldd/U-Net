import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #---------------------------------#
    # Set a random seed to ensure reproducibility of results.
    seed_value=0
    torch.manual_seed(seed_value)
    # If you are using CUDA,
    torch.cuda.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #---------------------------------#
    # Whether to use CUDA
    # Set to False if you don't have a GPU.
    #---------------------------------#
    Cuda = True
    #---------------------------------------------------------------------#
    # distributed
    # Used to specify whether to run on a single machine with multiple GPUs.
    # Terminal commands are supported on Ubuntu only. Use CUDA_VISIBLE_DEVICES to specify GPUs on Ubuntu.
    # Windows systems use DP mode by default and do not support DDP.

    # DP mode:
    # Set distributed = False
    # In the terminal, run: CUDA_VISIBLE_DEVICES=0,1 python train.py

    # DDP mode:
    # Set distributed = True
    # In the terminal, run: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    # sync_bn
    # Whether to use sync_bn (synchronized batch normalization).
    # Available for multi-GPU setups in DDP mode.
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    # fp16
    # Whether to use mixed precision training.
    # It can reduce memory usage by approximately half, and it requires PyTorch version 1.7.1 or higher.
    #---------------------------------------------------------------------#
    fp16            = True
    #-----------------------------------------------------#
    # num_classes
    # Must be modified for training on your own dataset.
    # Set it to the number of desired classes in your dataset plus 1 (e.g., 2 for your own dataset).
    #-----------------------------------------------------#
    num_classes = 2
    #-----------------------------------------------------#
    # Backbone network selection
    # Choose from:
    # - vgg
    # - resnet50
    #-----------------------------------------------------#
    backbone    = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    # pretrained
    # Whether to use pre-trained weights for the backbone network. These weights are loaded during model construction.
    # If model_path is set, the weights of the backbone do not need to be loaded, and the value of pretrained is irrelevant.
    # If model_path is not set and pretrained is True, only the backbone is loaded, starting from pre-trained weights.
    # If model_path is not set, pretrained is False, and Freeze_Train is False, training starts from scratch, and the backbone is not frozen.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained  = True
    #----------------------------------------------------------------------------------------------------------------------------#
    # For downloading weight files, please refer to the README, and you can download them from the provided cloud storage link. The pre-trained weights for the model's backbone network are generally universal across different datasets because the features are common.
    # The pre-trained weights, especially for the backbone feature extraction network, are crucial as they aid in feature extraction. In most cases, pre-trained weights must be used. Without them, the backbone's weights are too random, resulting in unclear feature extraction, and the training results may not be satisfactory.
    # When training your own dataset, it's normal to encounter dimension mismatches because the predictions will be different, leading to dimension mismatches.
    # If you interrupt the training process, you can set model_path to the weight file in the logs folder to reload the partially trained weights. Additionally, modify the parameters for the Freeze_Train or Unfreeze_Train stages to ensure the model's continuity across epochs.
    # When model_path is set to '', it means that no weights for the entire model are loaded.
    # The weights used here are for the entire model, and they are loaded in train.py. The pretrain option does not affect the loading of weights here.
    # If you want the model to start training from the pre-trained weights of the backbone, set model_path = '', pretrain = True. This loads only the backbone.
    # If you want to train the model from scratch, set model_path = '', pretrain = False, and Freeze_Train = False. This means training from scratch without freezing the backbone.
    # Generally, training a network from scratch yields poor results because the weights are too random, resulting in unclear feature extraction. Therefore, it is strongly discouraged to start training from scratch. If you must do so, consider training a classification model on the ImageNet dataset first, obtaining weights for the network's backbone, and then fine-tuning for your specific task based on those weights.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = ""
    #-----------------------------------------------------#
    # input_shape
    # The size of input images, which should be a multiple of 32.
    #-----------------------------------------------------#
    input_shape = [768, 1024]
    
    #----------------------------------------------------------------------------------------------------------------------------#
   # Training is divided into two stages: the freezing stage and the unfreezing stage. The freezing stage is designed to accommodate users with limited machine performance.

    # For users with limited GPU resources:
    # - You can set Freeze_Epoch equal to UnFreeze_Epoch, and it will only perform freezing training.
    # - You can consider adjusting the following parameters based on your specific requirements:

    ## Training from pre-trained weights for the entire model:
    #   - Adam:
    #     - Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-4, weight_decay = 0 (Freezing stage)
    #     - Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-4, weight_decay = 0 (Unfreezing stage)
    #   - SGD:
    #     - Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 1e-4 (Freezing stage)
    #     - Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 1e-4 (Unfreezing stage)
    #   - You can adjust UnFreeze_Epoch between 100-300.

    ## Training starting from the pre-trained weights of the backbone:
    #   - Adam:
    #     - Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-4, weight_decay = 0 (Freezing stage)
    #     - Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-4, weight_decay = 0 (Unfreezing stage)
    #   - SGD:
    #     - Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 120, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 1e-4 (Freezing stage)
    #     - Init_Epoch = 0, UnFreeze_Epoch = 120, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 1e-4 (Unfreezing stage)
    #   - You can adjust UnFreeze_Epoch between 120-300.

    # Regarding batch size:
    # - Use the largest batch size that your GPU can handle without running out of memory. If you encounter out-of-memory errors (OOM or CUDA out of memory), consider reducing the batch size.
    # - Keep in mind that for resnet50, batch_size should not be set to 1, as it contains BatchNormalization layers.
    # - Typically, Freeze_batch_size is recommended to be 1-2 times Unfreeze_batch_size. Avoid setting a large gap between them, as it affects the automatic adjustment of learning rates.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    # Training Parameters for the Freezing Stage
    # During this stage, the model's backbone is frozen, and the feature extraction network remains unchanged. It consumes less GPU memory and fine-tunes the network.

    # - Init_Epoch: The starting epoch of the training in this stage. Its value can be greater than Freeze_Epoch. For example:
    #   - Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    #   - This will skip the freezing stage and start from epoch 60, adjusting the corresponding learning rates.
    #   - (Used for resuming training from a checkpoint)

    # - Freeze_Epoch: The number of epochs for freezing training of the model.
    #   - (Becomes inactive when Freeze_Train = False)

    # - Freeze_batch_size: The batch size used for freezing training of the model.
    #   - (Becomes inactive when Freeze_Train = False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 16
    #------------------------------------------------------------------#
    # Training Parameters for the Unfreezing Stage
    # During this stage, the model's backbone is unfrozen, and the feature extraction network undergoes changes. It consumes more GPU memory, and all network parameters will change.

    # - UnFreeze_Epoch: The total number of epochs for training the model in the unfreezing stage.

    # - Unfreeze_batch_size: The batch size used for training the model after unfreezing.
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 200
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    # Freeze_Train
    # Whether to perform freezing training. By default, the model undergoes freezing training first and then unfreezing training.
    #------------------------------------------------------------------#
    Freeze_Train        = False
    #------------------------------------------------------------------#
    # Other training parameters: learning rate, optimizer, and learning rate scheduling.
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    # Init_lr
    # The initial maximum learning rate for the model.
    # It is recommended to set Init_lr=1e-4 when using the Adam optimizer.
    # It is recommended to set Init_lr=1e-2 when using the SGD optimizer.
    # Min_lr
    # The minimum learning rate for the model, which is typically set to 0.01 times the maximum learning rate by default.
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    # optimizer_type
    # The type of optimizer to use, with options such as adam and sgd.
    # It is recommended to set Init_lr=1e-4 when using the Adam optimizer.
    # It is recommended to set Init_lr=1e-2 when using the SGD optimizer.

    # momentum
    # The momentum parameter used internally by the optimizer.

    # weight_decay
    # Weight decay, which helps prevent overfitting.
    # Note that using weight_decay with Adam optimizer can lead to errors; it is recommended to set it to 0 when using Adam.
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    # lr_decay_type
    # The type of learning rate decay to use, with options such as 'step' and 'cos'.
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    # save_period
    # How often to save the model weights, specified in terms of epochs.
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    # save_dir
    # The directory where model weights and log files are saved.
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    # eval_flag
    # Whether to perform evaluation during training, with the evaluation being done on the validation set.

    # eval_period
    # How often to perform evaluation, specified in terms of epochs. Frequent evaluation is not recommended, as it can slow down training due to the time-consuming nature of the process.

    # The mAP obtained here may differ from the one obtained using get_map.py for two reasons:
    # (1) The mAP obtained here is for the validation set.
    # (2) The evaluation parameters set here are conservative to speed up the evaluation process.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5
    #------------------------------#
    # Dataset path
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #------------------------------------------------------------------#
    # Recommended options:
    # When there are few classes (a few categories), set it to True.
    # When there are many classes (tens of categories) and the batch_size is relatively large (above 10), set it to True.
    # When there are many classes (tens of categories) and the batch_size is relatively small (below 10), set it to False.
    #------------------------------------------------------------------#
    dice_loss       = True
    #------------------------------------------------------------------#
    # Whether to use focal loss to address class imbalance between positive and negative samples.
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    # Whether to assign different loss weights to different classes. The default is balanced.
    # If you choose to set different weights, make sure to use numpy arrays with a length equal to num_classes. For example:
    # num_classes = 3
    # cls_weights = np.array([1, 2, 3], dtype=np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    # num_workers
    # Used to control whether to use multithreading for data loading. Setting it to 1 disables multithreading.
    # Enabling multithreading can speed up data loading but may consume more memory. In some cases, enabling multithreading in Keras can actually slow down the process significantly.
    # It's recommended to enable multithreading when I/O is the bottleneck, meaning GPU computation is much faster than image loading.
    #------------------------------------------------------------------#
    num_workers     = 8

    #------------------------------------------------------#
    # Setting the GPUs used.
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    #----------------------------------------------------#
    # Download pre-trained weights.
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        # Please refer to the README for the weight files.
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        # Load the weights based on the keys of the pre-trained weights and the model's keys.
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        # Display keys that do not have a match.
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mFriendly reminder: It's normal for the head section not to be loaded. If the Backbone section is not loaded, it's an error.\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    # Torch 1.2 does not support AMP (Automatic Mixed Precision). 
    # It is recommended to use Torch 1.7.1 or higher for correct usage of fp16. Therefore, the error "could not be resolved" may occur with Torch 1.2.
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    # Synchronize Batch Normalization across multiple GPUs.
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            # Run in parallel across multiple GPUs.
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    # Read the text file corresponding to the dataset.
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    #------------------------------------------------------#
    # The backbone feature extraction network is commonly used and freezing it during training can speed up training and prevent weight corruption in the early stages.

    # - `Init_Epoch` is the starting epoch.
    # - `Interval_Epoch` is the number of epochs for frozen training.
    # - `Epoch` is the total number of training epochs.

    # If you encounter OOM (Out of Memory) errors or insufficient GPU memory, consider reducing the batch size.
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        # Freeze a certain portion for training.
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()
            
        #-------------------------------------------------------------------#
        # If you don't want to freeze training, simply set the batch_size to Unfreeze_batch_size.
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        # Adaptive learning rate adjustment based on the current batch size.
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
       # Choose the optimizer based on the optimizer_type.
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        # Obtain the formula for learning rate decay.
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        # Determine the length (number of batches or samples) of each epoch.
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please consider augmenting or expanding your dataset to improve the model's performance.")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)
        
        #----------------------#
        # Record the MAP (Mean Average Precision) curve for evaluation.
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        # Start model training.
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            # If the model has a frozen learning part, then unfreeze it and set the parameters.
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                # Adaptively adjust the learning rate based on the current batch size.
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                # Obtain the learning rate decay formula.
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training. Please consider augmenting or expanding your dataset to improve the model's performance.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
