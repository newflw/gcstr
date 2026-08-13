import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import scipy.stats as stats
import os
import copy
import shutil
from collections import defaultdict
import json
import time
import torchvision.transforms as transforms
from dotenv import load_dotenv

import log
from frame_dataset import FrameDataset, SubsetFrameDataset, create_cached, dataset_names
from dimension import get_dimension_channel_count
from preprocess import preprocess_dataset
from models import EnsemblePaSNetModel, EnsemblePaSNetModelStream, FeatureFusionPaSNetModel, AdaptedVisionTransformer
import util


# Supported model architectures.
# "merged_*" are single networks operating on concatenated dimensions.
# "pasnet*" variants are ensembles / feature-fusion models.
architecture_names = [
    'merged_vitb16', 
    'merged_mobilenetv3l', 'merged_mobilenetv3s',
    'merged_inceptionv3',
    'merged_resnet18', 'merged_resnet34', 'merged_resnet50',
    'pasnetff_resnet18', 'pasnetff_resnet34', 'pasnetff_resnet50', 
    'pasnetensum_resnet18', 'pasnetensum_resnet34', 'pasnetensum_resnet50',
    'pasnetenprobsum_resnet18', 
    'pasnetenprobmul_resnet18', 'pasnetenprobmul_resnet34', 'pasnetenprobmul_resnet50', 'pasnetenprobmul_mobilenetv3s', 'pasnetenprobmul_mobilenetv3l', 'pasnetenprobmul_inceptionv3',
]

def parse_args():
    parser = argparse.ArgumentParser(description="Main entry point for training and testing models and dimensions for the Generalized CSTR article 2025")

    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=dataset_names,
        default="daily_dvs_200", 
        help=f"Dataset to use. One of: {', '.join(dataset_names)}"
    )
    parser.add_argument(
        '--action', 
        type=str, 
        choices=['train', 'train_n', 'gif', 'dimensions', 'mp4', 'clear_image_cache', 'clear_augmentations', 'summarize_runs', 'test_augmentations', 'generate_augmentations', 'check', 'time_test', 'memory_test'],
        default="train_n",
        help="Main action"
    )
    parser.add_argument(
        '--result_folder', 
        type=str,         
        default="./results/general",
        help="Result folder"
    )    
    parser.add_argument(
        '--output_folder', 
        type=str,         
        default="./output",
        help="Output folder"
    )    
    parser.add_argument(
        '--evaluation_method', 
        type=str,
        choices=['test_split', 'k_fold_cross_validation'],
        default="test_split",
        help="Evaluation method"
    )    
    parser.add_argument(
        '--transform', 
        type=str,
        choices=['no_transform', 'imagenet_norm'],
        default="no_transform",
        help="Image transform type"
    )    
    parser.add_argument(
        '--save_model', 
        choices=['all', 'submodels_only'],
        type=str, 
        default="all",
        help="Result folder"
    )    
    parser.add_argument(
        '--media_count', 
        type=int,
        default=10,
        help="Media (eg. GIF/MP4/PNG) count to output per label"
    )
    parser.add_argument(
        '--media_event_size', 
        type=int,
        default=3,
        help="Media event size"
    )
    parser.add_argument(
        '--media_flip_x', 
        type=int,
        default=0,
        help="Media flip X"
    )
    parser.add_argument(
        '--media_flip_y', 
        type=int,
        default=0,
        help="Media flip Y"
    )
    parser.add_argument(
        '--media_frame_count', 
        type=int,
        default=-1,
        help="Frames to generate for the media. -1 means adapt to media_fps instead"
    )
    parser.add_argument(
        '--media_fps', 
        type=int,
        default=30,
        help="Framerate for media. Only applied if media_frame_count == -1"
    )
    parser.add_argument(
        '--run_index', 
        type=int, 
        default=0,
        help="Run index"
    )    
    parser.add_argument(
        '--use_image_cache',
        type=int, 
        default=1,
        help="Enable image cache"
    )
    parser.add_argument(
        '--use_sub_image_cache',
        type=int, 
        default=0,
        help="Enable sub image cache"
    )
    parser.add_argument(
        '--preprocess_mode', 
        type=str, 
        choices=["serial", "parallel"],
        default="serial",
        help=""
    )
    parser.add_argument(
        '--early_stopping_patience', 
        type=int, 
        default=10,
        help="Early stopping patience. Set to 0 for no early stopping"
    )
    parser.add_argument(
        '--validation_percent', 
        type=int, 
        default=20,
        help="Percent of training data to use for validation"
    )
    parser.add_argument(
        '-W', '--W', 
        type=int, 
        default=224,
        help="Input image tensor width"
    )
    parser.add_argument(
        '-H', '--H',
        type=int, 
        default=224,
        help="Input image tensor height"
    )
    parser.add_argument(
        '-T', '--T',
        type=int, 
        default=224,
        help="Input image tensor time bins"
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help="Epoch count"
    )
    parser.add_argument(
        '--stop_epoch', 
        type=int, 
        default=0,
        help="Stop epoch (will not be used for submodels)"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        '--finetune_epoch', 
        type=int, 
        default=0,
        help="Finetune epoch"
    )
    parser.add_argument(
        '--finetune_lr', 
        type=float, 
        default=0.0001,
        help="Finetune learning rate"
    )
    parser.add_argument(
        '--augmentation', 
        type=str, 
        default="",
        help="Augmentation"
    )
    parser.add_argument(
        '--augmentation_count', 
        type=int,
        default=5,
        help="Number of augmentations per instance"
    )
    parser.add_argument(
        '--image_cache_mode', 
        type=str, 
        choices=['precalculate', 'lazy'],
        default="lazy",
        help="Image cache mode"
    )
    parser.add_argument(
        '--run_count', 
        type=int,
        default=5,
        help="Number of training runs"
    )
    parser.add_argument(
        '--weights', 
        choices=[None, 'imagenet'],
        type=str,
        default=None,
        help="Starting weights"
    )
    parser.add_argument(
        '--weights_xy', 
        choices=['default', 'no_weights', 'imagenet'],
        type=str,
        default='default',
        help="Starting weights in XY"
    )
    parser.add_argument(
        '--weights_xt', 
        choices=['default', 'no_weights', 'imagenet'],
        type=str,
        default='default',
        help="Starting weights in XT"
    )
    parser.add_argument(
        '--weights_yt', 
        choices=['default', 'no_weights', 'imagenet'],
        type=str,
        default='default',
        help="Starting weights in YT"
    )
    parser.add_argument(
        '--architecture', 
        type=str, 
        choices=architecture_names, 
        default="merged_resnet18",
        help="Model architecture to use"
    )
    parser.add_argument(
        '--lr_scheduler', 
        type=str, 
        choices=['none', 'cosine'], 
        default="none", 
        help="Learning rate scheduler to use. One of: none, cosine"
    )
    parser.add_argument(
        '--dimensions', 
        type=str, 
        default="cstr_3",
        help="Dimensions to use"
    )
    parser.add_argument(
        '--dithering', 
        type=str, 
        choices=['nodithering', 'uniform'], 
        default="uniform",
        help="Dithering method for augmentation"
    )
    parser.add_argument(
        '--on_result_exists', 
        type=str, 
        choices=['train', 'ignore', 'eval'], 
        default="train",
        help="What to do if a result already exists"
    )

    return parser.parse_args()


def sort_dimensions_str(dims_str):
    """
    Sort a semicolon-separated dimension string deterministically.
    """    
    return ";".join(sorted(dims_str.split(";")))

def get_sub_weight_str(args, attr_name):
    """
    Build a suffix string for sub-weights (xy/xt/yt) if set to something
    other than 'default'.
    """
    if hasattr(args, attr_name) and getattr(args, attr_name) != "default":
        return f"{attr_name}{getattr(args, attr_name)}"
    return ""

def get_model_name(args, variant='accuracy'):
    """ 
    Construct a unique model name string based on all important hyperparameters.
    This is used for saving and loading models / JSON logs.
    """
    aug_str = f'_{args.augmentation}' if args.augmentation else ""
    weights_str = f'_{args.weights}' if args.weights else ""
    weights_str += get_sub_weight_str(args, "weights_xy")
    weights_str += get_sub_weight_str(args, "weights_xt")
    weights_str += get_sub_weight_str(args, "weights_yt")
    method_str = ""
    if getattr(args, "evaluation_method") and args.evaluation_method != "test_split" and args.evaluation_method:
        method_str = f"_{args.evaluation_method}"

    transform_str = ""
    if hasattr(args, "transform") and getattr(args, "transform") != "no_transform" and args.weights != "imagenet":
        # When using imagenet weights, we always use imagenet_norm
        transform_str = f"_{args.transform}"
    
    dithering = "uniform"
    if hasattr(args, "dithering"):
        dithering = args.dithering
    dither_str = f'_{dithering}' if dithering != "uniform" else "" # default is uniform dither, then dither_str == ""
    stop_str = f'_se{args.stop_epoch}' if args.stop_epoch != 0 else ""
    finetune_epoch_str = f'_fte{args.finetune_epoch}' if args.finetune_epoch != 0 else ""
    return f"{args.dataset}_{sort_dimensions_str(args.dimensions)}_{args.architecture}_{args.batch_size}_{args.W}x{args.H}x{args.T}_{args.batch_size}_{args.epochs}_{args.lr}_{args.lr_scheduler}_{variant}_{args.validation_percent}{aug_str}{weights_str}{stop_str}{finetune_epoch_str}{dither_str}{transform_str}{method_str}_{args.run_index}"

def get_summary_path(args, variant='accuracy'):
    return f"{util.get_model_folder()}/{get_model_name(args, variant)}_summary.json"

def get_model_path(args, variant='accuracy'):
    return f"{util.get_model_folder()}/{get_model_name(args, variant)}.pth"

def get_json_path(args, variant='accuracy'):
    return f"{util.get_model_folder()}/{get_model_name(args, variant)}.json"


def eval_model(model, loader, criterion, device, num_classes):
    """
    Evaluate a model on a given DataLoader and compute:
      - total correct / total samples
      - accumulated loss (sum over batches)
      - per-label accuracies
      - top-k accuracies (k=2..5 or up to num_classes)
    """
    model.eval()

    correct_per_label = defaultdict(int)
    total_per_label = defaultdict(int)
    
    correct = 0
    total = 0
    loss = 0
    top_k_accuracies = {k: 0 for k in range(2, min(num_classes + 1, 6))}  # Top-2 to Top-5

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()

            # Update the per-label statistics
            for label, prediction in zip(labels, predicted):
                total_per_label[label.item()] += 1
                if label == prediction:
                    correct_per_label[label.item()] += 1

            for k in top_k_accuracies:
                top_k_predictions = torch.topk(outputs, k=k, dim=1).indices
                # Check if the true label is among the top-k predictions
                top_k_correct = (top_k_predictions == labels.view(-1, 1)).any(dim=1).sum().item()
                top_k_accuracies[k] += top_k_correct

    for k in top_k_accuracies:
        top_k_accuracies[k] = 100 * top_k_accuracies[k] / total

    accuracy_per_label = {}
    for label in total_per_label:
        accuracy_per_label[label] = 100 * correct_per_label[label] / total_per_label[label]

    return correct, total, loss, accuracy_per_label, top_k_accuracies



def perform_action(args):
    """
    Main orchestration function.
    Handles:
      - train / train_n
      - cache clearing
      - summarizing multiple runs
      - dataset preprocessing / loading
      - actual training loop
    """
    # Multi-run training: train N runs and then summarize them
    if args.action == "train_n":
        print("Training N runs...")
        for i in range(args.run_count):
            args_copy = copy.deepcopy(args)
            args_copy.run_index = i
            args_copy.action = 'train'
            perform_action(args_copy)
        
        args_copy = copy.deepcopy(args)
        args_copy.action = 'summarize_runs'
        perform_action(args_copy)
        return

    
    batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dimensions = sort_dimensions_str(args.dimensions).split(";")

    test_dataset = None

    use_image_cache = args.use_image_cache != 0
    use_sub_image_cache = args.use_sub_image_cache != 0

    dataset_folder = util.get_dataset_folder(args.dataset)

    # Local helper to clear image cache for a dataset slice (train/test)
    def clear_image_cache(slice):
        for file in Path(f'{dataset_folder}/{slice}/.image_cache').glob('*.npz'):
            if file.is_file():
                file.unlink() 

    # Select train / test slices depending on evaluation method
    train_slice = "train"
    if args.evaluation_method == "k_fold_cross_validation":
        train_slice = f'train_{args.run_index}'
    test_slice = "test"
    if args.evaluation_method == "k_fold_cross_validation":
        test_slice = f'test_{args.run_index}'

    if args.action == "clear_image_cache":
        print("Clearing image cache...")
        clear_image_cache(train_slice)        
        clear_image_cache(test_slice)
        return    

    if args.action == "clear_augmentations":
        print("Clearing augmentations...")
        if args.augmentation:
            # Clearing augmentations
            for file in Path(f'{dataset_folder}/{train_slice}/augmentations').glob(f'*_{args.augmentation}_*.npz'):
                if file.is_file():
                    file.unlink()
            # Clearing image cache for those augmentations
            for file in Path(f'{dataset_folder}/{train_slice}/.image_cache').glob(f'*_{args.augmentation}_*.npz'):
                if file.is_file():
                    file.unlink()
        else:
            print("No augmentation specified")
        return    
    
    # Summarize multiple runs (train_n)
    if args.action == "summarize_runs":
        print("Summarizing runs...")
        os.makedirs(args.result_folder, exist_ok=True)
        accuracy_list = []
        val_accuracy_list = []
        for i in range(args.run_count):
            args_copy = copy.deepcopy(args)
            args_copy.run_index = i
            run_json_path = get_json_path(args_copy)
            shutil.copy2(run_json_path, args.result_folder)
            with open(run_json_path, 'r') as file:
                json_data = json.load(file)
                if 'testAccuracy' not in json_data:
                    print("No testAccuracy found in", file.name)
                    return
                accuracy_list.append(json_data['testAccuracy'])
                    
                if 'validationAccuracy' in json_data:
                    val_accuracy_list.append(json_data['validationAccuracy'])


        # Basic descriptive statistics for the multiple runs
        mean_accuracy = np.mean(accuracy_list)
        std_deviation = np.std(accuracy_list)
        min_accuracy = np.min(accuracy_list)
        max_accuracy = np.max(accuracy_list)
        median_accuracy = np.median(accuracy_list)

        # 95% confidence interval using Student's t-distribution
        confidence_level = 0.95
        degrees_freedom = len(accuracy_list) - 1
        confidence_interval = list(stats.t.interval(confidence_level, degrees_freedom, loc=mean_accuracy, scale=stats.sem(accuracy_list)))

        metrics = {
            "meanAccuracy": mean_accuracy,
            "standardDeviation": std_deviation,
            "minAccuracy": min_accuracy,
            "maxAccuracy": max_accuracy,
            "medianAccuracy": median_accuracy,
            "confidenceInterval95": confidence_interval,
            "args": vars(args),
        }
        if len(val_accuracy_list) > 0:
            metrics["meanValAccuracy"] = np.mean(val_accuracy_list)
            

        summary_path = get_summary_path(args)
        with open(summary_path, 'w') as f:
            json.dump(metrics, f)
        shutil.copy2(summary_path, args.result_folder)

        return

    # If processed dataset is missing, run preprocessing once per dataset
    if not Path(dataset_folder).exists():
        print(f"Could not find processed dataset at {dataset_folder}. Performing a once-per-dataset pre-processing operation that might take some time to complete...")
        preprocess_dataset(args.dataset, args.preprocess_mode)

    # Helper to load the main FrameDataset for the train split
    def load_dataset():
        return FrameDataset(dataset_folder, dimensions, args.W, args.H, args.T, slice=train_slice, use_image_cache=use_image_cache, use_subimage_cache=use_sub_image_cache, dither=args.dithering!="nodithering")

    dataset = load_dataset()
    if len(dataset) == 0:
        print(f"Could not find any instances in dataset folder {dataset_folder}. Performing a once-per-dataset pre-processing operation that might take some time to complete...")
        preprocess_dataset(args.dataset, args.preprocess_mode)
        dataset = load_dataset()
        
    if len(dataset) == 0:
        raise ValueError("Could not find any instances in dataset even after pre-processing")
    
    
    first, _ = dataset[0]
    channels = first.shape[0]

    # Optional normalization (Imagenet) transforms
    train_transform = None
    val_transform = None
    test_transform = None
    if args.weights or args.transform == "imagenet_norm":
        if channels % 3 != 0:
            raise ValueError(f"Channel count must be a multiple of 3 when using pretrained weights! Was {channels}")
        m = channels // 3
        normalize_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406]*m, std=[0.229, 0.224, 0.225]*m),
        ])
        train_transform = normalize_transform
        val_transform = normalize_transform
        test_transform = normalize_transform
    
    test_dataset = FrameDataset(dataset_folder, dimensions, args.W, args.H, args.T, slice=test_slice, use_image_cache=use_image_cache, transform=test_transform, use_subimage_cache=use_sub_image_cache)

    # Sanity check action: detect NaNs/Infs in the dataset
    if args.action == "check":
        n = len(dataset)
        for i, value in enumerate(dataset):
            
            if (i % 1000) == 0:
                print(f'{i} / {n}')
            data, label = value
            contains_nan = torch.isnan(data).any()
            contains_inf = torch.isinf(data).any()
            if contains_nan:
                print(f"{i} contains NaN")
            if contains_inf:
                print(f"{i} contains Inf")
        return

    is_aggregate = "pasnet" in args.architecture
    
    # Optionally precalculate and cache images to NPZ files
    if args.image_cache_mode == 'precalculate':
        if not args.augmentation and not is_aggregate:
            print("Creating cache for main dataset...")
            create_cached(dataset)
        if test_dataset:
            print("Creating cache for test dataset...")
            create_cached(test_dataset)
    
    first, _ = dataset[0]
    channels = first.shape[0]

    num_classes = len(dataset.classes)

    print("Creating test and validation datasets...")
    
    train_folder = f"{dataset_folder}/{train_slice}"
    val_split_path = util.get_validation_set_filename(dataset_folder, args.run_index, args.validation_percent * 0.01)
    
    # Create validation splits if they don't exist
    if not Path(val_split_path).exists():        
        val_fraction = args.validation_percent * 0.01
        if args.evaluation_method == "k_fold_cross_validation":
            # Each val split index is unique since there are multiple test and train sets
            util.create_validation_set(train_folder, util.get_validation_set_filename(dataset_folder, args.run_index, val_fraction), val_fraction)
        else:
            util.create_validation_sets(10, train_folder, dataset_folder, val_fraction)
            
    with open(val_split_path, 'r') as file:
        json_data = json.load(file)

    # Wrap in SubsetFrameDataset with optional augmentation/transform
    train_dataset = SubsetFrameDataset(dataset, dataset_folder, json_data['training'], augmentation=args.augmentation, transform=train_transform)
    val_dataset = SubsetFrameDataset(dataset, dataset_folder, json_data['validation'], transform=val_transform)

    
    if args.action == "generate_augmentations":
        train_dataset.create_augmentations()
        return

    if args.image_cache_mode == 'precalculate' and args.augmentation and not is_aggregate:
        for i in range(args.augmentation_count):
            print(f"Creating cache for train dataset augmentations. Augmentation {i+1}/{args.augmentation_count}")
            train_dataset.forced_aug_index = i
            create_cached(train_dataset)
            train_dataset.forced_aug_index = -1
        print("Creating cache for validation dataset...")
        create_cached(val_dataset)
       
    need_training = True
    finetune_epoch = -1
    
    load_model_data = True
    
    if args.action == "memory_test" or args.action == "time_test":
        load_model_data = False
    
    # Select and construct the model according to architecture
    if args.architecture == "merged_resnet18":
        model = util.get_resnet18(channels, num_classes, args.weights).to(device)        
    elif args.architecture == "merged_resnet34":
        model = util.get_resnet34(channels, num_classes, args.weights).to(device)
    elif args.architecture == "merged_resnet50":
        model = util.get_resnet50(channels, num_classes, args.weights).to(device)
    elif args.architecture == "merged_mobilenetv3l":
        model = util.adapt_mobilenetv3(util.get_mobilenetv3("large", args.weights), channels, num_classes).to(device)
    elif args.architecture == "merged_mobilenetv3s":
        model = util.adapt_mobilenetv3(util.get_mobilenetv3("small", args.weights), channels, num_classes).to(device)
    elif args.architecture == "merged_inceptionv3":
        model = util.get_inception_v3(channels, num_classes, args.weights == "imagenet").to(device)
    elif args.architecture == "merged_vitb16":
        model = AdaptedVisionTransformer(224, num_classes, channels).to(device)
    elif args.architecture == "pasnetensum_resnet18" or args.architecture == "pasnetensum_resnet50":
        need_training = False
        model = get_pasnet_ensemble_model(channels, num_classes, device, False, args, 'accuracy', load_model_data=load_model_data)
    elif args.architecture == "pasnetenprobsum_resnet18" or args.architecture == "pasnetenprobsum_resnet50":
        need_training = False
        model = get_pasnet_ensemble_model(channels, num_classes, device, False, args, 'accuracy', agg_type="prob_sum", load_model_data=load_model_data)
    elif (
         args.architecture == "pasnetenprobmul_resnet18" or args.architecture == "pasnetenprobmul_resnet50" or args.architecture == "pasnetenprobmul_resnet34" or 
         args.architecture == "pasnetenprobmul_mobilenetv3s" or args.architecture == "pasnetenprobmul_mobilenetv3l" or args.architecture == "pasnetenprobmul_inceptionv3"
         ):
        need_training = False
        model = get_pasnet_ensemble_model(channels, num_classes, device, False, args, 'accuracy', agg_type="prob_mul", load_model_data=load_model_data)
    elif args.architecture == "pasnetff_resnet18" or args.architecture == "pasnetff_resnet34":
        model = get_pasnet_feature_fusion_model(channels, num_classes, device, False, args, load_model_data=load_model_data)
        finetune_epoch = args.finetune_epoch
    elif args.architecture == "pasnetff_resnet50":
        model = get_pasnet_feature_fusion_model(channels, num_classes, device, False, args, feature_count=2048, load_model_data=load_model_data)
        finetune_epoch = args.finetune_epoch
        
    if args.weights:
        finetune_epoch = args.finetune_epoch
        util.set_requires_grad(model, False)
        util.set_fc_requires_grad(model, True)

    print(f'Model parameter count: {util.count_parameters(model)}, trainable count: {util.count_trainable_parameters(model)}')
    
    # Note: worker_count is 0 -> no multi-process dataloading
    worker_count = 0

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker_count, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=worker_count, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=worker_count, pin_memory=True)

    if args.action == "memory_test":
        model.eval()

        input_tensor = first.to(device).unsqueeze(0)
        
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output = model(input_tensor)

        print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("Peak:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

        return

    if args.action == "time_test":
        
        model.eval()

        # Disable gradients for inference
        torch.set_grad_enabled(False)
        input_tensor = first.to(device).unsqueeze(0)

        num_warmup = 1000
        for _ in range(num_warmup):
            _ = model(input_tensor)

        if device == "cuda":
            torch.cuda.synchronize()


        num_runs = 10000          
        # -------------------
        # Timed runs
        # -------------------
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()

            _ = model(input_tensor)

            # Important for GPU timing!
            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

        # -------------------
        # Results
        # -------------------
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        print(f"Average inference time: {avg_time * 1000:.3f} ms")
        print(f"Std deviation: {std_time * 1000:.3f} ms")
        print(f"Min: {min(times) * 1000:.3f} ms, Max: {max(times) * 1000:.3f} ms")

        return

    training_epochs = -1
    criterion = nn.CrossEntropyLoss()

    if need_training:
        if args.on_result_exists != "train":
            model_path = get_model_path(args, 'accuracy')
            json_path = get_json_path(args, 'accuracy')
            
            need_training = not Path(model_path).exists() or not Path(json_path).exists()
            if not need_training:
                # Verify that there is a test result in the JSON file
                with open(json_path, 'r') as file:
                    if 'testAccuracy' in json.load(file):
                        if args.on_result_exists == "ignore":
                            print(f"Result aldready exists. Ignoring")
                            return
                        else: # "eval"
                            util.load_model(model, None, get_model_path(args, 'accuracy'))
                            model.to(device)
                            need_training = False

    if need_training:
        
        last_improved_epoch = 0
        
        lr = args.lr
        num_epochs = args.epochs
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        scheduler = None
        
        def update_scheduler():
            """
            Initialize / reinitialize the LR scheduler (called at start and on fine-tune).
            """
            nonlocal scheduler
            if args.lr_scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        
        update_scheduler()
        
        best_accuracy = 0
        best_mean_loss = 1e100

        current_lr = lr

        print("Starting training...")

        quit = False

        epoch_times = []
        epoch_accuracies = []
        patiences = []

        training_epochs = 0

        # Standard supervised training loop
        for epoch in range(num_epochs):
            training_epochs += 1
            
            epoch_start_time = time.time()

            # Switch to finetuning at specified epoch (unfreeze, new LR)
            if finetune_epoch == epoch:                
                util.set_requires_grad(model, True)
                optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr)
                update_scheduler()
                last_improved_epoch = epoch # Reset patience
                print(f'Starting finetuning. Model parameter count: {util.count_parameters(model)}, trainable count: {util.count_trainable_parameters(model)}')

            # Before finetuning, ensure that patience does not trigger early stop
            if finetune_epoch >= 0 and epoch < finetune_epoch:
                last_improved_epoch = epoch
            
            model.train()


            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                model_loss = 0.0

                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                model_loss = loss.item()

                print(f'[Epoch {epoch + 1}, Batch {i + 1}] model loss: {model_loss}')
                           
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]


            correct, total, loss, accuracy_per_label, top_k_accuracies = eval_model(model, valloader, criterion, device, num_classes)

            print(f'Accuracy of the model on the val set: {100 * correct / total:.2f}%')
            if args.lr_scheduler != 'none':
                print(f'Current LR: {current_lr}')
            
            accuracy = 100 * correct / total
            mean_loss = loss / len(valloader)

            epoch_end_time = time.time()

            epoch_times.append(epoch_end_time - epoch_start_time)

            mean_epoch_time = sum(epoch_times) / len(epoch_times)


            epoch_accuracies.append(accuracy)
            data = {
                'validationAccuracy': accuracy,
                'validationMeanLoss': mean_loss,
                'validationLabelAccuracies': accuracy_per_label,
                'validationEpochAccuracies': epoch_accuracies,
                'validationTopKAccuracies': top_k_accuracies,
                'epoch': epoch,
                'epochTimes': epoch_times,
                'patiences': patiences,
                "args": vars(args),
            }

            def save_json(variant):
                with open(get_json_path(args, variant), 'w') as f:
                    json.dump(data, f)

            improved = False
            # Early stop / best model selection criteria:
            #  - improved val accuracy OR
            #  - lower mean val loss
            if accuracy > best_accuracy or mean_loss < best_mean_loss:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                if mean_loss < best_mean_loss:
                    os.makedirs(util.get_model_folder(), exist_ok=True)
                    util.save_model(model, optimizer, epoch, get_model_path(args, 'accuracy'))
                    save_json('accuracy')
                    best_mean_loss = mean_loss
                    best_accuracy = accuracy
                improved = True

            print(f'Best val accuracy: {best_accuracy:.2f}%')

            if improved:
                last_improved_epoch = epoch

            patience_left = 0
            if args.early_stopping_patience > 0:
                if not improved and last_improved_epoch + args.early_stopping_patience <= epoch:
                    print(f'Early stopping triggered. Patience {args.early_stopping_patience}, epoch {epoch}, last improved epoch {last_improved_epoch}')
                    quit = True
                else:
                    patience_left = args.early_stopping_patience - (epoch - last_improved_epoch)
                    print(f'Patience left: {patience_left}')
                    patiences.append(patience_left)
            if args.stop_epoch > 0 and args.stop_epoch <= epoch + 1:
                quit = True
                
            if quit:
                break
            
            epochs_left = num_epochs - epoch - 1
            if epochs_left > 0:
                print(f'Mean epoch time {mean_epoch_time} seconds. Predicted max time left {epochs_left * mean_epoch_time} seconds.')
                min_epochs_left = min(patience_left, epochs_left)
                if min_epochs_left > 0:
                    print(f'Predicted min time left {min_epochs_left * mean_epoch_time} seconds.')

        print(f'Finished Training, best val accuracy: {best_accuracy:.2f}%')

        # Load best model (based on val performance, rollback on validation loss)
        util.load_model(model, None, get_model_path(args, 'accuracy'))
        model.to(device)
    
    # End of training
    correct, total, loss, accuracy_per_label, top_k_accuracies = eval_model(model, testloader, criterion, device, num_classes)

    test_accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {test_accuracy:.2f}%')

    # Reading JSON file first. It will not exist for ensemble
    json_data = {}
    json_path = get_json_path(args, 'accuracy')
    if Path(json_path).exists():
        with open(json_path, 'r') as file:
            json_data = json.load(file)
    json_data['testAccuracyPerLabel'] = accuracy_per_label
    json_data['testTopKAccuracies'] = top_k_accuracies
    json_data['testAccuracy'] = test_accuracy
    json_data["args"] = vars(args)
    if training_epochs >= 0:
        json_data["trainingEpochs"] = training_epochs
    
    with open(get_json_path(args, 'accuracy'), 'w') as f:
        json.dump(json_data, f)
    


def load_label_weights(args, variant, num_classes):
    json_path = get_json_path(args, variant)
    weight_values = [1.0] * num_classes
    with open(json_path, 'r') as file:
        json_data = json.load(file)
        label_accuracies = json_data['validationLabelAccuracies']
        for label_str, label_acc in label_accuracies.items():
            weight_values[int(label_str)] = label_acc
    return weight_values


def get_plane_dimensions(dimensions):
    """
    Group dimensions by their plane prefix (xy/xt/ty).
    """
    plane_dimensions = defaultdict(list)

    for dim in dimensions:
        plane_name = dim.split("_")[0]
        plane_dimensions[plane_name].append(dim)
    return plane_dimensions

def get_sub_weights(group, args):
    dim_name = group
    if group == "xy" or group == "cstr":
        dim_name = "xy"
    elif group == "xt" or group == "tx":
        dim_name = "xt"
    elif group == "yt" or group == "ty":
        dim_name = "yt"
    key = f"weights_{dim_name}"
    if hasattr(args, key):
        sub_weights = getattr(args, key)
        if sub_weights == "default":
            return args.weights
        elif sub_weights == "no_weights":
            return None
        pass
    return args.weights

def get_split_model_parameters(channels, num_classes, device, train, args, variant='accuracy', weights=None, train_submodel=True, load_model_data=True):
    """
    Build sub-models and associated metadata for PaSNet architectures.
    
    Returns:
      - models: list of sub-models (one per plane group)
      - splits: list of channel counts per sub-model
      - train: passthrough flag
      - model_weights: list of per-class weight vectors for each sub-model
    """
    dimensions = sort_dimensions_str(args.dimensions).split(";")
    plane_dimensions = get_plane_dimensions(dimensions)
    base_network = args.architecture.split("_")[1]
  
    models = []
    splits = []
    model_weights = []
    model_weigths_sums = [0.0] * num_classes
    
    for group, dims in plane_dimensions.items():
        
        args_copy = copy.deepcopy(args)
        args_copy.dimensions = ";".join(sorted(dims))
        args_copy.architecture = f"merged_{base_network}"
        args_copy.stop_epoch = 0
        
        args_copy.weights = get_sub_weights(group, args)
        args_copy.weights_xy = "default"
        args_copy.weights_xt = "default"
        args_copy.weights_yt = "default"
        
        if args_copy.save_model == "submodels_only":
            args_copy.save_model = "all"
        
        model_path = get_model_path(args_copy, variant)
        json_path = get_json_path(args_copy, variant)
        
        need_train = not Path(model_path).exists() or not Path(json_path).exists()
        if not need_train:
            # Verify that there is a test result in the JSON file
            with open(json_path, 'r') as file:
                need_train = 'testAccuracy' not in json.load(file)
                if need_train:
                    print(f"Could not find test result in JSON file, need to retrain model...")

        if need_train:
            if train_submodel:
                print(f"Training submodel {model_path}")
                perform_action(args_copy)
            else:
                raise ValueError(f"Could not find submodel {model_path} and train_submodel==False")

        print(f"Loading submodel {model_path}")

        split = 0
        for dim in dims:
            split += get_dimension_channel_count(dim)
        splits.append(split)
            
        model = get_base_network(base_network, split, num_classes, device)
        models.append(model)
        if load_model_data:
            util.load_model(model, None, model_path)
        weight_values = [1.0] * num_classes
        if weights == 'label':
            weight_values = load_label_weights(args_copy, variant, num_classes)
        model_weights.append(weight_values)
        for i in range(num_classes):
            model_weigths_sums[i] += weight_values[i]

    for i in range(num_classes):
        for j in range(len(models)):
            model_weights[j][i] /= model_weigths_sums[i]

    return models, splits, train, model_weights


def get_pasnet_feature_fusion_model(channels, num_classes, device, train, args, variant='accuracy', weights=None, train_submodel=True, feature_count=512, load_model_data=True):
    """
    Construct a FeatureFusionPaSNetModel from submodels.
    """
    models, splits, train, model_weights = get_split_model_parameters(channels, num_classes, device, train, args, variant, weights, train_submodel, load_model_data)
    
    return FeatureFusionPaSNetModel(num_classes, models, splits, train, model_weights, feature_count).to(device)


def get_pasnet_ensemble_model(channels, num_classes, device, train, args, variant='accuracy', weights=None, train_submodel=True, agg_type="logit_sum", load_model_data=True):
    """
    Construct an EnsemblePaSNetModel from submodels with specified aggregation type.
    """
    models, splits, train, model_weights = get_split_model_parameters(channels, num_classes, device, train, args, variant, weights, train_submodel, load_model_data)
    return EnsemblePaSNetModel(models, splits, train, model_weights, agg_type).to(device)


def get_base_network(base_network, channels, num_classes, device):
    """
    Factory for base backbone networks, without pretrained weights.
    """
    if base_network == "resnet18":
        return util.get_resnet18(channels, num_classes).to(device)
    elif base_network == "resnet34":
        return util.get_resnet34(channels, num_classes).to(device)
    elif base_network == "resnet50":
        return util.get_resnet50(channels, num_classes).to(device)
    elif base_network == "vitb16":
        return AdaptedVisionTransformer(224, num_classes, channels).to(device)
    elif base_network == "mobilenetv3s":
        return util.adapt_mobilenetv3(util.get_mobilenetv3("small"), channels, num_classes).to(device)
    elif base_network == "mobilenetv3l":
        return util.adapt_mobilenetv3(util.get_mobilenetv3("large"), channels, num_classes).to(device)
    elif base_network == "inceptionv3":
        return util.get_inception_v3(channels, num_classes).to(device)
    return None


if __name__ == "__main__":
    load_dotenv()
    log.setup_log()    
    args = parse_args()
    print(args)
    perform_action(args)
    log.flush_all()

