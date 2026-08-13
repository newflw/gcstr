import numpy as np
import torch
import json
import os
from pathlib import Path
import random
import torchvision
import torch.nn as nn
from collections import defaultdict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_requires_grad(model, rg):
    if hasattr(model, 'set_requires_grad'):
        model.set_requires_grad(rg)
    else:
        for param in model.parameters():
            param.requires_grad = rg


def set_fc_requires_grad(model, rg):
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = rg


def save_model(model, optimizer, epoch, path, custom_data=None):
    """
    Saves the model state, optimizer state, and epoch number to a specified path.

    Args:
    model (torch.nn.Module): The model to be saved.
    optimizer (torch.optim.Optimizer): The optimizer used during training.
    epoch (int): The current epoch number.
    path (str): The destination path for saving the model.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'custom_data': custom_data,
    }, path)


def load_model(model, optimizer, path):
    """
    Loads the model state, optimizer state, and the last epoch from a specified path.

    Args:
    model (torch.nn.Module): The model that the state will be loaded into.
    optimizer (torch.optim.Optimizer): The optimizer for which the state will be restored.
    path (str): The path from where to load the model.

    Returns:
    int: The last epoch number from the saved state.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    custom_data = checkpoint.get('custom_data', {})
    return epoch, custom_data


def save_checkpoint(file, model, optimizer, epoch, data=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    if data:
        checkpoint['data'] = data
    torch.save(checkpoint, file)


def load_checkpoint(file, model, optimizer):
    checkpoint = torch.load(file)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    data = checkpoint.get('data', None)

    return epoch, data




def get_inception_v3(channels: int, num_classes: int, pretrained: bool = False):
    if channels != 3:
        raise ValueError("InceptionV3 only supports 3 channels")
    weights = torchvision.models.Inception_V3_Weights.DEFAULT if pretrained else None
    model = torchvision.models.inception_v3(weights=weights)

    # Disable/remove aux head
    model.aux_logits = False
    model.AuxLogits = None

    # Replace classification head
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model



def get_mobilenetv3(model_type, weights=""):
    pretrained = False
    if weights == "imagenet":
        pretrained = True
    if model_type == 'small':
        model = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
    else:
        model = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
    return model


def adapt_mobilenetv3(model, input_channels: int, num_classes: int):
    """
    Adapt MobileNetV3 for a different number of input channels and output classes.

    Args:
        input_channels (int): Number of input channels (e.g., 1 for grayscale, 4 for custom input).
        num_classes (int): Number of output classes for classification.
        model_type (str): Either 'small' or 'large' for MobileNetV3 model type.

    Returns:
        nn.Module: The adapted MobileNetV3 model.
    """

    # Modify the first convolutional layer to accept 'input_channels' instead of 3
    model.features[0][0] = nn.Conv2d(
        # Number of input channels (e.g., 1, 4, etc.)
        in_channels=input_channels,
        out_channels=model.features[0][0].out_channels,
        kernel_size=model.features[0][0].kernel_size,
        stride=model.features[0][0].stride,
        padding=model.features[0][0].padding,
        bias=False
    )

    # Modify the last fully connected layer to output 'num_classes' instead of 1000
    # The classifier typically looks like [Dropout, Linear, Dropout, Linear]
    # We're modifying the last Linear layer to output the correct number of classes
    model.classifier[-1] = nn.Linear(
        in_features=model.classifier[-1].in_features,
        out_features=num_classes
    )

    # Randomly initialize the weights of the newly modified layers
    nn.init.kaiming_normal_(
        model.features[0][0].weight, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(
        model.classifier[-1].weight, mode='fan_out', nonlinearity='linear')

    return model


def adapt_resnet(model, in_channels=2, num_classes=10, features=512):
    if num_classes > 0:
        model.fc = nn.Linear(features, num_classes)
    else:
        model.fc = nn.Identity()

    if in_channels != 3:

        model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        nn.init.kaiming_normal_(
            model.conv1.weight, mode='fan_out', nonlinearity='relu')

    return model


def get_resnet50(in_channels=2, num_classes=10, weights=None) -> nn.Module:
    if weights and in_channels != 3:
        raise ValueError("Do not use weights != None and channels != 3")
    if weights == "imagenet":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    return adapt_resnet(torchvision.models.resnet50(progress=True, weights=weights), in_channels, num_classes, 2048)


def get_resnet34(in_channels=2, num_classes=10, weights=None) -> nn.Module:
    if weights and in_channels != 3:
        raise ValueError("Do not use weights != None and channels != 3")
    if weights == "imagenet":
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
    return adapt_resnet(torchvision.models.resnet34(progress=True, weights=weights), in_channels, num_classes)


def get_resnet18(in_channels=2, num_classes=10, weights=None) -> nn.Module:
    if weights and in_channels != 3:
        raise ValueError("Do not use weights != None and channels != 3")
    if weights == "imagenet":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    return adapt_resnet(torchvision.models.resnet18(progress=True, weights=weights), in_channels, num_classes)



def make_structured_array(*args, dtype):
    """
    Make a structured array given values for each field.

    Args:
        *args: Arrays/lists of field values, in the order of `dtype.names`.
        dtype: Structured dtype describing the fields.

    Returns:
        struct_arr: numpy structured array
    """
    names = dtype.names
    assert len(args) == len(
        names), f"Expected {len(names)} args, got {len(args)}"

    # Not as memory efficient?
    # return np.fromiter(zip(*args), dtype=dtype)

    struct_arr = np.empty(len(args[0]), dtype=dtype)
    for name, arg in zip(names, args):
        struct_arr[name] = arg
    return struct_arr



def split_list_by_fractions(original_list, fractions):
    # Ensure that the fractions sum up to 1 (or close enough)
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Fractions must sum up to 1.")

    total_length = len(original_list)
    sizes = [round(f * total_length) for f in fractions]

    # Adjust if the rounding causes a mismatch in the total length
    if sum(sizes) != total_length:
        difference = total_length - sum(sizes)
        sizes[-1] += difference  # Adjust the last list size

    # Split the original list based on the calculated sizes
    sublists = []
    start_index = 0
    for size in sizes:
        sublists.append(original_list[start_index:start_index + size])
        start_index += size

    return sublists


def create_validation_set(train_folder, target_file, val_fraction):
    files = list(Path(train_folder).glob('*.npz'))
    label_paths = defaultdict(list)
    for path in files:
        label_paths[path.stem.split("_")[0]].append(path)

    val_filenames = []
    train_filenames = []
    for label, label_files in label_paths.items():
        random.shuffle(label_files)
        val_files, train_files = split_list_by_fractions(
            label_files, [val_fraction, 1.0 - val_fraction])
        val_filenames.extend([f.name for f in val_files])
        train_filenames.extend([f.name for f in train_files])

    with open(target_file, 'w') as f:
        json.dump({"validation": val_filenames,
                  "training": train_filenames}, f)


def get_validation_set_filename(target_folder, i, val_fraction):
    return f"{target_folder}/val_split_{i}_{int(100 * val_fraction)}.json"


def create_validation_sets(n, train_folder, target_folder, val_fraction):
    for i in range(n):
        create_validation_set(train_folder, get_validation_set_filename(
            target_folder, i, val_fraction), val_fraction)


def get_dataset_folder(dataset):
    return f"{get_data_folder()}/{dataset}/action_events"


def get_orig_dataset_folder():
    return os.getenv("DATASET_FOLDER", "./datasets")


def get_data_folder():
    return os.getenv("DATA_FOLDER", "./data")


def get_model_folder():
    return os.getenv("MODEL_FOLDER", "./saved_models")


def get_log_folder():
    return os.getenv("LOG_FOLDER", "./logs")
