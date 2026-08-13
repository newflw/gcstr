
import torch.nn as nn
import torch
import util
import math
import torchvision


class PaSNetModel(nn.Module):
    def __init__(self, models, splits, train=False, model_weights=None):

        super().__init__()
        
        self.models = nn.ModuleList(models)  # Store the models as a ModuleList for easy handling.
        self.splits = splits
        self.model_weights = model_weights

        if not train:
            # Ensure that the models do not have any trainable parameters.            
            self.set_requires_grad(False)

        
    def set_requires_grad(self, rg):
        for model in self.models:
            util.set_requires_grad(model, rg)
        
        
    def split_input(self, x):
        """
        This function will handle the input splitting logic.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            List of tensors, each corresponding to the input to be passed to each model.
        """
        result = []
        cur_channel = 0
        for split in self.splits:
            result.append(x[:, cur_channel:cur_channel+split, :, :])
            cur_channel += split
        return result


class FeatureFusionPaSNetModel(PaSNetModel):
    def __init__(self, num_classes, models, splits, train=False, model_weights=None, feature_count=512):
        super().__init__(models, splits, train, model_weights)

        for model in models:
            model.fc = nn.Identity()

        self.fc = nn.Linear(feature_count * len(models), num_classes)
    
    
    def forward(self, x):
        """
        Forward pass for the feature fusion model.
        
        Args:
            x (torch.Tensor): Input tensor to be split and fed to different models.
            
        Returns:
            torch.Tensor: Sum of outputs from all models.
        """
        inputs = self.split_input(x)
        
        model_outputs = []
        
        for model, input_data, weights in zip(self.models, inputs, self.model_weights):
            model_output = model(input_data)
            model_outputs.append(model_output)
        
        output = self.fc(torch.cat(model_outputs, dim=1))
        return output



class EnsemblePaSNetModel(PaSNetModel):
    def __init__(self, models, splits, train=False, model_weights=None, agg_type="logit_sum"):
        """
        EnsemblePaSNetModel takes in a list of pretrained models and sums their outputs.
        
        Args:
            models (list): A list of PyTorch models that are trained on the same number of labels.
            splits (list): A list of integers that specifies how many channels each model takes as input.
            train (bool): If the models can be trained or not
            model_weights (list): A list of per-label weight lists
        """
        super().__init__(models, splits, train, model_weights)
        self.agg_type = agg_type
               
    
    
    def forward(self, x):
        """
        Forward pass for the ensemble model.
        
        Args:
            x (torch.Tensor): Input tensor to be split and fed to different models.
            
        Returns:
            torch.Tensor: Sum of outputs from all models.
        """
        inputs = self.split_input(x)  # Split the input for each model.
        
        outputs = []
        
        for model, input_data, weights in zip(self.models, inputs, self.model_weights):
            model_output = model(input_data)
            outputs.append(model_output)
        
        if self.agg_type == "logit_sum":
            ensemble_output = torch.sum(torch.stack(outputs), dim=0)
        elif self.agg_type == "prob_sum":
            prob_outputs = [torch.softmax(output, dim=1) for output in outputs]
            ensemble_output = torch.sum(torch.stack(prob_outputs), dim=0)
        elif self.agg_type == "prob_mul":
            prob_outputs = [torch.softmax(output, dim=1) for output in outputs]
            ensemble_output = torch.prod(torch.stack(prob_outputs), dim=0)        
        
        return ensemble_output


class EnsemblePaSNetModelStream(PaSNetModel):
    def __init__(self, models, splits, train=False, model_weights=None, agg_type="logit_sum"):
        super().__init__(models, splits, train, model_weights)
        self.agg_type = agg_type
        self.streams = [torch.cuda.Stream() for _ in models] if torch.cuda.is_available() else None

    def forward(self, x):
        inputs = self.split_input(x)

        if not x.is_cuda or self.streams is None:
            outputs = [model(inp) for model, inp in zip(self.models, inputs)]
        else:
            outputs = [None] * len(self.models)
            current_stream = torch.cuda.current_stream(x.device)

            for i, (model, inp, stream) in enumerate(zip(self.models, inputs, self.streams)):
                with torch.cuda.stream(stream):
                    outputs[i] = model(inp)

            # Make sure the default/current stream waits for all model streams
            for stream in self.streams:
                current_stream.wait_stream(stream)

        if self.agg_type == "logit_sum":
            return torch.stack(outputs, dim=0).sum(dim=0)

        probs = [torch.softmax(o, dim=1) for o in outputs]
        if self.agg_type == "prob_sum":
            return torch.stack(probs, dim=0).sum(dim=0)
        elif self.agg_type == "prob_mul":
            return torch.stack(probs, dim=0).prod(dim=0)

        raise ValueError(f"Unknown agg_type: {self.agg_type}")


class AdaptedVisionTransformer(nn.Module):
    def __init__(self, img_size=224, num_classes=10, in_channels=3):
        super(AdaptedVisionTransformer, self).__init__()
        
        self.vit_model = torchvision.models.vit_b_16(weights=None)  # vit_b_16 is a base ViT model
        
        self.vit_model.conv_proj = nn.Conv2d(in_channels, self.vit_model.conv_proj.out_channels,
                                             kernel_size=self.vit_model.conv_proj.kernel_size,
                                             stride=self.vit_model.conv_proj.stride,
                                             padding=self.vit_model.conv_proj.padding)
        
        fan_in = self.vit_model.conv_proj.in_channels * self.vit_model.conv_proj.kernel_size[0] * self.vit_model.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.vit_model.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.vit_model.conv_proj.bias is not None:
            nn.init.zeros_(self.vit_model.conv_proj.bias)
               
        if isinstance(self.vit_model.heads, nn.Sequential):
            last_layer = self.vit_model.heads[-1]  # Get the last layer
            if isinstance(last_layer, nn.Linear):
                in_features = last_layer.in_features
                self.vit_model.heads[-1] = nn.Linear(in_features, num_classes)  # Replace the last layer
        else:
            # In case it's not a Sequential, handle the direct layer case
            in_features = self.vit_model.heads.in_features
            self.vit_model.heads = nn.Linear(in_features, num_classes)       

    def forward(self, x):
        return self.vit_model(x)
