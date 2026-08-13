
# Generalized CSTR

## How to Install

Install the libraries listed in requirements.txt

The code is tested with python 3.11.6

### CUDA Support

Pytorch with CUDA support (tested with 12.1) is installed separately. 
Example using pip:
```
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Datasets

Download from the sources and place in the dataset folder (specified by the DATASET_FOLDER environment variable)

| Dataset | Source URL |
|---------|-----|
| DailyDVS-200    | https://github.com/QiWang233/DailyDVS-200 |
| THU-EACT-50     | https://github.com/lujiaxuan0520/THU-EACT-50 |
| DVS-Gesture     | https://www.kaggle.com/datasets/xingfenyizhen/dvsgesture128 |
| ASL-DVS         | https://github.com/PIX2NVS/NVS2Graph |
| N-Caltech101    | https://www.garrickorchard.com/datasets/n-caltech101 |
| N-CARS          | https://www.prophesee.ai/2018/03/13/dataset-n-cars |
| SL-Animals-DVS  | http://www2.imse-cnm.csic.es/caviar/SL_Animals_Dataset/ |
| DVS-Lip         | https://sites.google.com/view/event-based-lipreading |
| CIFAR10-DVS     | https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671 |
| DailyAction-DVS | https://github.com/qianhuiliu/SNN-action-recognition |


### Special instructions for DailyDVS-200

Copy the text files from https://github.com/QiWang233/DailyDVS-200/tree/main/train_val_test and place them in the same folder as the action_* subfolders


## Test Splits

Some of the datasets does not have an official test split.
We generated random 80/20 splits and they are also included in this repository.
When pre-processing those datasets, the corresponding test splits are used automatically.


## Environment variables

The project uses the dotenv library.
To modify environment variables, place a .env file in the root folder.

| Variable Name  | Default Value   | Description |
|----------------|-----------------|-------------|
| DATASET_FOLDER | ./datasets      | Root folder for the extracted, raw datasets |
| DATA_FOLDER    | ./data          | Root folder where pre-processed files, cache and augmentations will be stored |
| MODEL_FOLDER   | ./saved_models  | Where models and model training results are stored |
| LOG_FOLDER     | ./logs          | Logs are stored here |


## Dimensions string format

The dimensions parameter --dimensions is used to specify the representations to use.

### Single plane

| Variant         | Dimension string             |
|-----------------|------------------------------|
| $\text{CSTR}_3$ | cstr_3                       |
| $\text{CSTR}_2$ | cstr_2                       |
| $\text{XT}_b^p$ | xt_density_bin;xt_mean_y_pol |
| $\text{XT}_p^b$ | xt_density_pol;xt_mean_y_bin |
| $\text{XT}_p$   | xt_density_pol;xt_zero       |
| $\text{XT}^p$   | xt_density_bin;xt_mean_y_pol |
| $\text{YT}_b^p$ | ty_density_bin;ty_mean_x_pol |
| $\text{YT}_p^b$ | ty_density_pol;ty_mean_x_bin |
| $\text{YT}_p$   | ty_density_pol;ty_zero       |
| $\text{YT}^p$   | ty_density_bin;ty_mean_x_pol |

Note that the dimension string uses "ty" instead of "yt" for this format so that the spatial coordinate is the same.

The current implementation will always order the subimages according to the name, which is a limitation if you want to test a different order in the image channels.

### Multiple planes

By mixing planes in the dimension string, the chosen architecture determines how this is handled.

The "merged" models will just use all images in order into a single image.

The "pasnet" models will split up the images and create merged models for each plane.


## How dimensions in combination with architecture affects training

Using merged architectures just stacks all channels. This is not compatible with using pretrained weights.

Using PaSNet architectures splits the training into up to 3 submodels. Always merges the same 
When using pretrained models it is required that the number of channels sums up to 3.


## Running

Training and testing 5 PaSNet ensembles (probability multiplication) with 3 branches (total 15 submodels and 5 ensembles without further training) using CSTR, YT with polarized event count and XT with polarized event count on the DailyAction-DVS dataset:

```
python main.py --action "train_n" --run_count 5 --dataset "daily_action_dvs" --dimensions "cstr_3;ty_density_pol;ty_zero;xt_density_pol;xt_zero" --architecture "pasnetenprobmul_resnet18"
```


### Once-per-dataset Pre-processing

The first time a dataset is used, some pre-processing is performed to convert the original dataset into a common file structure format.

For the larger datasets this can take several hours with the default options.

There is an option --preprocess_mode that you can set to "parallel" but it is currently not working very well on Windows.


## Augmentations
Augmentations are cached on disk for performance reasons.

The number of augmentations per instance is determined by the --augmentation_count parameter (default set to 5).

During training, when accessing an instance, a random variant of the N augmentations will be used and generated if missing from the cache.


## Image Cache

Calculated images are cached by default (--use_image_cache set to 1) and calculated when needed (--image_cache_mode set to "lazy").
Using the "lazy" mode makes the training slow during the first epoch (or more epochs when using augmentations).
It is possible to calculate all images as quickly as possible with --image_cache_mode set to "precalculate" but this mode will not work using augmentations on Linux or Mac at the moment.

It is also possible to cache each image component by setting --use_sub_image_cache to 1. This makes it faster to train multiple models that share image components but it also uses a lot more disk space.

The image cache can be cleared for a dataset by using --action parameter set to "clear_image_cache".

When using augmentations, the disk usage of image and sub image caches will be multiplied by the --augmentation_count parameter.

## Model Cache
Submodels that have been trained are stored in the models folder and can then be reused by multiple PaSNet architectures.
The models are found by using a very long file format and paired with a result JSON file.
A model can be reused if the model file and the JSON file exists and that there is a test result stored in the JSON file, which is a guard against using only partially trained models after an aborted run.


## Arguments to main.py


### The --action parameter
| Value  | Description |
|--------|-------------|
| train   | Perform a single training of a model |
| train_n | Perform N training runs followed by a "summarize_runs" action |
| summarize_runs | Summarizes N training runs and places the result in the folder specified by --result_folder |
| clear_augmentations | Removes all augmented instances for a dataset and augmentation type |
| clear_image_cache | Removes the image cache for a dataset |






- --dataset
    - One of: ncars, dvs_lip, asl_dvs, dvs_gesture, daily_action_dvs, ncaltech101, sl_animals_dvs, cifar10_dvs, thu_eact_50_chl, daily_dvs_200
- --action 
    - One of: train, train_n, clear_image_cache, clear_augmentations, summarize_runs


```
options:
  -h, --help            show this help message and exit
  --dataset {ncars,dvs_lip,asl_dvs,dvs_gesture,daily_action_dvs,ncaltech101,sl_animals_dvs,cifar10_dvs,thu_eact_50_chl,daily_dvs_200}
                        Dataset to use. One of: ncars, dvs_lip, asl_dvs, dvs_gesture, daily_action_dvs, ncaltech101,
                        sl_animals_dvs, cifar10_dvs, thu_eact_50_chl, daily_dvs_200
  --action {train,train_n,clear_image_cache,clear_augmentations,summarize_runs}
                        Main action
  --result_folder RESULT_FOLDER
                        Result folder
  --transform {no_transform,imagenet_norm}
                        Image transform type
  --save_model {all,submodels_only}
                        Result folder
  --run_index RUN_INDEX
                        Run index
  --use_image_cache USE_IMAGE_CACHE
                        Enable image cache
  --use_sub_image_cache USE_SUB_IMAGE_CACHE
                        Enable sub image cache
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Early stopping patience. Set to 0 for no early stopping
  --validation_percent VALIDATION_PERCENT
                        Percent of training data to use for validation
  -W W, --W W           Input image tensor width
  -H H, --H H           Input image tensor height
  -T T, --T T           Input image tensor time bins
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       Epoch count
  --stop_epoch STOP_EPOCH
                        Stop epoch (will not be used for submodels)
  --lr LR               Learning rate
  --finetune_epoch FINETUNE_EPOCH
                        Finetune epoch
  --finetune_lr FINETUNE_LR
                        Finetune learning rate
  --augmentation AUGMENTATION
                        Augmentation
  --augmentation_count AUGMENTATION_COUNT
                        Number of augmentations per instance
  --image_cache_mode {precalculate,lazy}
                        Image cache mode
  --run_count RUN_COUNT
                        Number of training runs
  --weights {None,imagenet}
                        Starting weights
  --weights_xy {default,no_weights,imagenet}
                        Starting weights in XY
  --weights_xt {default,no_weights,imagenet}
                        Starting weights in XT
  --weights_yt {default,no_weights,imagenet}
                        Starting weights in YT
  --architecture {merged_vitb16,merged_resnet18,merged_resnet34,merged_resnet50,pasnetff_resnet18,pasnetff_resnet34,pasnetff_resnet50,pasnetensum_resnet18,pasnetensum_resnet34,pasnetensum_resnet50,pasnetenprobsum_resnet18,pasnetenprobmul_resnet18}
                        Model architecture to use
  --lr_scheduler {none,cosine}
                        Learning rate scheduler to use. One of: none, cosine
  --dimensions DIMENSIONS
                        Dimensions to use
  --dithering {nodithering,uniform}
                        Dithering method for augmentation

```


