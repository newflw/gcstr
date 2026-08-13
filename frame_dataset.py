
from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom
import os
import math
from multiprocessing import Pool
import random
from PIL import Image, ImageDraw

import util
from augmentation import get_augmentation_path, create_augmentation
from dimension import Dimension


dataset_names = [
    'ncars', 'dvs_lip', 'asl_dvs', 'dvs_gesture', 'daily_action_dvs', 'ncaltech101', 'sl_animals_dvs', 
    'cifar10_dvs', 'thu_eact_50_chl', 'daily_dvs_200', 'sl_animals_dvs_k_fold'
]

def normalize(arr):
    min_val = arr.min()
    max_val = arr.max()
    span = max_val - min_val
    if span == 0:
        span = 1
    normalized_arr = (arr - min_val) / span
    return normalized_arr


def get_timestamps(data, shape):
    timestamps = np.zeros(shape)
    return timestamps[np.newaxis, :, :]


def get_max_timestamps(data, shape):
    timestamps = np.zeros(shape)
    if len(data) > 1:        
        min_t = data[0]['t']
        max_t = data[-1]['t']
        s = 1.0 / (max_t - min_t)
        W, H = shape
        for ev in data:
            x = min(W - 1, ev['x'])
            y = min(H - 1, ev['y'])
            t = ev['t'] - min_t
            timestamps[int(x), int(y)] = max(timestamps[int(x), int(y)], t * s)
    else:
        timestamps = np.ones(shape) * 0.5
    return timestamps[np.newaxis, :, :]


def get_mean_timestamps(data, shape):
    W, H = shape
    mean_timestamps = np.zeros(shape)
    counts = np.ones(shape) * 0.01  # Avoid division by zero

    if len(data) > 1:
        t_values = data['t']
        min_t = t_values[0]
        max_t = t_values[-1]

        t_range = max_t - min_t
        if t_range == 0:
            t_range = 1
        t_scale = 1.0 / t_range

        for i in range(len(data)):
            x = min(W - 1, data['x'][i])
            y = min(H - 1, data['y'][i])
            t = data['t'][i]
            tc = t_scale * (t - min_t)
            mean_timestamps[x, y] += tc
            counts[x, y] += 1

        mean_timestamps /= counts
        max_mean_t = np.max(mean_timestamps)
    else:
        mean_timestamps = np.ones(shape) * 0.5
        max_mean_t = 1

    return (mean_timestamps / max_mean_t)[np.newaxis, :, :]


def compute_mean_ys(data, W, T):
    mean_ys = np.zeros((W, T))
    counts = np.ones((W, T)) * 0.01 # Avoid division by zero

    t_values = data['t']
    min_t = t_values[0]
    max_t = t_values[-1]

    t_range = max_t - min_t
    if t_range == 0:
        t_range = 1

    t_scale = T / t_range

    for ev in data:
        x = min(W - 1, ev['x'])
        y = ev['y']
        t = ev['t']
        t = min(T - 1, int(t_scale * (t - min_t)))
        mean_ys[int(x), int(t)] += y
        counts[int(x), int(t)] += 1

    mean_ys /= counts
    max_mean_y = np.max(mean_ys)

    return mean_ys, max_mean_y


def get_xt_mean_ys(data, W, T):
    if len(data) > 1:
        mean_ys, max_mean_y = compute_mean_ys(data, W, T)
    else:
        mean_ys = np.ones((W, T)) * 0.5
        max_mean_y = 1

    return (mean_ys / max_mean_y)[np.newaxis, :, :]


def get_ty_mean_xs(data, H, T):
    mean_xs = np.zeros((T, H))

    if len(data) > 1:
        counts = np.full((T, H), 0.01)  # Initialize counts with 0.01

        t_values = data['t']
        min_t = t_values[0]
        max_t = t_values[-1]

        t_range = max_t - min_t
        if t_range == 0:
            t_range = 1

        t_scale = T / t_range
        
        for i in range(len(data['x'])):
            x = data['x'][i]
            y = min(H - 1, data['y'][i])
            t = data['t'][i]
            t = min(T - 1, math.floor(t_scale * (t - min_t)))
            mean_xs[t, y] += x
            counts[t, y] += 1

        mean_xs /= counts
        max_mean_x = np.max(mean_xs)
    else:
        mean_xs = np.full((T, H), 0.5)  # Initialize mean_xs with 0.5
        max_mean_x = 1

    return (mean_xs / max_mean_x)[np.newaxis, :, :]


def get_data(dataset, index):
    return dataset[index]
    

def create_cached(dataset):
    arguments = []
    for i in range(len(dataset)):
        arguments.append((dataset, i))

    max_procs = max(1, (os.cpu_count() or 1) // 2)
        
    with Pool(processes=max_procs, initializer=_init_worker) as pool:
        async_result = pool.starmap_async(get_data, arguments)
        try:
            results = async_result.get(timeout=2200) # Just setting a fixed timeout per augmentation index
        except Exception:
            print("Timed out!")
            pool.terminate()  # forcefully stop workers
            pool.join()





def extract_event_frames(arr, fps=-1, frame_len=-1):
    if arr.size == 0:
        return []

    max_t = np.max(arr['t'])
    min_t = np.min(arr['t'])

    if fps > 0:
        frame_len = 1000000 / fps
    elif frame_len <= 0:
        raise ValueError("Need one of fps or frame_len to be > 0")

    t = min_t
    prev_t = t
    arrs = []
    while t < max_t:
        t += frame_len
        filter = (arr['t'] >= prev_t) & (arr['t'] < t)
        arrs.append(arr[filter])
        prev_t = t
    return arrs



def create_image_from_frame(w, h, frame_events, event_size=5, flip_y=False):
    img = Image.new('RGB', (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for ev in frame_events:
        color = (255, 0, 0) if ev['p'] else (0, 0, 255)
        ex, ey = ev['x'], ev['y']
        x = ex * event_size
        if flip_y:
            y = h - (ey + 1) * event_size
        else:
            y = ey * event_size
        draw.rectangle([(x, y), (x + event_size, y + event_size)],
                       outline=color, fill=color)
    return img


def create_images(events, w, h, frame_count, event_size):
    if len(events) < 2:
        return []
    frame_us = (events[-1]['t'] - events[0]['t']) // frame_count
    event_frames = extract_event_frames(events, frame_len=frame_us)[:frame_count]
    images = []
    for i, frame in enumerate(event_frames):
        img = create_image_from_frame(w * event_size, h * event_size, frame, event_size=event_size)
        images.append(img)
    return images

def _init_worker():
    # Prevent libraries from oversubscribing
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    

class SubsetFrameDataset(Dataset):
    def __init__(self, orig_dataset, folder, filenames, transform=None, augmentation=""):
        """
        SubsetFrameDataset constructor.
        
        Parameters:
        - orig_dataset (Dataset): The original FrameDataset instance.
        - folder (str): The root folder where the dataset is stored (usually this will be passed to FrameDataset).
        - filenames (list of str): The list of filenames to use as a subset from the original dataset.
        """
        self.orig_dataset = orig_dataset
        self.folder = folder
        self.filenames = set(filenames)  # Use a set for fast lookup of filenames
        self.transform = transform
        self.augmentation = augmentation
        self.forced_aug_index = -1
        
        # Filter the original dataset files based on the filenames provided
        self.subset_indices = []
        for i, file_path in enumerate(self.orig_dataset.files):
            file_name = Path(file_path).name  # Extract the filename from the full path
            if file_name in self.filenames:
                self.subset_indices.append(i)
                
        # Make sure labels and files are subset correctly
        self.files = [self.orig_dataset.files[i] for i in self.subset_indices]
        self.labels = [self.orig_dataset.labels[i] for i in self.subset_indices]

    def create_augmentations(self):
        orig = self.orig_dataset
        os.makedirs(f'{orig.root_dir}/augmentations', exist_ok=True)
        max_procs = max(1, (os.cpu_count() or 1) // 2)
        for a in range(orig.augmentation_count):
            print(f"Creating augmentation {a+1}/{orig.augmentation_count} for augmentation {self.augmentation}")
            arguments = []
            for index in self.subset_indices:
                file_path = orig.files[index]
                arguments.append((Path(file_path), self.augmentation, a, orig.dither))

            with Pool(processes=max_procs, initializer=_init_worker) as pool:
                async_result = pool.starmap_async(create_augmentation, arguments)
                try:
                    results = async_result.get(timeout=3600) # Just setting a fixed timeout of one hour per augmentation index
                except TimeoutError:
                    print("Timed out!")
                    pool.terminate()  # forcefully stop workers
                    pool.join()


    def __len__(self):
        """Return the length of the subset."""
        return len(self.subset_indices)

    def __getitem__(self, idx):
        """
        Retrieve the item (data, label) at the given index in the subset.
        
        Parameters:
        - idx (int): The index of the item to retrieve.
        
        Returns:
        - (data, label): The data and label corresponding to the subset index.
        """
        orig_idx = self.subset_indices[idx]  # Map to original dataset index        
        old_augmentation = self.orig_dataset.augmentation
        old_forced_aug_index = self.orig_dataset.forced_aug_index
        self.orig_dataset.augmentation = self.augmentation
        self.orig_dataset.forced_aug_index = self.forced_aug_index
        tensor, label = self.orig_dataset[orig_idx]
        self.orig_dataset.augmentation = old_augmentation
        self.orig_dataset.forced_aug_index = old_forced_aug_index
        
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor, label


class FrameDataset(Dataset):

    def __init__(self, root_dir, dimensions, W=128, H=128, T=128, slice=None, transform=None, use_image_cache=True, augmentation=None, augmentation_count=5, use_subimage_cache=True, dither=True):
        if slice:
            root_dir = f"{root_dir}/{slice}"
        self.use_image_cache = use_image_cache
        self.use_subimage_cache = use_subimage_cache
        self.slice = slice
        self.root_dir = root_dir
        self.transform = transform
        self.dimensions = dimensions
        self.augmentation = augmentation
        self.augmentation_count = augmentation_count
        self.W = W
        self.H = H
        self.T = T
        self.forced_aug_index = -1
        self.dither = dither

        paths = sorted(Path(self.root_dir).glob("*.npz"), key=lambda p: p.stem)
        self.files = []
        
        self.labels = []
        
        class_set = set()

        for file in paths:
            cls = int(file.stem.split("_")[0])
            self.files.append(str(file))
            self.labels.append(cls)
            class_set.add(cls)
            
        self.classes = list(class_set)


    def __len__(self):
        return len(self.files)
        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_path: Path = self.files[idx]

        augmentation_str = ""
        label = self.labels[idx]
        cache_folder = f"{self.root_dir}/.image_cache"
        
        created_augmentation = False
        
        if self.augmentation:
            if self.forced_aug_index >= 0:
                augmentation_index = self.forced_aug_index
            else:
                augmentation_index = random.randint(0, self.augmentation_count - 1)
            dither_str = "" if self.dither else "_nodither"
            augmentation_str = f"_{self.augmentation}{dither_str}_{augmentation_index}"
            orig_file_path = Path(file_path)
            file_path = get_augmentation_path(orig_file_path, self.augmentation + dither_str, augmentation_index)
            if not file_path.exists() or file_path.stat().st_size == 0:
                os.makedirs(f'{self.root_dir}/augmentations', exist_ok=True)
                create_augmentation(orig_file_path, self.augmentation, augmentation_index, dither=self.dither)
                created_augmentation = True
                

        os.makedirs(cache_folder, exist_ok=True)
        
        
        def get_cache_filename(s):
            return f"{cache_folder}/{idx}_{self.W}_{self.H}_{self.T}{augmentation_str}_{s}.npz"
        
        
        cached_image_path = get_cache_filename('_'.join(self.dimensions) + "_image")
        
        def dimension_exists(s):
            dim_path = Path(get_cache_filename(s))
            return dim_path.exists() and not created_augmentation and dim_path.stat().st_size > 0
        
        
        def get_or_create_cached_value(s, calc):
            if self.use_subimage_cache:
                path = Path(get_cache_filename(s))
                if path.exists():
                    try:
                        return np.load(path)['arr_0']
                    except Exception as e:
                        print(f"Exception when loading sub image cache {path}, {e}")
                value = calc()                
                np.savez_compressed(path, value)
                return value
            else:
                return calc()
        
        got_image = False
        if self.use_image_cache and Path(cached_image_path).exists() and not created_augmentation:
            try:
                image = np.load(cached_image_path)['arr_0']
                got_image = True
            except Exception as e:
                print(f"Exception when loading image cache {cached_image_path}, {e}")                
            
        if not got_image:

            data, events, w, h, pos, neg, min_t, max_t, max_x, max_y = None, None, None, None, None, None, None, None, None, None            
            
            def load_data_if_necessary():
                nonlocal data, events, w, h, pos, neg, min_t, max_t, max_x, max_y
                if data:
                    return
                data = np.load(file_path)
                
                events = data["events"]
                w, h = data['shape']
            
                pos = events[events['p'] == True]
                neg = events[events['p'] == False]
                
                if len(events) == 0:
                    print(f"Event counts 0 for {idx}")
                elif len(pos) == 0:
                    print(f"Positive event counts 0 for {idx}")
                elif len(neg) == 0:
                    print(f"Negative event counts 0 for {idx}")
                
                min_t = events[0]['t']
                max_t = events[-1]['t']
                    
                max_x = w - 1
                max_y = h - 1

            T = self.T
            
            def load_data_if_not_dimension_data_exists(s):
                if not dimension_exists(s):
                    load_data_if_necessary()
            
            def compute_or_get_dimension(dim):
                calc = None
                if dim == Dimension.CSTR_2:
                    load_data_if_not_dimension_data_exists(dim)
                    xy_mean_t_pol = compute_or_get_dimension(Dimension.XY_MEAN_T_POL)
                    return np.concatenate((xy_mean_t_pol[1:2], np.zeros((1, w, h)), xy_mean_t_pol[0:1]))
                if dim == Dimension.CSTR_3:
                    load_data_if_not_dimension_data_exists(dim)
                    xy_mean_t_pol = compute_or_get_dimension(Dimension.XY_MEAN_T_POL)
                    return np.concatenate((xy_mean_t_pol[1:], compute_or_get_dimension(Dimension.XY_DENSITY_BIN), xy_mean_t_pol[0:1]))
                if dim == Dimension.XY_ZERO:
                    load_data_if_not_dimension_data_exists(dim)
                    return np.zeros((1, w, h))
                if dim == Dimension.XT_ZERO:
                    load_data_if_not_dimension_data_exists(dim)
                    return np.zeros((1, w, T))
                if dim == Dimension.TY_ZERO:
                    load_data_if_not_dimension_data_exists(dim)
                    r = np.zeros((1, T, h))
                    return r
                if dim == Dimension.XY_T_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.XY_T_POS), compute_or_get_dimension(Dimension.XY_T_NEG)))
                if dim == Dimension.XY_T_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_timestamps(events, (w, h))
                if dim == Dimension.XY_T_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_timestamps(pos, (w, h))
                if dim == Dimension.XY_T_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_timestamps(neg, (w, h))
                if dim == Dimension.XY_MAX_T_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.XY_MAX_T_POS), compute_or_get_dimension(Dimension.XY_MAX_T_NEG)))
                if dim == Dimension.XY_MAX_T_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_max_timestamps(events, (w, h))
                if dim == Dimension.XY_MAX_T_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_max_timestamps(pos, (w, h))
                if dim == Dimension.XY_MAX_T_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_max_timestamps(neg, (w, h))
                if dim == Dimension.XY_MEAN_T_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.XY_MEAN_T_POS), compute_or_get_dimension(Dimension.XY_MEAN_T_NEG)))
                if dim == Dimension.XY_MEAN_T_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_mean_timestamps(events, (w, h))
                if dim == Dimension.XY_MEAN_T_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_mean_timestamps(pos, (w, h))
                if dim == Dimension.XY_MEAN_T_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_mean_timestamps(neg, (w, h))
                if dim == Dimension.XT_MEAN_Y_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.XT_MEAN_Y_POS), compute_or_get_dimension(Dimension.XT_MEAN_Y_NEG)))
                if dim == Dimension.XT_MEAN_Y_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_xt_mean_ys(events, w, T)
                if dim == Dimension.XT_MEAN_Y_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_xt_mean_ys(pos, w, T)
                if dim == Dimension.XT_MEAN_Y_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_xt_mean_ys(neg, w, T)
                if dim == Dimension.TY_MEAN_X_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.TY_MEAN_X_POS), compute_or_get_dimension(Dimension.TY_MEAN_X_NEG)))
                if dim == Dimension.TY_MEAN_X_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_ty_mean_xs(events, h, T)
                if dim == Dimension.TY_MEAN_X_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_ty_mean_xs(pos, h, T)
                if dim == Dimension.TY_MEAN_X_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: get_ty_mean_xs(neg, h, T)
                if dim == Dimension.XY_HIST_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(pos['x'], pos['y'], bins=[w, h], range=[[0, max_x], [0, max_y]])[0][np.newaxis, :, :]
                if dim == Dimension.XY_HIST_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(neg['x'], neg['y'], bins=[w, h], range=[[0, max_x], [0, max_y]])[0][np.newaxis, :, :]
                if dim == Dimension.XY_HIST_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(events['x'], events['y'], bins=[w, h], range=[[0, max_x], [0, max_y]])[0][np.newaxis, :, :]
                if dim == Dimension.XY_HIST_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.XY_HIST_POS), compute_or_get_dimension(Dimension.XY_HIST_NEG)))
                if dim == Dimension.XT_HIST_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(pos['x'], pos['t'], bins=[w, T], range=[[0, max_x], [min_t, max_t]])[0][np.newaxis, :, :]
                if dim == Dimension.XT_HIST_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(neg['x'], neg['t'], bins=[w, T], range=[[0, max_x], [min_t, max_t]])[0][np.newaxis, :, :]
                if dim == Dimension.XT_HIST_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(events['x'], events['t'], bins=[w, T], range=[[0, max_x], [min_t, max_t]])[0][np.newaxis, :, :]
                if dim == Dimension.XT_HIST_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.XT_HIST_POS), compute_or_get_dimension(Dimension.XT_HIST_NEG)))
                if dim == Dimension.YT_HIST_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(pos['y'], pos['t'], bins=[h, T], range=[[0, max_y], [min_t, max_t]])[0][np.newaxis, :, :]
                if dim == Dimension.YT_HIST_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(neg['y'], neg['t'], bins=[h, T], range=[[0, max_y], [min_t, max_t]])[0][np.newaxis, :, :]
                if dim == Dimension.YT_HIST_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(events['y'], events['t'], bins=[h, T], range=[[0, max_y], [min_t, max_t]])[0][np.newaxis, :, :]
                if dim == Dimension.YT_HIST_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.YT_HIST_POS), compute_or_get_dimension(Dimension.YT_HIST_NEG)))
                if dim == Dimension.TY_HIST_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(pos['t'], pos['y'], bins=[T, h], range=[[min_t, max_t], [0, max_y]])[0][np.newaxis, :, :]
                if dim == Dimension.TY_HIST_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(neg['t'], neg['y'], bins=[T, h], range=[[min_t, max_t], [0, max_y]])[0][np.newaxis, :, :]
                if dim == Dimension.TY_HIST_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(events['t'], events['y'], bins=[T, h], range=[[min_t, max_t], [0, max_y]])[0][np.newaxis, :, :]
                if dim == Dimension.TY_HIST_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.TY_HIST_POS), compute_or_get_dimension(Dimension.TY_HIST_NEG)))
                if dim == Dimension.TX_HIST_POS:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(pos['t'], pos['x'], bins=[T, h], range=[[min_t, max_t], [0, max_x]])[0][np.newaxis, :, :]
                if dim == Dimension.TX_HIST_NEG:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(neg['t'], neg['x'], bins=[T, h], range=[[min_t, max_t], [0, max_x]])[0][np.newaxis, :, :]
                if dim == Dimension.TX_HIST_BIN:
                    load_data_if_not_dimension_data_exists(dim)
                    calc = lambda: np.histogram2d(events['t'], events['x'], bins=[T, h], range=[[min_t, max_t], [0, max_x]])[0][np.newaxis, :, :]
                if dim == Dimension.TX_HIST_POL:
                    return np.concatenate((compute_or_get_dimension(Dimension.TX_HIST_POS), compute_or_get_dimension(Dimension.TX_HIST_NEG)))
                if dim == Dimension.TY_DENSITY_POS:
                    return normalize(compute_or_get_dimension(Dimension.TY_HIST_POS))
                if dim == Dimension.TY_DENSITY_NEG:
                    return normalize(compute_or_get_dimension(Dimension.TY_HIST_NEG))
                if dim == Dimension.TY_DENSITY_BIN:
                    return normalize(compute_or_get_dimension(Dimension.TY_HIST_BIN))
                if dim == Dimension.TY_DENSITY_POL:
                    return normalize(compute_or_get_dimension(Dimension.TY_HIST_POL))
                if dim == Dimension.YT_DENSITY_POS:
                    return normalize(compute_or_get_dimension(Dimension.YT_HIST_POS))
                if dim == Dimension.YT_DENSITY_NEG:
                    return normalize(compute_or_get_dimension(Dimension.YT_HIST_NEG))
                if dim == Dimension.YT_DENSITY_BIN:
                    return normalize(compute_or_get_dimension(Dimension.YT_HIST_BIN))
                if dim == Dimension.YT_DENSITY_POL:
                    return normalize(compute_or_get_dimension(Dimension.YT_HIST_POL))
                if dim == Dimension.XT_DENSITY_POS:
                    return normalize(compute_or_get_dimension(Dimension.XT_HIST_POS))
                if dim == Dimension.XT_DENSITY_NEG:
                    return normalize(compute_or_get_dimension(Dimension.XT_HIST_NEG))
                if dim == Dimension.XT_DENSITY_BIN:
                    return normalize(compute_or_get_dimension(Dimension.XT_HIST_BIN))
                if dim == Dimension.XT_DENSITY_POL:
                    return normalize(compute_or_get_dimension(Dimension.XT_HIST_POL))
                if dim == Dimension.TX_DENSITY_POS:
                    return normalize(compute_or_get_dimension(Dimension.TX_HIST_POS))
                if dim == Dimension.TX_DENSITY_NEG:
                    return normalize(compute_or_get_dimension(Dimension.TX_HIST_NEG))
                if dim == Dimension.TX_DENSITY_BIN:
                    return normalize(compute_or_get_dimension(Dimension.TX_HIST_BIN))
                if dim == Dimension.TX_DENSITY_POL:
                    return normalize(compute_or_get_dimension(Dimension.TX_HIST_POL))
                if dim == Dimension.XY_DENSITY_POS:
                    return normalize(compute_or_get_dimension(Dimension.XY_HIST_POS))
                if dim == Dimension.XY_DENSITY_NEG:
                    return normalize(compute_or_get_dimension(Dimension.XY_HIST_NEG))
                if dim == Dimension.XY_DENSITY_BIN:
                    return normalize(compute_or_get_dimension(Dimension.XY_HIST_BIN))
                if dim == Dimension.XY_DENSITY_POL:
                    return normalize(compute_or_get_dimension(Dimension.XY_HIST_POL))
                if dim == Dimension.TY_LOG10PDENSITY_POS:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.TY_HIST_POS)))
                if dim == Dimension.TY_LOG10PDENSITY_NEG:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.TY_HIST_NEG)))
                if dim == Dimension.TY_LOG10PDENSITY_BIN:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.TY_HIST_BIN)))
                if dim == Dimension.TY_LOG10PDENSITY_POL:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.TY_HIST_POL)))
                if dim == Dimension.YT_LOG10PDENSITY_POS:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.YT_HIST_POS)))
                if dim == Dimension.YT_LOG10PDENSITY_NEG:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.YT_HIST_NEG)))
                if dim == Dimension.YT_LOG10PDENSITY_BIN:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.YT_HIST_BIN)))
                if dim == Dimension.YT_LOG10PDENSITY_POL:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.YT_HIST_POL)))
                if dim == Dimension.XT_LOG10PDENSITY_POS:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.XT_HIST_POS)))
                if dim == Dimension.XT_LOG10PDENSITY_NEG:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.XT_HIST_NEG)))
                if dim == Dimension.XT_LOG10PDENSITY_BIN:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.XT_HIST_BIN)))
                if dim == Dimension.XT_LOG10PDENSITY_POL:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.XT_HIST_POL)))
                if dim == Dimension.TX_LOG10PDENSITY_POS:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.TX_HIST_POS)))
                if dim == Dimension.TX_LOG10PDENSITY_NEG:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.TX_HIST_NEG)))
                if dim == Dimension.TX_LOG10PDENSITY_BIN:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.TX_HIST_BIN)))
                if dim == Dimension.TX_LOG10PDENSITY_POL:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.TX_HIST_POL)))
                if dim == Dimension.XY_LOG10PDENSITY_POS:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.XY_HIST_POS)))
                if dim == Dimension.XY_LOG10PDENSITY_NEG:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.XY_HIST_NEG)))
                if dim == Dimension.XY_LOG10PDENSITY_BIN:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.XY_HIST_BIN)))
                if dim == Dimension.XY_LOG10PDENSITY_POL:
                    return normalize(np.log10(1 + compute_or_get_dimension(Dimension.XY_HIST_POL)))

                if not calc:
                    raise ValueError(f"Could not calculate '{dim}'")
                return get_or_create_cached_value(dim, calc)
            
            image_dims = []
            for dim in self.dimensions:
                val = compute_or_get_dimension(dim)
                
                contains_nan = np.isnan(val).any()
                contains_inf = np.isinf(val).any()
                
                # Emit warning printouts when finding NaN or Inf in the dimension data
                if contains_nan:
                    print(f'Warning: Found NaN when processing dimension {dim}, idx {idx}')
                if contains_inf:
                    print(f'Warning: Found Inf when processing dimension {dim}, idx {idx}')
                if contains_inf or contains_nan:
                    print(f'NaN/Inf debug stats: All events len {len(events)}, positive events len {len(pos)}, negative events len {len(neg)}')
                dim_val = zoom(val, (1, self.W / val.shape[1], self.H / val.shape[2]), order=1)
                image_dims.append(dim_val)

            image = np.concatenate(image_dims)
            if self.use_image_cache:
                np.savez_compressed(cached_image_path, image)
                
        
        image = image.astype(np.float32)
        
        image_tensor = torch.tensor(image).float()

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label


