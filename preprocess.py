import argparse
import pandas as pd
import random
import shutil
import os
import numpy as np
from pathlib import Path
import scipy.io as scio
from multiprocessing import Pool
import random
from collections import defaultdict
from dotenv import load_dotenv
import json
from tqdm import tqdm
from typing import Optional, Tuple, BinaryIO, List, Optional, Sequence, Union, Generator, Iterable
import struct
import re
import dv_processing as dv


from frame_dataset import dataset_names
import util


event_sa_dtype = [('x', '<u2'), ('y', '<u2'), ('t', '<u8'), ('p', '?')]

PathLike = Union[str, Path, os.PathLike[str]]


def normalize(arr):
    min_val = arr.min()
    max_val = arr.max()
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr
 

def split_dataset(folder, splits, split_fractions):
    files = list(Path(folder).glob('*.npz'))
    
    label_paths = defaultdict(list)

    for path in files:
        label_paths[path.stem.split("_")[0]].append(path)

    split_dict = defaultdict(list)

    for _, label_files in label_paths.items():
        random.shuffle(label_files)
        split_files = util.split_list_by_fractions(label_files, split_fractions)
        for split, group_files in zip(splits, split_files):
            split_dict[split].extend(group_files)

    for split, split_files in split_dict.items():
        split_folder = f'{folder}/{split}'
        os.makedirs(split_folder, exist_ok=True)
        for file in split_files:
            shutil.move(file, split_folder)        


def parse_aer_header_from_file(filename: str | Path) -> Tuple[Optional[float], int]:
    """
    Parse header lines that start with '#' and extract an !AER-DAT version if present.
    Returns (data_version, data_start), where:
      - data_version is a float if found, else None
      - data_start is the byte offset where non-comment data begins
    """
    path = Path(filename).expanduser()
    data_version: Optional[float] = None

    # Pattern like: "# ... !AER-DAT 1.23 ..." (allowing optional spaces)
    version_re = re.compile(rb"!AER-DAT\s*([0-9]+(?:\.[0-9]+)?)")

    with path.open("rb") as f:
        # Read byte-by-byte only to check for leading '#', then use readline for the rest
        first = f.read(1)

        while first == b"#":
            # We already consumed the '#', so combine it with the rest of the line
            line_rest = f.readline()
            header_line = first + line_rest  # still bytes

            m = version_re.search(header_line)
            if m:
                try:
                    data_version = float(m.group(1).decode("ascii"))
                except ValueError:
                    # If decoding or conversion fails, leave data_version as None
                    pass

            # Peek the next byte to see if the next line is also a comment
            first = f.read(1)

        # We read one byte too far (the first non-# byte), so rewind by one if not EOF
        if first:
            f.seek(-1, 1)

        data_start = f.tell()
    return data_version, data_start


def get_aer_events_from_file(filename: PathLike) -> np.ndarray:
    """
    Read AER events from a file whose header has already been parsed by
    `parse_aer_header_from_file`.

    Supported formats (mirrors the original logic, but more explicit/robust):
      - 2.x <= version < 3.0  : big-endian 32-bit (address,timeStamp) pairs,
                                contiguous until EOF.
      - version > 3.0         : little-endian 32-bit pairs grouped into
                                batches, each preceded by a 28-byte header
                                where uint32 capacity is at byte offset [16:20]
                                in little-endian.

    Returns
    -------
    np.ndarray
        Structured array with dtype [('address','>u4' or '<u4'), ('timeStamp', ...)].
        May be empty if the file has no events.

    Raises
    ------
    FileNotFoundError, ValueError, NotImplementedError
    """
    # Normalize/expand path
    path = Path(os.path.expanduser(str(filename)))

    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    data_version, data_start = parse_aer_header_from_file(path)

    if data_version is None:
        raise ValueError(
            "AER-DAT version not found in header (data_version=None).")

    # Open once; always close via context manager
    with path.open("rb") as f:
        # Seek to the start of the binary payload (after comments)
        f.seek(data_start)

        # 2.x format: big-endian contiguous stream
        if 2.0 <= data_version < 3.0:
            event_dtype = np.dtype([("address", ">u4"), ("timeStamp", ">u4")])
            # np.fromfile reads until EOF with given dtype
            try:
                events = np.fromfile(f, dtype=event_dtype, count=-1)
            except (OSError, ValueError) as e:
                raise ValueError(
                    f"Failed reading events (v{data_version:.2f}): {e}") from e
            return events

        # >3.0 format: little-endian events grouped by headers
        elif data_version > 3.0:
            event_dtype = np.dtype([("address", "<u4"), ("timeStamp", "<u4")])
            itemsize = event_dtype.itemsize  # 8 bytes per event
            batches: list[np.ndarray] = []

            # Pre-compute file size to sanity-check capacities
            try:
                file_size = path.stat().st_size
            except OSError:
                file_size = None  # Fallback if stat fails

            while True:
                header = f.read(28)
                if not header:
                    break  # clean EOF
                if len(header) < 28:
                    # Truncated header at end of file
                    raise ValueError(
                        f"Truncated batch header (expected 28 bytes, got {len(header)})."
                    )

                # Capacity is a little-endian uint32 at bytes [16:20]
                # (Make endianness explicit—avoid native-default 'I'.)
                capacity = struct.unpack("<I", header[16:20])[0]

                if capacity == 0:
                    # Zero-sized batch: allowed but skip reading events
                    continue

                # Sanity-check capacity versus remaining bytes (if file size is known)
                if file_size is not None:
                    cur = f.tell()
                    remaining = max(file_size - cur, 0)
                    max_events = remaining // itemsize
                    if capacity > max_events:
                        raise ValueError(
                            f"Batch declares {capacity} events but only {max_events} "
                            f"fit in remaining bytes ({remaining}B)."
                        )

                try:
                    batch = np.fromfile(f, dtype=event_dtype, count=capacity)
                except (OSError, ValueError) as e:
                    raise ValueError(
                        f"Failed reading batch of {capacity} events: {e}") from e

                if batch.size != capacity:
                    raise ValueError(
                        f"Unexpected EOF reading batch: expected {capacity}, got {batch.size}."
                    )

                batches.append(batch)

            if not batches:
                # Return an empty structured array with the right dtype
                return np.empty(0, dtype=event_dtype)

            return np.concatenate(batches, axis=0)

        else:
            # Keeping parity with the original behavior (version == 3 not supported)
            raise NotImplementedError(
                f"AER-DAT version {data_version} is not supported by this reader."
            )


def parse_dvs_128(filename):
    all_events = get_aer_events_from_file(filename)
    all_addr = all_events["address"]
    t = all_events["timeStamp"]

    x = (all_addr >> 8) & 0x007F
    y = (all_addr >> 1) & 0x007F
    p = all_addr & 0x1

    dtype = np.dtype([("x", np.uint16), ("y", np.uint16),
                     ("t", np.uint64), ("p", bool)])
    return (128, 128), util.make_structured_array(x, y, t, p, dtype=dtype)


def parse_dvs_ibm(filename):
    all_events = get_aer_events_from_file(filename)
    all_addr = all_events["address"]
    t = all_events["timeStamp"]

    x = (all_addr >> 17) & 0x00001FFF
    y = (all_addr >> 2) & 0x00001FFF
    p = (all_addr >> 1) & 0x00000001

    dtype = np.dtype([("x", np.uint16), ("y", np.uint16),
                     ("t", np.uint64), ("p", bool)])
    return (128, 128), util.make_structured_array(x, y, t, p, dtype=dtype)


def read_mnist_file(bin_file, dtype):
    with open(bin_file, "rb") as fp:
        raw = np.fromfile(fp, dtype=np.uint8).astype(np.uint32)

    x_all = raw[0::5]
    y_all = raw[1::5]
    p_all = (raw[2::5] & 128) >> 7  # bit 7
    ts_all = (raw[4::5]) | ((raw[2::5] & 127) << 16) | (raw[3::5] << 8)

    # Process overflow events
    time_increment = 2**13
    overflow_indices = np.where(y_all == 240)[0]
    for overflow_index in overflow_indices:
        ts_all[overflow_index:] += time_increment

    # Not overflow
    td_indices = np.where(y_all != 240)[0]

    return util.make_structured_array(
        x_all[td_indices],
        y_all[td_indices],
        ts_all[td_indices],
        p_all[td_indices],
        dtype=dtype,
    )

def parse_aedat4_dv_processing(in_file):
    capture = dv.io.MonoCameraRecording(in_file)
    W, H = capture.getEventResolution()
    all_events = dv.EventStore()
    while capture.isRunning():

        # Read batch of events
        batch_events = capture.getNextEventBatch()

        if batch_events is not None:
            all_events.add(batch_events)
    evs = all_events.numpy()
    # print(evs.dtype)
    # dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.uint64), ("p", bool)])
    # print(dtype)
    dst = np.dtype({
    'names':   ['x',   'y',   't',   'p'],
    'formats': ['<u2', '<u2', '<u8', '?'],
    'offsets': [8,     10,    0,     12],
    'itemsize': 16
    })
    array = evs.view(dst)
    return (W, H), array



def parse_dat_header(f: BinaryIO) -> Tuple[int, int, int, Tuple[Optional[int], Optional[int]]]:
    """
    Parse the header of a .dat file

    Args:
        f: Binary file handle opened in 'rb' mode.

    Returns:
        bod: int position of the file cursor after the header
        ev_type: int type of event
        ev_size: int size of event in bytes
        size: (height, width) tuple of int or None
    """
    f.seek(0, os.SEEK_SET)

    bod = 0
    num_comment_line = 0
    height: Optional[int] = None
    width: Optional[int] = None

    # Parse header lines that start with b"% "
    while True:
        bod = f.tell()
        line = f.readline()
        if not line:
            # EOF reached
            break
        if line[:2] != b"% ":
            # First non-comment line
            break

        words = line.split()
        if len(words) > 1:
            key = words[1]
            if key == b"Height" and len(words) >= 3:
                height = int(words[2])
            elif key == b"Width" and len(words) >= 3:
                width = int(words[2])
            # b"Date" (and other keys) are ignored intentionally
        num_comment_line += 1

    # Position file at the first non-comment byte
    f.seek(bod, os.SEEK_SET)

    if num_comment_line > 0:
        # Read event type and size (1 byte each)
        ev_type = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
        ev_size = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
    else:
        # Compatibility with very old files: synthesize defaults
        ev_type = 0
        dtype = [('t', 'u4'), ('_', 'i4')]
        ev_size = sum(int(n[-1]) for _, n in dtype)

    bod = f.tell()
    return bod, ev_type, ev_size, (height, width)


def load_td_data(
    filename: PathLike,
    ev_count: Optional[int] = None,
    ev_start: int = 0,
) -> np.ndarray:
    """
    Load TD data from a .dat file and (optionally) untangle x/y/p from the '_' field.

    Args:
        filename: Path to the .dat file.
        ev_count: Number of events to read. None -> read all remaining.
        ev_start: Number of events to skip from the first event (after the header).

    Returns:
        A NumPy structured array. If the input dtype contains the '_' field,
        the returned array replaces '_' with unsigned ('u2') fields: x, y, p.
    """
    if ev_start < 0:
        raise ValueError("ev_start must be >= 0")
    if ev_count is not None and ev_count < 0:
        raise ValueError("ev_count must be None or >= 0")

    # NOTE: parse_dat_header is provided by you.
    with open(filename, "rb") as f:
        bod, ev_type, ev_size, _size = parse_dat_header(f)

        # Move to the start of the requested event
        if ev_start:
            # Use absolute seek from the beginning of data (safer & clearer than whence=1)
            f.seek(bod + ev_start * ev_size, 0)

        # Base dtype we expect from the file (matches your original)
        base_dtype: List[Tuple[str, str]] = [("t", "u4"), ("_", "i4")]

        # NumPy uses count=-1 to mean "read all"
        count = -1 if ev_count is None else ev_count

        # Read the raw events
        # type: ignore[arg-type]
        dat = np.fromfile(f, dtype=base_dtype, count=count)

    # If '_' exists, untangle x/y/p via bit ops
    if "_" in dat.dtype.names:  # type: ignore[truthy-bool]
        word = dat["_"].astype(np.uint32)

        # ---- Bitfield constants for '_' word -------------------------------------------------
        _X_MASK: np.uint32 = np.uint32(0x00003FFF)   # 14 bits: 0..13
        _Y_MASK: np.uint32 = np.uint32(0x0FFFC000)   # 14 bits: 14..27
        _P_MASK: np.uint32 = np.uint32(0x10000000)   # 1  bit : 28
        _Y_SHIFT: int = 14
        _P_SHIFT: int = 28

        x = (word & _X_MASK).astype(np.uint16)
        y = ((word & _Y_MASK) >> _Y_SHIFT).astype(np.uint16)
        p = ((word & _P_MASK) >> _P_SHIFT).astype(np.uint16)

        # Build the new dtype by replacing '_' with x/y/p (unsigned 2-byte each)
        new_dtype: List[Tuple[str, str]] = []
        for name, kind in base_dtype:
            if name == "_":
                new_dtype.extend([("x", "u2"), ("y", "u2"), ("p", "u2")])
            else:
                new_dtype.append((name, kind))

        return transfer_dat(dat, new_dtype, xyp=(x, y, p))

    # If '_' is not present, just pass-through (still normalized in _dat_transfer)
    return transfer_dat(dat, tuple([("t", "u4")]))


def transfer_dat(
    dat: np.ndarray,
    new_dtype: Sequence[Tuple[str, str]],
    *,
    xyp: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """
    Transfer fields from `dat` into a new structured array described by `new_dtype`.

    - If `xyp` is provided, `new_dtype` must contain ('x','u2'), ('y','u2'), ('p','u2').
    - Any fields present in `new_dtype` that also exist in `dat` are copied over.
    - Missing fields (except x/y/p, which come from `xyp`) raise clear errors.

    Args:
        dat: Structured array as read from file.
        new_dtype: Target dtype as sequence of (name, kind) pairs.
        xyp: Optional tuple (x, y, p) arrays, each uint16, extracted from old '_' field.

    Returns:
        New structured array with fields defined by `new_dtype`.
    """
    # Normalize dtype & allocate
    target_dtype = np.dtype(list(new_dtype))

    out = np.empty(dat.shape[0], dtype=target_dtype)

    # If xyp provided, ensure required fields exist in target dtype
    if xyp is not None:
        for field in ("x", "y", "p"):
            if field not in out.dtype.names:
                raise ValueError(
                    f"new_dtype is missing required '{field}' field while xyp was provided."
                )
        out["x"] = xyp[0]
        out["y"] = xyp[1]
        out["p"] = xyp[2]

    # Copy overlapping fields (excluding x/y/p which we already set, if present)
    src_fields = set(dat.dtype.names or [])
    tgt_fields = set(out.dtype.names or [])
    copy_fields = (tgt_fields & src_fields) - {"x", "y", "p"}

    for name in copy_fields:
        out[name] = dat[name]

    # Validate that all target fields are satisfied either from dat or from xyp
    missing = [
        name
        for name in tgt_fields
        if name not in copy_fields and name not in {"x", "y", "p"}
    ]
    if missing:
        # If these are exactly x/y/p and xyp was given, we already populated them.
        # Otherwise, it's a genuine mismatch.
        unresolved = [m for m in missing if m not in {"x", "y", "p"}]
        if unresolved:
            raise ValueError(
                f"Cannot populate target fields {unresolved!r}: not present in source and not provided."
            )

    return out



def extract_action_events(events, actions):
    current_index = 0
    c, start_us, end_us = actions[0]

    result = []
    current_events = []
    
    for ev in events:        
        t = ev[2]
        if t >= end_us:
            # print(f"Extracted action {c}, {start_us} -> {end_us}")
            arr = np.array(current_events, dtype=event_sa_dtype)
            result.append((c, arr))
            current_index += 1
            if current_index >= len(actions):
                break
            current_events = []
            c, start_us, end_us = actions[current_index]
        
        if t >= start_us:
            current_events.append(ev)
    return result


def split_events(events, frame_ms, step_ms):
    min_t = events[0]['t']
    MS_PER_SECOND = 1000
    max_t = min_t + frame_ms * MS_PER_SECOND
    
    final_t = events[-1]['t']
    
    result = []
    while True:
        if max_t > final_t:
            break
        frame_events = events[(events['t'] >= min_t) & (events['t'] < max_t)]        
        min_t += step_ms * MS_PER_SECOND
        max_t = min_t + frame_ms * MS_PER_SECOND
        if len(frame_events) == 0:
            print("No frame events", min_t, max_t)
            break
        result.append(frame_events)
    return result
    


def create_split_json(root_folder, output_file="split.json"):
    folder_dict = {}

    # Walk only the first level (root_folder)
    for entry in os.scandir(root_folder):
        if entry.is_dir():  # only process folders
            folder_name = entry.name
            folder_path = entry.path

            # List files only, ignore subfolders
            files = [f.name for f in os.scandir(folder_path) if f.is_file()]
            
            folder_dict[folder_name] = files

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(folder_dict, f, indent=4, ensure_ascii=False)



def organize_from_json(mapping_json_path, root_folder):
    """
    Read a JSON mapping of {subfolder_name: [file_name, ...]} and organize files.

    For each key (subfolder) in the JSON:
      - create the subfolder under `root_folder` if it doesn't exist
      - clear that subfolder of all *files* (leaves any subdirectories untouched)
      - move each listed file from the root folder into that subfolder

    The function FAILS if:
      - any listed file is missing from the root folder
      - a listed item includes a path separator (must be a plain file name)
      - the same file is assigned to more than one subfolder

    Args:
        mapping_json_path (str | Path): Path to the JSON file.
        root_folder (str | Path): Path to the directory containing the flat files.
    """
    mapping_json_path = Path(mapping_json_path)
    root_folder = Path(root_folder)

    # --- Load and validate JSON ---
    try:
        mapping = json.loads(mapping_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {mapping_json_path}: {e}") from e

    if not isinstance(mapping, dict):
        raise ValueError("JSON root must be an object/dict of {subfolder: [files,...]}")

    # Ensure values are lists of plain file names (no separators)
    all_assigned = []
    for subfolder, files in mapping.items():
        if not isinstance(subfolder, str):
            raise ValueError("All keys (subfolder names) must be strings.")
        if not isinstance(files, list):
            raise ValueError(f"Value for '{subfolder}' must be a list of file names.")
        for name in files:
            if not isinstance(name, str):
                raise ValueError(f"All file names must be strings (found {type(name)} in '{subfolder}').")
            if ("/" in name) or ("\\" in name):
                raise ValueError(f"File names must not contain path separators: '{name}' in '{subfolder}'.")
        all_assigned.extend(files)

    # Check duplicates (a file cannot appear in multiple buckets)
    seen = set()
    duplicates = sorted({f for f in all_assigned if f in seen or seen.add(f)})
    if duplicates:
        raise RuntimeError(f"File(s) assigned to multiple subfolders: {duplicates}")

    # --- Check existence of all required files in the ROOT (not inside subfolders) ---
    missing = []
    for name in all_assigned:
        p = root_folder / name
        if not p.exists() or not p.is_file():
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            "These files are missing from the root folder (or are not plain files): "
            + ", ".join(missing)
        )

    # --- Organize: create/clear subfolders, then move files ---
    for subfolder, files in mapping.items():
        target_dir = root_folder / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)

        # Clear only FILES from the target subfolder (leave subdirectories untouched)
        for entry in target_dir.iterdir():
            try:
                if entry.is_file() or entry.is_symlink():
                    entry.unlink()
            except Exception as e:
                raise RuntimeError(f"Failed clearing '{entry}': {e}") from e

        # Move listed files from the ROOT into the subfolder
        for name in files:
            src = root_folder / name
            dst = target_dir / name
            # At this point src is guaranteed to exist & be a file (we checked)
            try:
                shutil.move(str(src), str(dst))
            except Exception as e:
                raise RuntimeError(f"Failed moving '{src}' -> '{dst}': {e}") from e


def execute_split(dataset_name):
    split_json_path = f"./splits/test_split_{dataset_name}.json"
    organize_from_json(split_json_path, util.get_dataset_folder(dataset_name))

    
def get_orig_dataset_folder_or_error(subfolder):
    
    base = Path(f"{util.get_orig_dataset_folder()}/{subfolder}")
    if not base.exists():
        raise ValueError(f"Could not find original dataset subfolder {base}. You need to download the datasets yourself and place in that folder")
    return base

    
    
def create_dvs_lip_data(dataset_name, parallel=False):
    
    slices = ["train", "test"]

    all_labels_set = set()

    orig_ds_folder = get_orig_dataset_folder_or_error("DVS-Lip")
    orig_train_folder = orig_ds_folder / "train"
    ot_folders = list(orig_train_folder.glob('*'))
    for ot_folder in ot_folders:
        all_labels_set.add(ot_folder.stem)

    all_labels_list = sorted(all_labels_set)

    index = 0

    if parallel:
        print(f"Parallel mode not implemented for {dataset_name}")

    for slice in slices:
        lip_folder = orig_ds_folder / slice
        folders = list(lip_folder.glob('*'))
        base_output_folder = f"{util.get_data_folder()}/{dataset_name}/action_events/{slice}"
        os.makedirs(base_output_folder, exist_ok=True)
        for folder in folders:
            string_label = folder.stem
            label = all_labels_list.index(folder.stem)
            files = list(folder.glob('*.npy'))
            for file in tqdm(files, desc=f"Processing {string_label} in {slice} split"):
                data = np.load(file)
                events = []
                for ev in data:
                    events.append((ev[1], ev[2], ev[0], ev[3] == 1))
                np_events = np.array(events, dtype=event_sa_dtype)
                output_file = f"{base_output_folder}/{label}_{string_label}_{file.stem}_{index}.npz"
                np.savez_compressed(output_file, events=np_events, shape=(128, 128))
                index += 1
   


def process_daily_dvs_200_data(file, output_file):
    shape, raw_events = parse_aedat4_dv_processing(file)
    h, w = shape
    events = []
    for ev in raw_events:
        events.append((ev[0], ev[1], ev[2], ev[3]))
    events.sort(key=lambda ev: ev[2])
    np_events = np.array(events, dtype=event_sa_dtype)
    np.savez_compressed(output_file, events=np_events, shape=(w, h))





def create_daily_dvs_200_data(dataset_name, parallel=False):
    base = get_orig_dataset_folder_or_error("DailyDvs-200")

    base_output_folder = f"{util.get_data_folder()}/{dataset_name}/action_events"
    os.makedirs(base_output_folder, exist_ok=True)

    def read_split(file):
        with open(file, 'r') as file:
            lines = file.readlines()
        return lines

    split_files = defaultdict(list)
    for split in ['train', 'test', 'val']:
        split_files[split] = read_split(base / f'{split}.txt')
    
    
    arguments = []

    val_split_data = {
        'training': [],
        'validation': [],
    }

    for split in ['train', 'test', 'val']:
        lines = read_split(base / f'{split}.txt')
        orig_split = split
        if split == 'val':
            split = 'train' # The validation samples are expected to be in the train folder
        split_folder = f'{base_output_folder}/{split}'
        os.makedirs(split_folder, exist_ok=True)
        for line in lines:
            parts = line.split()
            orig_file = parts[0]
            label = int(parts[1])
            parts = orig_file.split('/')
            input_file = f'{base}/action_{str(label+1).zfill(3)}/{parts[1]}'
            output_basename = f'{label}_{parts[1]}.npz'
            output_file = f'{split_folder}/{output_basename}'
            output_filename = Path(output_file).name
            arguments.append((input_file, output_file))
            if orig_split == 'val':
                val_split_data['validation'].append(output_filename)
            elif orig_split == 'train':
                val_split_data['training'].append(output_filename)

    if parallel:
        with Pool() as pool:
            pool.starmap(process_daily_dvs_200_data, arguments)
    else:
        for input_file, output_file in tqdm(arguments, desc="Processing"):
            process_daily_dvs_200_data(input_file, output_file)


def process_cifar10_dvs_data(file, label_index, label_name, index, dataset_name):
    shape, raw_events = parse_dvs_128(file)

    base_output_folder = f"{util.get_data_folder()}/{dataset_name}/action_events"
    os.makedirs(base_output_folder, exist_ok=True)

    events = []
    for ev in raw_events:
        events.append((ev[0], ev[1], ev[2], ev[3]))
    events.sort(key=lambda ev: ev[2])
    np_events = np.array(events, dtype=event_sa_dtype)
    output_file = f"{base_output_folder}/{label_index}_{label_name}_{file.stem}.npz"
    np.savez_compressed(output_file, events=np_events, shape=shape)
    

def create_cifar10_dvs_data(dataset_name, parallel=False):
    
    class_folder_base = get_orig_dataset_folder_or_error("CIFAR10-DVS/classes")

    arguments = []

    index = 0
    class_folders = sorted(class_folder_base.glob('*'), key=lambda v: v.stem)
    for i, class_folder in enumerate(class_folders):
        files = list(class_folder.glob('*.aedat'))
        for file in files:            
            arguments.append((file, i, class_folder.stem, index, dataset_name))
            index += 1

    if parallel:
        with Pool() as pool:
            pool.starmap(process_cifar10_dvs_data, arguments)
    else:
        for file, label_index, label_name, index, ds_name in tqdm(arguments, desc="Processing"):
            process_cifar10_dvs_data(file, label_index, label_name, index, ds_name)


    execute_split(dataset_name)



def process_sl_animals_file(filename, dataset_name):
    base = Path(f"{util.get_orig_dataset_folder()}/SL-Animals-DVS")
    output_base = f"{util.get_data_folder()}/{dataset_name}/action_events"
    os.makedirs(output_base, exist_ok=True)
    shape, raw_events = parse_dvs_128(filename)
    labels_data = pd.read_csv(f'{base}/{filename.stem}.csv')
    for index, row in labels_data.iterrows():
        label, start, end = row['class'], row['startTime_ev'], row['endTime_ev']
        arr = np.array(raw_events[start:end], dtype=event_sa_dtype)
        
        events = []
        for ev in arr:
            events.append((ev[1], ev[0], ev[2], ev[3]))
        events.sort(key=lambda ev: ev[2])
        np_events = np.array(events, dtype=event_sa_dtype)

        np.savez_compressed(f'{output_base}/{int(label)-1}_{filename.stem}.npz', events=np_events, shape=shape)


USER_RE = re.compile(r"user(\d+)")

def _extract_user_id(filename: str) -> str:
    """
    Extracts the user identifier string from a filename.
    E.g. '0_user24_sunlight.npz' -> 'user24'
    """
    m = USER_RE.search(filename)
    if not m:
        raise ValueError(f"Could not find user id in filename: {filename}")
    return f"user{m.group(1)}"


def grouped_kfold_by_user(
    data_dir_or_files: Union[str, Path, Iterable[str]],
    n_splits: int = 4,
    seed: int = 42,
    export_dir: Union[str, Path] = "./",
    file_exts: Tuple[str, ...] = (".npz",),
) -> Generator[Tuple[List[str], List[str]], None, None]:
    """
    Perform K-fold cross-validation where the folds are created at the *user* level.
    All files belonging to a user are assigned to the same fold.

    Parameters
    ----------
    data_dir_or_files : str | Path | iterable of str
        - If str/Path: directory containing flat-stored files (e.g., '*.npz').
        - If iterable: explicit list of file paths.
    n_splits : int
        Number of folds (e.g., 4).
    seed : int
        RNG seed used to shuffle *users* deterministically.
    file_exts : tuple[str]
        File extensions to include (default: ('.npz',)).

    Yields
    ------
    (train_files, test_files) : tuple[list[str], list[str]]
        Absolute file paths for train/test for each fold, in order 1..n_splits.
    """
    # 1) Collect file paths
    if isinstance(data_dir_or_files, (str, Path)):
        data_dir = Path(data_dir_or_files)
        files = [str(p.resolve()) for p in data_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in file_exts]
    else:
        files = [str(Path(f).resolve()) for f in data_dir_or_files]

    if len(files) == 0:
        raise ValueError("No files found. Check your directory or file list and extensions.")

    # 2) Map user -> list of files
    user_to_files = {}
    for f in files:
        user = _extract_user_id(os.path.basename(f))
        user_to_files.setdefault(user, []).append(f)

    users = np.array(sorted(user_to_files.keys()))  # sorted for stability, then shuffled
    if len(users) < n_splits:
        raise ValueError(f"Need at least {n_splits} distinct users, found {len(users)}.")

    # 3) Shuffle users (deterministic w.r.t. seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    # 4) Chunk shuffled users into n_splits folds (as balanced as possible)
    user_folds = np.array_split(users, n_splits)

    for i, test_users in enumerate(user_folds, start=1):
        test_users = set(test_users.tolist())
        train_users = set(users.tolist()) - test_users

        train_files = []
        for u in train_users:
            train_files.extend(user_to_files[u])

        test_files = []
        for u in test_users:
            test_files.extend(user_to_files[u])

        yield train_files, test_files

def create_sl_animals_dvs_data(dataset_name, parallel=False):
    base = get_orig_dataset_folder_or_error('SL-Animals-DVS')
    files = sorted(base.glob("*.aedat"))

    if parallel:
        with Pool() as pool:
            pool.starmap(process_sl_animals_file, [(file, dataset_name) for file in files])
    else:        
        for file in tqdm(files, desc="Processing"):
            process_sl_animals_file(file, dataset_name)
   
    execute_split(dataset_name)


def create_sl_animals_dvs_k_fold_data(dataset_name, parallel=False):
    base = get_orig_dataset_folder_or_error('SL-Animals-DVS')
    files = sorted(base.glob("*.aedat"))
    
    if parallel:
        with Pool() as pool:
            pool.starmap(process_sl_animals_file, [(file, dataset_name) for file in files])
    else:        
        for file in tqdm(files, desc="Processing"):
            process_sl_animals_file(file, dataset_name)

    output_base = f"{util.get_data_folder()}/{dataset_name}/action_events"
    
    grouped_kfold_by_user(output_base, n_splits=4)
    
    for fold_idx, (train, test) in enumerate(
        grouped_kfold_by_user(output_base, n_splits=4, seed=1337)):
        train_folder = f'{output_base}/train_{fold_idx}'
        test_folder = f'{output_base}/test_{fold_idx}'
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        for f in train:
            shutil.copy(str(f), Path(train_folder) / Path(f).name)
        for f in test:
            shutil.copy(str(f), Path(test_folder) / Path(f).name)


def process_ncaltech101_file(source_file, output_file):
    raw_events = read_mnist_file(source_file, dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]))
    
    shape = (233, 173)
    events = []
    for ev in raw_events:
        events.append((ev['x'], ev['y'], ev['t'], ev['p'] == 1))
    events.sort(key=lambda ev: ev[2])
    np_events = np.array(events, dtype=event_sa_dtype)
    np.savez_compressed(output_file, events=np_events, shape=shape)

   

def create_ncaltech101_data(dataset_name, parallel=False):
    base = get_orig_dataset_folder_or_error("Caltech101")

    index = 0
    class_folders = sorted(base.glob('*'), key=lambda v: v.stem)
    
    arguments = []

    base_output_folder = f"{util.get_data_folder()}/{dataset_name}/action_events"
    os.makedirs(base_output_folder, exist_ok=True)

    
    for i, class_folder in enumerate(class_folders):
        files = list(class_folder.glob('*.bin'))
        for source_file in files:

            output_file = f"{base_output_folder}/{i}_{class_folder.stem}_{source_file.stem}.npz"
            
            arguments.append((source_file, output_file))
            index += 1

    if parallel:
        with Pool() as pool:
            pool.starmap(process_ncaltech101_file, arguments)
    else:
        for file, output in tqdm(arguments, desc="Processing"):
            process_ncaltech101_file(file, output)
    
    execute_split(dataset_name)
    


def create_ncars_data(dataset_name, parallel=False):
    slices = ["train", "test"]
    classes = ["cars", "background"]

    if parallel:
        print(f"Parallel mode not implemented for {dataset_name}")

    orig_base = get_orig_dataset_folder_or_error('NCARS')
    index = 0
    for slice in slices:
        for i, cls in enumerate(classes):
            ncars_ds_folder = f"{orig_base}/n-cars_{slice}/{cls}"
            files = list(Path(ncars_ds_folder).glob('*'))
            base_output_folder = f"{util.get_data_folder()}/{dataset_name}/action_events/{slice}"
            os.makedirs(base_output_folder, exist_ok=True)

            for file in tqdm(files, desc=f'Processing {cls} in {slice} split'):
                data = load_td_data(file)

                events = []
                for ev in data:
                    events.append((ev[1], ev[2], ev[0], ev[3] == 1))
                np_events = np.array(events, dtype=event_sa_dtype)

                output_file = f"{base_output_folder}/{i}_{cls}_{file.stem}_{index}.npz"
                np.savez_compressed(output_file, events=np_events, shape=(120, 100))
                index += 1


def process_asl_dvs_file(file, label, slice, index, dataset_name):
    target_folder_base = Path(f"{util.get_data_folder()}/{dataset_name}/action_events/{slice}")
    os.makedirs(target_folder_base, exist_ok=True)
    
    raw_events = scio.loadmat(file)
    shape = (240, 180)
    
    events = []
    for x, y, t, p in zip(raw_events['x'], raw_events['y'], raw_events['ts'], raw_events['pol']):
        events.append((x[0], shape[1] - y[0] - 1, t[0], p[0]))

    event_arr = np.array(events, dtype=event_sa_dtype)
       
    npz_filename = f'{target_folder_base}/{label}_{file.stem}.npz'
    np.savez_compressed(npz_filename, events=event_arr, shape=shape)



def create_asl_dvs_data(dataset_name, parallel=False):
    base_folder = get_orig_dataset_folder_or_error("ASL-DVS")
    
    slices = ['test', 'train']

    index = 0

    arguments = []
    for slice in slices:
        labels_data = pd.read_csv(f'{base_folder}/ASL-DVS_{slice}.csv')
        for index, row in labels_data.iterrows():
            local_path, class_index = row['events_file_path'], row['class_index']
            path = f'{base_folder}/{local_path}'
            arguments.append((Path(path), class_index, slice, index, dataset_name))
            index += 1
    if parallel:
        with Pool() as pool:
            pool.starmap(process_asl_dvs_file, arguments)
    else:
        for file, label, slice, index, ds_name in tqdm(arguments, desc="Processing"):
            process_asl_dvs_file(file, label, slice, index, ds_name)        


def process_thu_eact_50_chl_file(path, label, slice, index, dataset_name):
    file = Path(path)
    data = np.load(file)
    
    target_folder_base = f"{util.get_data_folder()}/{dataset_name}/action_events/{slice}"
    os.makedirs(target_folder_base, exist_ok=True)
    
    events = []
    for x, y, t, p in data:
        events.append((int(x), int(y), int(t), p == 1))

    event_arr = np.array(events, dtype=event_sa_dtype)
    npz_filename = f'{target_folder_base}/{label}_{file.stem}_{index}.npz'
    shape = (346, 260)
    np.savez_compressed(npz_filename, events=event_arr, shape=shape)
    


def create_thu_eact_50_chl_data(dataset_name, parallel=False):
    base_folder = get_orig_dataset_folder_or_error("THU-EACT-50-CHL")
    
    slices = ['test', 'train']

    index = 0

    arguments = []
    for slice in slices:
        with open(f'{base_folder}/{slice}.txt') as file:
            lines = [line.strip() for line in file]
            for line in lines:
                rel_file, label_str = line.split()
                file_path = f'{base_folder}/{Path(rel_file).name}'
                arguments.append((file_path, int(label_str), slice, index, dataset_name))
                index += 1

    if parallel:
        with Pool() as pool:
            pool.starmap(process_thu_eact_50_chl_file, arguments)
    else:
        for file_path, label, slice, index, dataset_name in tqdm(arguments, desc=f"Processing {dataset_name} file"):
            process_thu_eact_50_chl_file(file_path, label, slice, index, dataset_name)




def create_dvs_gesture_data_from_aedat(filename, split, dataset_name):
    base_folder = get_orig_dataset_folder_or_error("DvsGesture")
    target_folder_base = Path(f"{util.get_data_folder()}/{dataset_name}/action_events/{split}")

    os.makedirs(target_folder_base, exist_ok=True)

    csv_filename = base_folder / f"{Path(filename).stem}_labels.csv"
    labels_data = pd.read_csv(csv_filename)

    shape, events = parse_dvs_ibm(filename)

    actions = []
    for index, row in labels_data.iterrows():
        actions.append((row['class']-1, row['startTime_usec'], row['endTime_usec']))
    
    action_events = extract_action_events(events, actions)

    index = 1
    for label, events in action_events:
        sub_events_list = split_events(events, 500, 250)
        for i, sub_events in enumerate(sub_events_list):
            npz_filename = f'{target_folder_base}/{label}_{Path(filename).stem}_{index}.npz'
            np.savez_compressed(npz_filename, events=sub_events, shape=shape)
            index += 1



def create_dvs_gesture_data(dataset_name, parallel=False):
    base_folder = get_orig_dataset_folder_or_error("DvsGesture")

    arguments = []
    
    splits = ['train', 'test']
    for split in splits:
        filename = f'{base_folder}/trials_to_{split}.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:      
                line = line.strip()
                if len(line) > 0:
                    arguments.append((f'{base_folder}/{line}', split, dataset_name))

    if parallel:
        with Pool() as pool:
            pool.starmap(create_dvs_gesture_data_from_aedat, arguments)
    else:
        for filename, split, ds_name in tqdm(arguments, desc="Processing"):
            create_dvs_gesture_data_from_aedat(filename, split, ds_name)


 

def process_daily_action_dvs_data(file, class_name, index, i, dataset_name):
    base_output_folder = f"{util.get_data_folder()}/{dataset_name}/action_events"
    shape, raw_events = parse_dvs_128(file)
    events = []
    for ev in raw_events:
        events.append((ev[1], shape[1] - ev[0] - 1, ev[2], ev[3]))
    events.sort(key=lambda ev: ev[2])
    np_events = np.array(events, dtype=event_sa_dtype)
    output_file = f"{base_output_folder}/{i}_{class_name}_{file.stem}.npz"
    np.savez_compressed(output_file, events=np_events, shape=shape)


def create_daily_action_dvs_data(dataset_name, parallel=False):
    
    base = get_orig_dataset_folder_or_error("DailyAction-DVS")

    base_output_folder = f"{util.get_data_folder()}/{dataset_name}/action_events"
    os.makedirs(base_output_folder, exist_ok=True)

    arguments = []

    index = 0
    class_folders = sorted(base.glob('*'), key=lambda v: v.stem)
    for i, class_folder in enumerate(class_folders):
        files = list(class_folder.glob('*.aedat'))
        for file in files:
            if "(1)" in file.name: # The downloaded zip file contained duplicates
                continue
            arguments.append((file, class_folder.stem, index, i, dataset_name))
            index += 1

    if parallel:
        with Pool() as pool:
            pool.starmap(process_daily_action_dvs_data, arguments)           
    else:
        for file, class_name, index, i, ds_name in tqdm(arguments, desc="Processing"):
            process_daily_action_dvs_data(file, class_name, index, i, ds_name)

    execute_split(dataset_name)



def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process datasets for Generalized CSTR article 2025")

    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=dataset_names,
        default="daily_dvs_200",
        help=f"Dataset to use. One of: {', '.join(dataset_names)}"
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=["serial", "parallel"],
        default="serial",
        help=""
    )

    return parser.parse_args()


def preprocess_dataset(dataset_name, mode="serial"):
    parallel = mode == "parallel"
    print(f"Pre-processing dataset {dataset_name}")
    if dataset_name == "daily_dvs_200":
        create_daily_dvs_200_data(dataset_name, parallel)
    elif dataset_name == "dvs_gesture":
        create_dvs_gesture_data(dataset_name, parallel)
    elif dataset_name == "dvs_lip":
        create_dvs_lip_data(dataset_name, parallel)
    elif dataset_name == "ncars":
        create_ncars_data(dataset_name, parallel)
    elif dataset_name == "daily_action_dvs":
        create_daily_action_dvs_data(dataset_name, parallel)
    elif dataset_name == "ncaltech101":
        create_ncaltech101_data(dataset_name, parallel)
    elif dataset_name == "sl_animals_dvs":
        create_sl_animals_dvs_data(dataset_name, parallel)
    elif dataset_name == "sl_animals_dvs_k_fold":
        create_sl_animals_dvs_k_fold_data(dataset_name, parallel)
    elif dataset_name == "cifar10_dvs":
        create_cifar10_dvs_data(dataset_name, parallel)
    elif dataset_name == "asl_dvs":
        create_asl_dvs_data(dataset_name, parallel)
    elif dataset_name == "thu_eact_50_chl":
        create_thu_eact_50_chl_data(dataset_name, parallel)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    preprocess_dataset(args.dataset, args.mode)    


