import os
import copy
import random
import numpy as np
import random
import torch

def truncate_feats_fixed_stride(
    data_dict,
    max_seq_len,
    stride,
    has_action=True,
    no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item with a fixed stride.

    Parameters:
        data_dict = {
            'video_id': str,
            'feats': Tensor C x T,
            'segments': Tensor N x 2 (in feature grid),
            'labels': Tensor N,
            'fps': float,
            'feat_stride': int,
            'feat_num_frames': int
        }
        max_seq_len: Maximum sequence length after truncation.
        stride: Fixed stride (frame interval) for truncation.
        has_action: Whether to ensure at least one action segment is preserved.
        no_trunc: Whether to avoid truncating any action segments.
    """
    # Get the meta info
    feat_len = data_dict['feats'].shape[1]
    num_segs = data_dict['segments'].shape[0]

    # Check if the feature length is less than or equal to max_seq_len
    if feat_len <= max_seq_len:
        return data_dict  # No need to truncate

    # Deep copy the data dictionary
    data_dict = copy.deepcopy(data_dict)

    # Calculate the number of strides needed
    num_strides = feat_len // stride
    # Adjust the number of strides to match max_seq_len
    adjusted_strides = min(num_strides, max_seq_len // stride)

    # Initialize new segments and labels lists
    new_segments = []
    new_labels = []

    # Iterate over each stride
    for i in range(adjusted_strides):
        start = i * stride
        end = start + stride

        # Update segments and labels based on the current stride
        left = start
        right = min(feat_len, end)

        # Filter segments that intersect with the current stride
        segs_in_range = (data_dict['segments'][:, 0] <= right) & (data_dict['segments'][:, 1] >= left)
        filtered_segments = data_dict['segments'][segs_in_range]
        filtered_labels = data_dict['labels'][segs_in_range]

        # Shift the segments to the new start point
        shifted_segments = filtered_segments.clone()
        shifted_segments[:, 0] -= left
        shifted_segments[:, 1] -= left

        # Append the updated segments and labels
        new_segments.append(shifted_segments)
        new_labels.append(filtered_labels)

    # Concatenate the updated segments and labels
    data_dict['segments'] = torch.cat(new_segments, dim=0)
    data_dict['labels'] = torch.cat(new_labels, dim=0)

    # Truncate the features tensor
    data_dict['feats'] = data_dict['feats'][:, :max_seq_len]

    return data_dict


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def truncate_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = data_dict['feats'].shape[1]
    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0], data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1], data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict
