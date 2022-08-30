import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable
from collections import defaultdict


def load_frame(frame_file, bbox, video_w, video_h):
    data = Image.open(frame_file)
    if bbox is not None:
        if video_w is not None and video_h is not None:
            data = data.resize((video_w, video_h), Image.ANTIALIAS)
        data = data.crop(bbox)
    data = data.resize((340, 256), Image.ANTIALIAS)
    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1
    assert data.max() <= 1.0
    assert data.min() >= -1.0
    return data


def load_rgb_batch(
    frames_dir, rgb_files, frame_indices, detection, video_w, video_h, start
):
    if detection is None:
        detection = defaultdict(lambda: None)
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(
                os.path.join(frames_dir, rgb_files[frame_indices[i][j] - start]),
                detection[frame_indices[i][j]],
                video_w,
                video_h,
            )
    return batch_data


def oversample_data(data):
    data_flip = np.array(data[:, :, :, ::-1, :])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_f_1,
        data_f_2,
        data_f_3,
        data_f_4,
        data_f_5,
    ]


def run(
    i3d,
    frequency,
    frames_dir,
    batch_size,
    sample_mode,
    detection=None,
    pad=False,
    video_w=1024,
    video_h=576,
):
    assert sample_mode in ["oversample", "center_crop"]
    chunk_size = 8

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)  # b,c,t,h,w  # 40x3x16x224x224
        with torch.no_grad():
            b_data = Variable(b_data.cuda()).float()
            inp = {"frames": b_data}
            features = i3d(inp)
        return features.cpu().numpy()

    rgb_files = natsorted(
        [
            i
            for i in os.listdir(frames_dir)
            if detection is None or int(i.split(".")[0]) in detection
        ]
    )
    frame_cnt = len(rgb_files)
    start = int(rgb_files[0].split(".")[0])
    end = int(rgb_files[-1].split(".")[0])
    assert frame_cnt > chunk_size
    clipped_length = frame_cnt - chunk_size
    clipped_length = (
        clipped_length // frequency
    ) * frequency  # The start of last chunk
    frame_indices = []  # Frames to chunks
    for i in range(clipped_length // frequency + 1):
        frame_indices.append(
            [j + start for j in range(i * frequency, i * frequency + chunk_size)]
        )
    frame_indices = np.array(frame_indices)
    chunk_num = frame_indices.shape[0]
    batch_num = int(np.ceil(chunk_num / batch_size))  # Chunks to batches
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)

    if sample_mode == "oversample":
        full_features = [[] for i in range(10)]
    else:
        full_features = [[]]

    for batch_id in range(batch_num):
        batch_data = load_rgb_batch(
            frames_dir,
            rgb_files,
            frame_indices[batch_id],
            detection,
            video_w,
            video_h,
            start,
        )
        if sample_mode == "oversample":
            batch_data_ten_crop = oversample_data(batch_data)
            for i in range(10):
                assert batch_data_ten_crop[i].shape[-2] == 224
                assert batch_data_ten_crop[i].shape[-3] == 224
                temp = forward_batch(batch_data_ten_crop[i])
                full_features[i].append(temp)

        elif sample_mode == "center_crop":
            batch_data = batch_data[:, :, 16:240, 58:282, :]
            assert batch_data.shape[-2] == 224
            assert batch_data.shape[-3] == 224
            temp = forward_batch(batch_data)
            full_features[0].append(temp)

    full_features = [np.concatenate(i, axis=0) for i in full_features]
    full_features = [np.expand_dims(i, axis=0) for i in full_features]
    full_features = np.concatenate(full_features, axis=0)
    full_features = full_features[:, :, :, 0, 0, 0]
    full_features = np.array(full_features).transpose([1, 0, 2]).mean(1).squeeze()
    if pad:
        shape = full_features.shape[0]
        left_pad = (frame_cnt - shape) // 2
        right_pad = (frame_cnt - shape) - left_pad
        full_features = np.pad(
            full_features, ((left_pad, right_pad), (0, 0)), mode="edge"
        )
    print("full_features.shape={}".format(full_features.shape))
    return full_features, start, end
