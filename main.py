from pathlib import Path
import shutil
import argparse
import numpy as np
import time
from extract_features import run
from utils.resnet import i3_res50
import os
import pickle
from pims import PyAVReaderIndexed
from tqdm import tqdm
import cv2


def generate(
    datasetpath,
    outputpath,
    pretrainedpath,
    frequency,
    batch_size,
    sample_mode,
    video_w,
    video_h,
    detection_folder=None,
    detection_suffix=None,
    min_frames=0,
    pad=False,
    save_metadata=False,
    i3d_suffix="_i3d.npy",
    gpu=0,
    background_extraction=False,
    expand_bboxes=True
):
    if outputpath is None:
        outputpath = datasetpath
    if detection_folder is None:
        detection_folder = datasetpath
    device = f'cuda:{gpu}'
    print('device', device)
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = outputpath + "/temp/"
    rootdir = Path(datasetpath)
    videos = [str(f) for f in rootdir.glob("**/*.mp4")]
    if detection_folder is not None and detection_suffix is not None:
        videos = [x for x in videos if os.path.exists(os.path.join(detection_folder, f'{os.path.basename(x).split(".")[0]}{detection_suffix}'))]
    # setup the model
    i3d = i3_res50(400, pretrainedpath)
    i3d.to(device)
    i3d.train(False)  # Set model to evaluate mode
    for i, video in enumerate(videos):
        videoname = video.split("/")[-1].split(".")[0]
        startime = time.time()
        print("Generating for {0} ({1} / {2})".format(video, i + 1, len(videos)))
        stream = PyAVReaderIndexed(video)
        lazy_imread = stream.get_frame
        length = len(stream)
        print(f'LENGTH {len(stream)}')
        print("Preprocessing done..")
        mean_frame = None
        if background_extraction:
            frames = []
            print('Computing the mean frame...')
            for f_i in tqdm(range(0, length, 200)):
                frames.append(np.array(lazy_imread(f_i), dtype=np.float))
            frames = np.stack(frames, 0)
            mean_frame = np.median(frames, axis=0)
            mean_frame = cv2.cvtColor(mean_frame.astype(np.float32), cv2.COLOR_BGR2GRAY)
        if detection_folder is not None and detection_suffix is not None:
            detection_file = os.path.join(
                detection_folder, videoname + detection_suffix
            )
            with open(detection_file, "rb") as f:
                detection = pickle.load(f)
        else:
            detection = {"ind0": None}
        features = {}
        min_frames_dict = {}
        max_frames_dict = {}
        for key, value in detection.items():
            if value is None or len(value) >= min_frames:
                if value is None:
                    clip_len = length
                else:
                    clip_len = len(value)
                try:
                    features[key], min_frames_dict[key], max_frames_dict[key] = run(
                        i3d=i3d,
                        frequency=frequency,
                        batch_size=batch_size,
                        sample_mode=sample_mode,
                        detection=value,
                        pad=pad,
                        video_w=video_w,
                        video_h=video_h,
                        device=device,
                        lazy_imread=lazy_imread,
                        frame_cnt=clip_len,
                        mean_frame=mean_frame,
                        expand_bboxes=(sample_mode == "expand_bboxes"),
                    )
                except Exception as e:
                    print(e)
        if save_metadata:
            features["min_frames"] = min_frames_dict
            features["max_frames"] = max_frames_dict
        np.save(outputpath + "/" + videoname + i3d_suffix, features)
        print("done in {0}.".format(time.time() - startime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasetpath",
        type=str,
        default="samplevideos/",
        help="The path to a folder containing the videos",
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        default="output",
        help="The path to a folder where the output will be saved (in .npy format)",
    )
    parser.add_argument(
        "--pretrainedpath",
        type=str,
        default="pretrained/i3d_r50_kinetics.pth",
        help="The path to a .pth checkpoint file",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=1,
        help="The distance between the starts of neighboring input chunks",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="expand_bboxes",
        help="Either 'oversample', 'expand_bboxes' or 'center_crop'",
    )
    parser.add_argument(
        "--tracking_folder",
        type=str,
        required=False,
        help="The path to the folder containing tracking files",
    )
    parser.add_argument(
        "--tracking_suffix",
        type=str,
        required=False,
        help="The suffix of the tracking files (for some_video.mp4 the corresponding tracking file should be named some_video{suffix},"
        "e.g. some_video_detection.pickle, if tracking_suffix is _detection.pickle), see github README for more information"
        "on the format",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=9,
        help="Tracklets shorter than this number of frames will be omitted",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        help="If True, the output features will be padded with the edge values to keep the length intact",
    )
    parser.add_argument(
        "--video_w",
        type=int,
        required=False,
        help="The video width (it will be resized to this value before cropping)",
    )
    parser.add_argument(
        "--video_h",
        type=int,
        required=False,
        help="The video height (it will be resized to this value before cropping)",
    )
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="If true, save a dictionary of min and max frames",
    )
    parser.add_argument(
        "--i3d_suffix",
        type=str,
        default="_i3d.npy",
        help="The suffix to add to the output files",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="The index of the gpu to use",
    )
    parser.add_argument(
        "--subtract_background",
        action="store_true",
        help="If true, the median frame of the video is set to gray before the feature extraction",
    )
    args = parser.parse_args()
    generate(
        datasetpath=args.datasetpath,
        outputpath=str(args.outputpath),
        pretrainedpath=args.pretrainedpath,
        frequency=args.frequency,
        batch_size=args.batch_size,
        sample_mode=args.sample_mode,
        video_w=args.video_w,
        video_h=args.video_h,
        detection_folder=args.tracking_folder,
        detection_suffix=args.tracking_suffix,
        min_frames=args.min_frames,
        pad=args.pad,
        save_metadata=args.save_metadata,
        i3d_suffix=args.i3d_suffix,
        gpu=args.gpu,
        background_extraction=args.subtract_background,
    )
