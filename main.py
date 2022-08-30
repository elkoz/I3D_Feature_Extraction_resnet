from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
from extract_features import run
from utils.resnet import i3_res50
import os
import pickle


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
    i3d_suffix="_i3d.npy"
):
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = outputpath + "/temp/"
    rootdir = Path(datasetpath)
    videos = [str(f) for f in rootdir.glob("**/*.mp4")]
    # setup the model
    i3d = i3_res50(400, pretrainedpath)
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode
    for i, video in enumerate(videos):
        videoname = video.split("/")[-1].split(".")[0]
        startime = time.time()
        print("Generating for {0} ({1} / {2})".format(video, i + 1, len(videos)))
        Path(temppath).mkdir(parents=True, exist_ok=True)
        ffmpeg.input(video).output(
            "{}%d.jpg".format(temppath), start_number=0
        ).global_args("-loglevel", "quiet").run()
        print("Preprocessing done..")
        if detection_folder is not None and detection_suffix is not None:
            detection_file = os.path.join(
                detection_folder, videoname + detection_suffix
            )
            with open(detection_file, "rb") as f:
                detection = pickle.load(f)
        else:
            detection = {"": None}
        features = {}
        min_frames_dict = {}
        max_frames_dict = {}
        for key, value in detection.items():
            if value is None or len(value) >= min_frames:
                print("KEY=", key)
                features[key], min_frames_dict[key], max_frames_dict[key] = run(
                    i3d,
                    frequency,
                    temppath,
                    batch_size,
                    sample_mode,
                    value,
                    pad,
                    video_w,
                    video_h,
                )
        if save_metadata:
            features["min_frames"] = min_frames_dict
            features["max_frames"] = max_frames_dict
        np.save(outputpath + "/" + videoname + i3d_suffix, features)
        shutil.rmtree(temppath)
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
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="center_crop",
        help="Either 'oversample' or 'center_crop'",
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
        default="i3d_suffix",
        help="The suffix to add to the output files",
    )
    args = parser.parse_args()
    generate(
        args.datasetpath,
        str(args.outputpath),
        args.pretrainedpath,
        args.frequency,
        args.batch_size,
        args.sample_mode,
        args.video_w,
        args.video_h,
        args.tracking_folder,
        args.tracking_suffix,
        args.min_frames,
        args.pad,
        args.save_metadata,
        args.i3d_suffix
    )
