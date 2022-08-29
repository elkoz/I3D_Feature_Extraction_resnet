# I3D Features for Action Segmentation
This repo contains code to extract I3D features with resnet50 backbone given a folder of videos and a folder of tracking 
files.

## Credits
This code is a version of [this repository](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet)
adapted for extracting frame-wise features and using tracking files.

## Overview
For each of your videos, the following will happen.
1. If you set the `tracking_folder` and `tracking_suffix` options, the corresponding tracking file will be opened. 
If the video is named `some_video.mp4`, the tracking file should be at `tracking_folder/some_video{tracking_suffix}`. 
For instance, if `tracking_suffix=_detections.pickle`, it would be `tracking_folder/some_video_detections.pickle`. 
The tracking file should be a pickled nested dictionary where first-level keys are individual ids, second-level keys
are frame indices (without any frames missing between start and end!) and values are bounding box arrays in the 
\[left, upper, right, lower\] format. 
2. For each individual from the tracking file, the input video will be cropped in spatial and temporal dimensions and passed
to a pre-trained model in 8-frame chunks with the frequency you set in the options (each chunk maps to one frame feature).
If you don't provide tracking information, the video will not be cropped. Before the cropping, we will resize the video
to `(video_w, video_h)` if you set those options (you need to set both).
3. The output will saved at `outputpath/some_video.npy` as a dictionary where keys are individual ids (`""` if there's 
no tracking) and values are `numpy` arrays of shape `(N, 2048)`. If you use the `--pad` option and set frequency to 1, 
the `N` dimension will be the original number of frames.

## Usage
### Setup
Run this in your terminal to install.
```bash
git clone https://github.com/elkoz/I3D_Feature_Extraction_resnet
cd I3D_Feature_Extraction_resnet
conda env create -f environment.yaml
conda activate i3d
wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.pkl -P pretrained/
python -m utils.convert_weights pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_kinetics.pth
```

### Parameters
<pre>
--datasetpath:       folder of input videos (contains videos or subdirectories of videos)
--outputpath:        folder of extracted features
--frequency:         how many frames between adjacent snippet
--batch_size:        batch size for snippets
--tracking_folder:   path to the folder containing tracking files
--tracking_suffix:   suffix of the tracking files
--min_frames:        tracklets shorter than this number of frames will be omitted
--pad                if true, the output features will be padded with the edge values to keep the length intact
--video_w:           video width (it will be resized to this value before cropping to the bounding boxes)
--video_h:           video height (it will be resized to this value before cropping to the bounding boxes)
</pre>

### Run
```bash
python main.py --datasetpath=samplevideos/ --outputpath=output --pad --tracking_folder=tracking_folder --tracking_suffix=tracking_suffix
```
