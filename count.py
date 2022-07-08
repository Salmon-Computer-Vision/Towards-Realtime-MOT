"""Demo file for running the JDE tracker on custom video sequences for pedestrian tracking.

This file is the entry point to running the tracker on custom video sequences. It loads images from the provided video sequence, uses the JDE tracker for inference and outputs the video with bounding boxes indicating pedestrians. The bounding boxes also have associated ids (shown in different colours) to keep track of the movement of each individual. 

Examples:
        $ python demo.py --input-video path/to/your/input/video --weights path/to/model/weights --output-root path/to/output/root


Attributes:
    input-video (str): Path to the input video for tracking.
    output-root (str): Output root path. default='results'
    weights (str): Path from which to load the model weights. default='weights/latest.pt'
    cfg (str): Path to the cfg file describing the model. default='cfg/yolov3.cfg'
    iou-thres (float): IOU threshold for object to be classified as detected. default=0.5
    conf-thres (float): Confidence threshold for detection to be classified as object. default=0.5
    nms-thres (float): IOU threshold for performing non-max supression. default=0.4
    min-box-area (float): Filter out boxes smaller than this area from detections. default=200
    track-buffer (int): Size of the tracking buffer. default=30
    output-format (str): Expected output format, can be video, or text. default='video'
    

Todo:
    * Add compatibility for non-GPU machines (would run slow)
    * More documentation
"""

import logging
import argparse
import os
import math

from utils.utils import mkdir_if_missing
from utils.utils import osp
import torch
import cv2

#from utils.utils import *
from utils.log import logger
from utils.timer import Timer
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
#from track import eval_seq

from utils import visualization as vis
import ffmpeg

import pandas as pd

logger.setLevel(logging.INFO)

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, left_direct=False):
    from tracker.multitracker import JDETracker
    '''
       Processes the video sequence given and provides the output of tracking result (write the results in video file)

       It uses JDE model for getting information about the online targets present.

       Parameters
       ----------
       opt : Namespace
             Contains information passed as commandline arguments.

       dataloader : LoadVideo
                    Instance of LoadVideo class used for fetching the image sequence and associated data.

       data_type : String
                   Type of dataset corresponding(similar) to the given video.

       result_filename : String
                         The name(path) of the file for storing results.

       save_dir : String
                  Path to the folder for storing the frames containing bounding box information (Result frames).

       show_image : bool
                    Option for shhowing individial frames during run-time.

       frame_rate : int
                    Frame-rate of the given video.

       Returns
       -------
       (Returns are not significant here)
       frame_id : int
                  Sequence number of the last sequence
       '''

    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0

    COL_COUNTABLE_ID = 'Countable ID'
    COL_FRAME_NUM = 'Frame Num'
    COL_DIRECTION = 'Direction'
    VAL_LEFT = 'Left'
    VAL_RIGHT = 'Right'

    num_ids = {}
    # IDs that are countable
    counted_ids = pd.DataFrame(columns=[COL_FRAME_NUM, COL_COUNTABLE_ID, COL_DIRECTION]) 

    count_thresh = 0.40 # Box must be past this percentage of the screen in the direction specified

    right_dir_thresh = opt.img_size[0] * count_thresh
    left_dir_thresh = opt.img_size[0] * (1.0 - count_thresh)

    hist_thresh = math.ceil(frame_rate / 4) # A quarter of a second
    horiz_thresh = 1.3 # Forces rectangular bounding boxes (Rect ratio)
    logger.info(f'Count threshold: {count_thresh}')
    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if torch.cuda.is_available():
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            horizontal = tlwh[2] / tlwh[3] > horiz_thresh
            if tlwh[2] * tlwh[3] > opt.min_box_area and horizontal:
                if not tid in num_ids:
                    num_ids[tid] = 1
                else:
                    num_ids[tid] += 1

                    past_left = tlwh[0] + tlwh[2] < left_dir_thresh
                    past_right = tlwh[0] > right_dir_thresh
                    if past_left or past_right:
                        det_id = counted_ids[counted_ids[COL_COUNTABLE_ID] == tid]
                        if not det_id.empty:
                            last_row = det_id.iloc[-1,:]
                            # Check if last entry was a left or right direction
                            # This is done to prevent constant entries
                            not_last_left = last_row[COL_DIRECTION] != VAL_LEFT
                            not_last_right = last_row[COL_DIRECTION] != VAL_RIGHT
                        else:
                            not_last_left = True
                            not_last_right = True

                        def append_id(direction):
                            temp_df = pd.DataFrame([[frame_id, tid, direction]], 
                                    columns=[COL_FRAME_NUM, COL_COUNTABLE_ID, COL_DIRECTION])
                            return pd.concat([counted_ids, temp_df], ignore_index=True)

                        countable = num_ids[tid] > hist_thresh
                        if past_left and not_last_left and countable:
                            counted_ids = append_id(VAL_LEFT)
                        elif past_right and not_last_right and countable:
                            counted_ids = append_id(VAL_RIGHT)

                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))

        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)

    return frame_id, timer.average_time, timer.calls, counted_ids

def track(opt):
    result_root = opt.output_root if opt.output_root!='' else '.'
    mkdir_if_missing(result_root)

    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # run tracking
    timer = Timer()
    accs = []
    n_frame = 0

    logger.info('Starting tracking...')
    if opt.input_format == 'video':
      dataloader = datasets.LoadVideo(opt.input, opt.img_size)
      result_filename = os.path.join(result_root, 'results.txt')
      frame_rate = dataloader.frame_rate 

    frame_dir = None if opt.output_format=='text' else osp.join(result_root, 'frame')
    #try:
    _,_,_,counted_ids = eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=opt.show_image, frame_rate=frame_rate)
    #except Exception as e:
    #    logger.info(e)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')

        (
          ffmpeg.input(f"{osp.join(result_root, 'frame')}/%05d.jpg", f='image2', framerate=frame_rate)
          .output(output_video_path, vcodec='libx264')
          .run()
        )

        #cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v libx264 {}'.format(osp.join(result_root, 'frame'), output_video_path)

    name = osp.splitext(osp.basename(opt.input))[0]
    outpath = osp.join(result_root, f'{name}.csv')
    counted_ids.to_csv(outpath)
    logger.info(f"saved counts to {outpath}")
    logger.info(counted_ids)
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Records long rectangular countable object IDs in a csv. Whether they moved towards the left or right is also added.")
  parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
  parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
  parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
  parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
  parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
  parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
  parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
  parser.add_argument('--input-format', type=str, default='video', choices=['video', 'images'], help='Expected input format (Default: Video)')
  parser.add_argument('--input', type=str, help='path to the input video or image directory depending on input format.')
  parser.add_argument('--output-format', type=str, default='video', choices=['video', 'text'], help='Expected output format (Default: Video)')
  parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
  parser.add_argument('--show-image', action='store_true', help='Show image frames as they are being processed')
  #parser.add_argument('-l', '--left', action='store_true', help='Count in the left direction')
  parser.add_argument('--count-thresh', default=0.4, help='Ratio of how far across the screen the object must be before being counted. Default 0.4, meaning 40%% of the screen must be crossed before being countable.')

  subp = parser.add_subparsers()
  detect_p = subp.add_parser('detect', help='Adds tensorflow detection to classify categories')
  detect_p.add_argument('model_dir', help='Path to the model folder. Must have "pipeline.json" in the directory')
  detect_p.add_argument('ckpt_path', help='Path to the checkpoint file. Eg. /path/to/ckpt-0')
  detect_p.add_argument('labels_path', help='Path to the labels text file. Eg. /path/to/labels.pbtxt')

  opt = parser.parse_args()
  print(opt, end='\n\n')

  track(opt)

