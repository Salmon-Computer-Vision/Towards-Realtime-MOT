#!/usr/bin/env python3
import torch
import torch.onnx
import argparse
from models import *
from utils.parse_config import parse_model_cfg

BATCH_SIZE=1

def main(opt):
  cfg_dict = parse_model_cfg(opt.cfg)
  opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]
  print(opt.img_size)

  model = model = Darknet(opt.cfg)
  model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)
  if torch.cuda.is_available():
      model.cuda().eval()
  else:
      model.eval()

  dummy_input = torch.randn(BATCH_SIZE, 3, opt.img_size[0], opt.img_size[1], device=torch.device('cuda'))
  torch.onnx.export(model, dummy_input, "darknet_mot.onnx", verbose=False, opset_version=11)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert MOT Darknet model weights to ONNX.")
  parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
  parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')

  opt = parser.parse_args()
  main(opt)
