import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import numpy as np 

from src.core import YAMLConfig

import torch
import torch.nn as nn 
import onnx



def readONNX(args,):
  # 加载ONNX模型
  model = onnx.load(args.onnxPath) 
  # 打印模型的输入和输出
  #print(model.graph.input)
  #print(model.graph.output)
  with open("file.txt", "w") as file:
    for node in model.graph.node:
  
       file.write(str(node)+'\r')
       file.write('-------------------------------\r')
       #print(node)

       if node.op_type=='TopK' and node.name=='/model/decoder/TopK':
          print(node)
          model.graph.node.remove(node)
  onnx.checker.check_model(model)
  onnx.save(model, 'DADR_DETR1.onnx')

   
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnxPath', '-p', type=str, )
    args = parser.parse_args()

    readONNX(args)