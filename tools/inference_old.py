import onnxruntime as ort 
from PIL import Image, ImageDraw,ImageFont
from torchvision.transforms import ToTensor
import torch
# print(onnx.helper.printable_graph(mm.graph))
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('imgpath',  help='image path')
    parser.add_argument('checkpoint', help='checkpoint file onnx format')
    parser.add_argument('outpath',help='save path')
    parser.add_argument('--name',help='algorithname',default='RT-DETR')
    parser.add_argument('--save_txt', action='store_true', default=False,)
    args = parser.parse_args()
    return args

def resize_bbox(bbox, original_size, target_size):
    """
    Resize a single bounding box according to the change in image size.

    Parameters:
    - bbox: list or array of length 4, format [xmin, ymin, xmax, ymax].
    - original_size: tuple of (height, width) of the original image.
    - target_size: tuple of (height, width) of the target image.

    Returns:
    - resized_bbox: list of length 4 with the resized bounding box.
    # 示例用法
    bbox = [100, 200, 300, 400]
    original_size = (960, 960)
    target_size = (576, 960)
    """
    # 解构原始和目标尺寸
    orig_height, orig_width = original_size
    target_height, target_width = target_size

    # 计算宽度和高度的缩放因子
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height

    # 应用缩放因子
    xmin = bbox[0] * scale_x
    xmax = bbox[2] * scale_x
    ymin = bbox[1] * scale_y
    ymax = bbox[3] * scale_y

    # 返回调整后的边界框
    return [xmin, ymin, xmax, ymax]



def main():
  args = parse_args()
  # 配置文件和预训练权重的路径
  checkpoint_file = args.checkpoint
  # 测试图片路径
  imgpath = args.imgpath # 替换为你的图片路径
  #图片保存路径
  outpath=args.outpath
  im = Image.open(imgpath).convert('RGB')
  im_origin=im
  im = im.resize((960, 960))
  size = torch.tensor([[960,960]])
  im_data = ToTensor()(im)[None]
  print(im_data.shape)

  

  sess = ort.InferenceSession(checkpoint_file)
  output = sess.run(
      # output_names=['labels', 'boxes', 'scores'],
      output_names=None,
      input_feed={'images': im_data.data.numpy(), "orig_target_sizes":  size.data.numpy()}
  )
  labels, bboxes, scores = output
  print(output)
  
  if args.save_txt:
     fmt = '[' + ','.join(['%f']*bboxes[0].shape[1]) + '],'
     np.savetxt('bboxes.txt', bboxes[0], fmt=fmt)
     np.savetxt('scores.txt', scores, delimiter=',',fmt='%.4f')

  draw = ImageDraw.Draw(im_origin)
  font1 = ImageFont.truetype("Arial.ttf", 16)
  font2= ImageFont.truetype("Arial.ttf", 25)
  thrh = 0.5

  algorith_name=args.name
  bbox_color='#a566ff'
  font_color='#a566ff'
  # 获取文本大小
  left, top, right, bottom= draw.textbbox((0,0),algorith_name, font=font2)
  text_width, text_height = right-left,bottom-top
  #绘制文本呈现算法名称
  draw.text(((im.width - text_width) / 2,text_height/2), algorith_name, fill='red',font=font2)

   
  for i in range(im_data.shape[0]):
      
      scr =scores[i]
      lab = labels[i][scr > thrh]
      box = bboxes[i][scr > thrh]

      print(i, sum(scr > thrh))
      n=0
      for b in box:
        if scr[n]<thrh:
             continue
         
        x, y, x_max, y_max = b
    
        bbox=resize_bbox([x, y, x_max, y_max], (960,960), (576,960))
        x, y, x_max, y_max=bbox
       
        # 绘制矩形框
        draw.rectangle(bbox, outline=f'{bbox_color}', width=2)          
         
        text = f"passenger:{round(float(scr[n]) * 100, 1)} "
        # 使用 ImageFont 对象的 getsize() 方法获取文本的大小
        left, top, right, bottom= draw.textbbox((0,0),text, font=font1)
        text_width, text_height = right-left,bottom-top
        #print("text_width:%d,text_height:%d"%(text_width, text_height))
        text_x = x+3
        text_y = y+3 
        # 绘制文字背景（黑色边框效果）
        draw.rectangle([text_x - 1, text_y - 1, text_x + text_width + 1, text_y + text_height + 1], fill='black')
          # 绘制白色文字
        draw.text((text_x, text_y-2), text, fill=f'{font_color}',font=font1)  # 如果使用自定义
        n=n+1

  im_origin.save(args.name+'.jpg')
     
if __name__ == '__main__':
    main()

'''
python tools/inference.py  /public/home/houjie/video_structure/datasets/pedestrainDatasets/BusPassengers/visualization/3.PNG /public/home/houjie/video_structure/models_trained/object_detection/rtdetr_dualencode_level4_encodestage2_residual_AttentionGated_r50vd_6x_buspassenger_random/dualattention_stage2_EAR_AttentionGated_best_w960_h960.onnx  ./ --name DAEAR-DETR

'''    
 

 

 