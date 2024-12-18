'''
by lyuwenyu
'''
import time 
import json
import datetime
import time
import torch 
import os
import os.path as osp
from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from thop import profile
from torchvision import transforms
from PIL import Image
import onnxruntime as ort 
import numpy as np

class DetSolver(BaseSolver):
    transform = transforms.Compose([
    transforms.Resize((960, 960)),  # 或其他尺寸，根据需要调整
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    
    

      



    def _format_size(self,x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
      """Formats an integer for printing in a table or model representation.

      Expresses the number in terms of 'kilo', 'mega', etc., using
      'K', 'M', etc. as a suffix.

      Args:
          x (int): The integer to format.
          sig_figs (int): The number of significant figures to keep.
              Defaults to 3.
          hide_zero (bool): If True, x=0 is replaced with an empty string
              instead of '0'. Defaults to False.

      Returns:
          str: The formatted string.
      """
      if hide_zero and x == 0:
          return ''

      def fmt(x: float) -> str:
          # use fixed point to avoid scientific notation
          return f'{{:.{sig_figs}f}}'.format(x).rstrip('0').rstrip('.')

      if abs(x) > 1e14:
          return fmt(x / 1e15) + 'P'
      if abs(x) > 1e11:
          return fmt(x / 1e12) + 'T'
      if abs(x) > 1e8:
          return fmt(x / 1e9) + 'G'
      if abs(x) > 1e5:
          return fmt(x / 1e6) + 'M'
      if abs(x) > 1e2:
          return fmt(x / 1e3) + 'K'
      return str(x)
    def test_fps(self,model, device, num_frames=100,batch_size=1,width=640,height=640):
      # 假设输入图像大小为 640x480，这里创建一个随机张量模拟输入图像
      # 实际应用中应根据模型实际输入大小调整
      print("batch_size:%d,width=%d,height=%d"%(batch_size,width,height))
      dummy_input = torch.randn(batch_size, 3, width, height).to(device)
      # 确保模型在评估模式下
      model.eval()
      
      # 预热GPU，避免启动CUDA时的延迟影响性能测量
      for _ in range(10):
          _ = model(dummy_input)
      
      # 开始测试FPS
      start_time = time.time()
      for _ in range(num_frames):
          with torch.no_grad():
              _ = model(dummy_input)
      end_time = time.time()
      
      # 计算平均处理时间和FPS
      total_time = end_time - start_time
      avg_time_per_frame = total_time / num_frames
      fps = 1 / avg_time_per_frame
      
      print(f"Average time per frame: {avg_time_per_frame:.3f} seconds")
      print(f"Estimated FPS: {fps:.2f}")

    # 动态冻结层函数
    def freeze_layers_dynamically(model, epoch, freeze_schedule):
          """根据当前 epoch 和冻结计划调整 requires_grad"""
          for layer_name, freeze_epoch in freeze_schedule.items():
              if epoch >= freeze_epoch:
                  # 解冻当前层
                  for name, param in getattr(model, layer_name).named_parameters():
                      param.requires_grad = True
              else:
                  # 冻结当前层
                  for name, param in getattr(model, layer_name).named_parameters():
                      param.requires_grad = False
    def fit(self, ):
        print("Start training")
        self.train()
       
        args = self.cfg
        
        #如果加载预训练模型进行训练，则使用冻结骨干网部分层的技术
        if self.cfg.tuning:
          # 冻结conv1, layer1, layer2较低层参数，训练高层参数
          initial_freeze = self.get_freeze_list(self.model, ['conv1', 'layer1', 'layer2'])
          self.freeze_layers(self.model, initial_freeze) 
       
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        #print(self.model)
          

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            #如果加载预训练模型进行训练，则使用冻结骨干网部分层的技术
            if self.cfg.tuning:
                #epoch=1/3*epoches的时候解冻layer2
                if (epoch+1)==args.epoches//3:
                  freezeLayers = self.get_freeze_list(self.model, ['conv1', 'layer1'])
                  self.freeze_layers(self.model, freezeLayers)

                #epoch=1/2*epoches的时候解冻layer1
                if (epoch+1)==args.epoches//2:
                  freezeLayers = self.get_freeze_list(self.model, ['conv1'])
                  self.freeze_layers(self.model, freezeLayers)

                #epoch=epoches-2的时候解冻所有层
                if (epoch+1)==args.epoches-2:
                  freezeLayers = self.get_freeze_list(self.model, [])
                  self.freeze_layers(self.model, freezeLayers)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # TODO 
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.test_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.test_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
    
    def get_paramnum_floaps(self,batch_size,width,height):
        self.eval()
        base_ds = get_coco_api_from_dataset(self.test_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        #module=self.cfg.model
        inputs = torch.randn(batch_size, 3, width, height).to(self.device) 
        flops, params = profile(module, (inputs,))
        print('flops: ', self._format_size(flops), 'params: ', self._format_size(params))

    
    def get_fps(self,num_frames,batch_size,width,height):
        self.eval()
        base_ds = get_coco_api_from_dataset(self.test_dataloader.dataset)
        module = self.ema.module if self.ema else self.model

        self.test_fps(module,self.device,num_frames,batch_size,width,height)
    
    def jitTrace(self,num_frames,batch_size,width,height):
        self.eval()
        base_ds = get_coco_api_from_dataset(self.test_dataloader.dataset)
        module = self.ema.module if self.ema else self.model
        inputs = torch.randn(batch_size, 3, width, height).to(self.device) 
        traced_script_module=torch.jit.trace(module,inputs,strict=False)
        traced_script_module.save(self.output_dir / "DADR_DETR.pt")


 
 
 

    def get_feature_maps(self,imagePath):
      self.setup()
      #base_ds = get_coco_api_from_dataset(self.train_dataloader.dataset)
      module = self.ema.module if self.ema else self.model
      # 存储特征图的字典
      activations = dict()

      # 定义钩子函数
      def get_activation(name):
          def hook(model, input, output):
              activations[name] = output.detach()
          return hook
      def get_activation_input(name):
          def hook(model, input, output):
              activations[name] = input[0].detach()
          return hook
       
      # 注册钩子,模型结构可以使用Netron打开一个pt文件获得
      # 输入图片尺寸为960*960
      module.backbone.conv1.register_forward_hook(get_activation('backbone_conv1'))
      #backbone stage1特征图输出,torch.Size([1, 256, 240, 240])
      module.backbone.res_layers[0].blocks[2].branch2c.conv.register_forward_hook(get_activation('backbone_stage1'))
      #backbone stage2特征图输出,torch.Size([1, 512, 120, 120])
      module.backbone.res_layers[1].blocks[3].branch2c.conv.register_forward_hook(get_activation('backbone_stage2'))
      #backbone stage3特征图输出,torch.Size([1, 1024, 60, 60])
      module.backbone.res_layers[2].blocks[5].branch2c.conv.register_forward_hook(get_activation('backbone_stage3'))
      #backbone stage4特征图输出,torch.Size([1, 2048, 30, 30])
      module.backbone.res_layers[3].blocks[2].branch2c.conv.register_forward_hook(get_activation('backbone_stage4'))


      #backbone stage1 inputproj特征图输出,torch.Size([1, 256, 240, 240])
      module.encoder.input_proj[0][1].register_forward_hook(get_activation('backbone_stage1_inputproj'))
      #backbone stage2 inputproj特征图输出,torch.Size([1, 256, 120, 120])
      module.encoder.input_proj[1][1].register_forward_hook(get_activation('backbone_stage2_inputproj'))
      #backbone stage3 inputproj特征图输出,torch.Size([1, 256, 60, 60])
      module.encoder.input_proj[2][1].register_forward_hook(get_activation('backbone_stage3_inputproj'))
      #backbone stage4 inputproj特征图输出,torch.Size([1, 256, 30, 30])
      module.encoder.input_proj[3][1].register_forward_hook(get_activation('backbone_stage4_inputproj'))


      #backbone stage2 self-attention特征图输出,torch.Size([16, 900, 256])
      module.encoder.encoder_ll.layers[0].norm2.register_forward_hook(get_activation('backbone_stage2_attention'))
      #backbone stage4 self-attention特征图输出,torch.Size([1, 900, 256])
      module.encoder.encoder[0].layers[0].norm2.register_forward_hook(get_activation('backbone_stage4_attention'))

      #获取双向跨尺度特征融合结果
      #stage1 fusion result torch.Size([1, 256, 240, 240])
      module.decoder.input_proj[0].conv.register_forward_hook(get_activation_input('stage1_featurefusion_result'))
      #torch.Size([1, 256, 120, 120])
      module.decoder.input_proj[1].conv.register_forward_hook(get_activation_input('stage2_featurefusion_result'))
      #torch.Size([1, 256, 60, 60])
      module.decoder.input_proj[2].conv.register_forward_hook(get_activation_input('stage3_featurefusion_result'))
      #torch.Size([1, 256, 30, 30])
      module.decoder.input_proj[3].conv.register_forward_hook(get_activation_input('stage4_featurefusion_result'))

      


      image = Image.open(imagePath).convert('RGB')  # 使用PIL库打开图像
      #转换为tensor
      tensor_image = self.transform(image).to(self.device)
      # 前向传播
      output = module(tensor_image)
      

      return activations


    def testImg(self,imgpath):
        self.eval()
        module = self.ema.module if self.ema else self.model
         
        # 加载图片
        image = Image.open(imgpath).convert('RGB')
        
        # 定义图像转换
        
        transform = transforms.Compose([
            transforms.Resize((960,576)),  # 调整图像大小，这里假设为800x800
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
       
        image_data = transform(image).unsqueeze(0).cuda()   # 添加批次维度
         
        
        with torch.no_grad():  # 不计算梯度
          prediction = module(image_data)
          print(prediction)
          logits_tensor = torch.tensor(prediction['pred_logits'])

          # 应用 softmax 归一化
          probabilities = torch.nn.functional.softmax(logits_tensor, dim=-1)
          # 应用 softmax 来转换 logits 至概率
          #probabilities = torch.nn.functional.log_softmax(prediction['pred_logits'], dim=1)
          print(probabilities)
          # 获取边界框
          bboxes = prediction['pred_boxes']
          # 假设你想打印每个边界框和它的最可能类别
          for i, (bbox, logits) in enumerate(zip(bboxes, probabilities)):
              class_id = torch.argmax(logits).item()  # 获取概率最高的类别 ID
              print(f"Object {i}: Class {class_id}, BBox {bbox}")  
          #print(prediction)
          #bboxes = prediction.pred_boxes.cpu()  # 将tensor移动到CPU
          bboxes_norm = prediction['pred_boxes'].detach().cpu().numpy()[0]  # 转换为NumPy数组，如果在GPU上
          #labels = prediction['labels'].detach().cpu().numpy()
          scores = probabilities.detach().cpu().numpy()[0]
 
          # 图像尺寸
          image_width = 960
          image_height = 576

          # 转换为像素坐标
          pixel_coords = np.zeros_like(bboxes_norm)
          pixel_coords[:, 0] = bboxes_norm[:, 0] * image_width
          pixel_coords[:, 1] = bboxes_norm[:, 1] * image_height
          pixel_coords[:, 2] = bboxes_norm[:, 2] * image_width
          pixel_coords[:, 3] = bboxes_norm[:, 3] * image_height

          # 将浮点数结果转换为整数
          #pixel_coords = pixel_coords.astype(int)

          # 输出像素坐标
          print(pixel_coords)
          #numpy_array = bboxes.numpy()
          fmt = '[' + ','.join(['%f']*pixel_coords.shape[1]) + '],'
          np.savetxt('bboxes.txt', pixel_coords, fmt=fmt)
          np.savetxt('scores.txt', scores, delimiter=',',fmt='%.4f')


def inferenceImage(self,module,imgpath):
        # 加载图片
        image = Image.open(imgpath).convert('RGB')  
        # 定义图像转换
        transform = transforms.Compose([
            transforms.Resize((960,960)),  # 调整图像大小，这里假设为800x800
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        image_data = transform(image).unsqueeze(0).cuda()   # 添加批次维度
        # 记录推理开始时间
        start_time = time.time() 
        with torch.no_grad():  # 不计算梯度
          prediction = module(image_data)
          #print(prediction)
        # 记录推理结束时间
        end_time = time.time()
        return end_time-start_time

def inferenceTime(self,imgpath):
        self.eval()
        module = self.ema.module if self.ema else self.model
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
      
        # 推理并计算平均推理时间
        total_time = 0
        imageNum=0
        if imgpath!=None:
            #如果是目录
            if os.path.isdir(imgpath):
              print("ext:0000")
              # 获取当前目录下的所有文件名
              filenames = os.listdir(imgpath)
              # 遍历文件名
              for filename in filenames:
                    # 获取文件扩展名
                    ext = filename.split('.')[-1].lower()
                    # 检查文件是否为图片
                    if ext in image_extensions:
                      imageNum=imageNum+1
                      filepath=os.path.join(imgpath,filename)
                      infer_time=self.inferenceImage(module,filepath)
                      print("%d.inference %s time:%f"%(imageNum,filepath,infer_time))
                      total_time += infer_time
                      
            
            elif os.path.isfile(imgpath):
              
              filename=os.path.basename(imgpath)
              # 获取文件扩展名
              ext = filename.split('.')[-1].lower()
              
              # 检查文件是否为图片
              if ext in image_extensions:
                imageNum=imageNum+1
                # 累计单张图片的推理时间
                infer_time=self.inferenceImage(module,imgpath)
                print("%d.inference %s time:%f"%(imageNum,imgpath,infer_time))
                total_time += infer_time
            
        # 计算平均推理时间
        print("total_time:%f\n"%total_time)
        print("imageNum:%d\n"%imageNum)
        if imageNum>0:
            average_inference_time = total_time /imageNum
            print(f'Average inference time per image: {average_inference_time:.3f} seconds')