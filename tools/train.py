"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import torch 
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_attention(attention_map):
    sns.heatmap(attention_map, cmap='inferno')
    plt.title('Attention Weights')
    plt.xlabel('Query Index')
    plt.ylabel('Key Index')
    plt.show()

#保存特征图
def save_feature_maps(feature_maps,savePath,prefix=''):
    num_maps = feature_maps.size(0)
    for i in range(num_maps):
        plt.imshow(feature_maps[i].cpu().numpy(),cmap='viridis')
        plt.axis('off')  # 不显示坐标轴
        filePath=os.path.join(savePath,prefix+'_'+str(i)+'.png')
        plt.savefig(filePath, bbox_inches='tight', pad_inches=0.0)
        plt.close()

 
def plot_feature_maps(feature_maps, num_cols=8):
    
    num_maps = feature_maps.size(0)
    num_rows = (num_maps + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    elif args.paramnums_floaps:
        solver.get_paramnum_floaps(1,960,960)#num_frames,batch_size,width,height
    elif args.fps:
        solver.get_fps(1,1,960,960)#num_frames,batch_size,width,height
    elif args.jittrace:
        solver.jitTrace(1,1,960,960)
    elif args.featrueCom!=None:
        #获取特征图结果   
        activations =solver.get_feature_maps(args.featrueCom)
        #输出指定钩子获得的特征图维度，conv1_output维度输出为torch.Size([1, 64, 480, 480])
        print("backbone_stage1.shape:"+str(activations['backbone_stage1'].shape))
        print("backbone_stage2.shape:"+str(activations['backbone_stage2'].shape))
        print("backbone_stage3.shape:"+str(activations['backbone_stage3'].shape))
        print("backbone_stage4.shape:"+str(activations['backbone_stage4'].shape))

        print("backbone_stage1_inputproj.shape:"+str(activations['backbone_stage1_inputproj'].shape))
        print("backbone_stage2_inputproj.shape:"+str(activations['backbone_stage2_inputproj'].shape))
        print("backbone_stage3_inputproj.shape:"+str(activations['backbone_stage3_inputproj'].shape))
        print("backbone_stage4_inputproj.shape:"+str(activations['backbone_stage4_inputproj'].shape))
     
        #因为stage2自注意力运算完后是生成16个小窗口，所以还需要将其还原为初始窗口大小
        print("backbone_stage2_attention.shape:"+str(activations['backbone_stage2_attention'].shape))
        # 将处理后的窗口拼接回原来的特征图形状
        B,C, H,W=activations['backbone_stage2_inputproj'].shape
        h,w=activations['backbone_stage4_inputproj'].shape[2:]
        src_s2=activations['backbone_stage2_attention']
        src_s2 = src_s2.view(B, H // h, W // w, h, w, C)
        src_s2 = src_s2.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, C, H, W) 
        print("backbone_stage2_attention_deal.shape:"+str(src_s2.shape))
        print("backbone_stage4_attention.shape:"+str(activations['backbone_stage4_attention'].shape))
        
        #因为backbone_stage4_attention.shape:torch.Size([1, 900, 256])，需要转换为(B,C,W,H)再输出特征图
        src_s4=activations['backbone_stage4_attention']
        src_s4=src_s4.permute(0,2,1).contiguous().view(B,C, h, w)
        print("backbone_stage4_attention_deal.shape:"+str(src_s4.shape))

        #获取特征融合结果
        #第一阶段输出融合结果
        print("stage1_featurefusion_result.shape:"+str(activations['stage1_featurefusion_result'].shape))
        #第二阶段输出融合结果
        print("stage2_featurefusion_result.shape:"+str(activations['stage2_featurefusion_result'].shape))
        #第三阶段输出融合结果
        print("stage3_featurefusion_result.shape:"+str(activations['stage3_featurefusion_result'].shape))
        #第四阶段输出融合结果
        print("stage4_featurefusion_result.shape:"+str(activations['stage4_featurefusion_result'].shape))
        # 可视化第二层的部分特征图
        #plot_feature_maps(activations['backbone_stage1'][0, :40])
        #save_feature_maps(activations['backbone_stage1'][0,:256],'/home/Disk/hou/video_structure/models_trained/object_detection/rtdetr_dualencode_level4_encodestage2_residual_r50vd_6x_buspassenger_random/20240407_093749/features_images',prefix='backbone_stage1')
        #plot_feature_maps(src_s2[0, :4])
        #可视化注意力权重
        visualize_attention(activations['backbone_stage4'][0,1,:].cpu())
        visualize_attention(src_s4[0,1,:].cpu())
    elif args.testimg:
        solver.testImg(args.imgpath)
    elif args.infertime:
        solver.inferenceTime(args.imgpath)
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )#Fully load model
    parser.add_argument('--tuning', '-t', type=str, )#only load model for tuning and skip missed/dismatched keys
    parser.add_argument('--imgpath', '-i', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--paramnums-floaps', action='store_true', default=False,)
    parser.add_argument('--fps', action='store_true', default=False,)
    parser.add_argument('--jittrace', action='store_true', default=False,)
    parser.add_argument('--featrueCom', '-f',type=str,)
    parser.add_argument('--testimg', action='store_true', default=False,)
    parser.add_argument('--infertime', action='store_true', default=False,)
    args = parser.parse_args()

    main(args)
