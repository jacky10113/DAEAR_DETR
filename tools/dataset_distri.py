import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

#绘制标注框尺寸分布散点图
def plot_bbox_scatterplot(df):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 创建一个1行2列的子图布局

    # 第一个散点图：中心点的x和y坐标
    sns.scatterplot(x='Box_x_center', y='Box_y_center', data=df, ax=axs[0], color='blue', s=100)
    axs[0].set_title('Distribution of Bounding Box Centers')
    axs[0].set_xlabel('Center X Coordinate')
    axs[0].set_ylabel('Center Y Coordinate')

    # 第二个散点图：宽度和高度
    sns.scatterplot(x='BBox_Width', y='BBox_Height', data=df, ax=axs[1], color='green', s=100)
    axs[1].set_title('Width vs. Height of Bounding Boxes')
    axs[1].set_xlabel('Width')
    axs[1].set_ylabel('Height')

    plt.tight_layout()  # 调整布局以防止重叠
    plt.show()

 

if __name__ == '__main__':
  train_ann_file='../datasets/pedestrainDatasets/BusPassengers/buspassenger_w960_h576_20240229_random/train_seg.json'
  val_ann_file='../datasets/pedestrainDatasets/BusPassengers/buspassenger_w960_h576_20240229_random/val_seg.json'
  test_ann_file='../datasets/pedestrainDatasets/BusPassengers/buspassenger_w960_h576_20240229_random/test_seg.json'
  ann_file=[train_ann_file,val_ann_file,test_ann_file]

  data = {
      'Image_ID': [],
      'Label': [],
      'BBox_x': [],
      'BBox_y': [],
      'BBox_Width': [],
      'BBox_Height': []
  }

  for item in ann_file:
    with open(item, 'r') as f:
        json_data= json.load(f)
        allImages= json_data["images"]
        Annotations=json_data["annotations"]
        for line in Annotations:
          data['Image_ID'].append('train_'+str(line['image_id']))
          data['Label'].append('buspassenger')
          data['BBox_x'].append(line['bbox'][0])
          data['BBox_y'].append(line['bbox'][1])
          data['BBox_Width'].append(line['bbox'][2])
          data['BBox_Height'].append(line['bbox'][3])



  # 示例数据
  """ data = {
      'Image_ID': ['img1', 'img1', 'img2', 'img2', 'img3', 'img3'],
      'Label': ['person', 'person', 'person', 'person', 'person', 'person'],
      'BBox_Width': [50, 60, 55, 45, 65, 75],
      'BBox_Height': [80, 90, 85, 75, 95, 85]
  } """
  df = pd.DataFrame(data)
  # 计算中心点坐标
  df['Box_x_center'] = df['BBox_x'] + df['BBox_Width'] / 2
  df['Box_y_center'] = df['BBox_y'] + df['BBox_Height'] / 2

  plot_bbox_scatterplot(df)
  