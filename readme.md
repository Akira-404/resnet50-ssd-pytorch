# ResNet50-SSD-PyTorch

## 环境配置：

```
numpy==1.17.0
matplotlib
tqdm==4.42.1
pycocotools
torch==1.6.0
torchvision==0.7.0
lxml
Pillow
```

## 文件结构：

```
├── src: 实现SSD模型的相关模块    
│     ├── resnet50_backbone.py   使用resnet50网络作为SSD的backbone  
│     ├── ssd_model.py           SSD网络结构文件 
│     └── utils.py               训练过程中使用到的一些功能实现
├── train_utils: 训练验证相关模块（包括cocotools）  
├── my_dataset.py: 自定义dataset用于读取VOC数据集    
├── train_ssd300.py: 以resnet50做为backbone的SSD网络进行训练    
├── train_multi_GPU.py: 针对使用多GPU的用户使用    
├── predict_test.py: 简易的预测脚本，使用训练好的权重进行预测测试    
├── pascal_voc_classes.json: pascal_voc标签文件    
├── plot_curve.py: 用于绘制训练过程的损失以及验证集的mAP
└── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record
```

## 预训练权重下载地址（下载后放入src文件夹中）：

```
ResNet50+SSD: https://ngc.nvidia.com/catalog/models
搜索ssd -> 找到SSD for PyTorch(FP32) -> download FP32 -> 解压文件
```



## 训练方法

- 确保提前准备好数据集
- 确保提前下载好对应预训练模型权重
- 单GPU训练或CPU，直接使用train_ssd300.py训练脚本
- 若要使用多GPU训练，使用 "python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py" 指令,nproc_per_node参数为使用GPU数量

## 