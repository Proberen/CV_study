# 计算机视觉学习笔记

## Pytorch——识别人民币
使用  **识别人民币**  的案例进行学习

#### RMB_data目录：数据集
#### 练习目录：分步骤进行练习项目
#### 优化目录：整体优化，适合实战项目

## yolo_v3——识别地下城人物(目标检测)
Yolo_v3源码：https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch

本项目使用  **识别地下城人物**  的案例进行学习

#### 建立数据集
1、获取：使用python基础内的screen_capture.py进行屏幕抓取（见学习目录下data目录下的images目录）

2、标注：使用专业的标注软件进行标注，获取到xml文件（见学习目录下data目录下的xml目录）

#### 数据的预处理
1、训练集、测试集、验证集划分（学习目录下的makeTxt.py文件）

2、xml文件转txt文件（学习目录下的voc_label.py文件），最后在txt目录下生成每个xml文件对应的标签结果，在根目录下生成train.txt、val.txt、text.txt

#### 模型训练
运行train.py文件
#### 测试
在data/samples/目录下放置图片或者视频

结果输出到output目录下

最终输出结果预览

![1598671537012](https://user-images.githubusercontent.com/57889284/109632633-38922980-7b82-11eb-9cb4-d123ae58912c.png)

