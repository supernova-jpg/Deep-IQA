# CNN-FRIQA
这是论文Convolutional Neural Network for Full-Reference color Image Quality Assessment的代码复现，本人的工作成果主要是对[Bobholamovic](https://github.com/Bobholamovic)大佬的工作进行了初步的debug和代码注释，目前针对TID2013数据集已经跑通了流程。

论文可以从[deepIQA](https://arxiv.org/abs/1612.01697)进行下载. 这篇论文的官方复现位于[dmaniry/deepIQA](https://github.com/dmaniry/deepIQA). 
  
  
## 运行源代码所需要的环境
> Ubuntu 16.04 64-bit, Visual Studio Code, Python 3.5.2, Pytorch 0.4.0, shutils
  
  
  
## 如何运行代码

### 数据准备（从官网下载数据集）
在运行代码之前，请从[tid2013](https://www.ponomarenko.info/tid2013.html)下载数据集，下载完成后，解压到'./data/TID2013/'这一相对路径中。

### 训练集/测试集分割，生成.json文件
运行`TID2013_make_list.py`,这个程序会自动将数据集分割为测试集和验证集，并生成一个`.json`文件。
目前数据已经被存储到了 `.json` 文件中. TID2013数据集中，失真图像被存储在 `data-dir` 文件夹中，而质量分数（真实值）包含在  `json ` 对象的三维数组中，分别用  `img `、 `ref ` 和  `score ` 字段指定。例如， `train_data.json ` 一般为如下格式：

```
{
  "img":
    [
      "distorted/img11_2_4.bmp", 
      "distorted/img6_3_3.bmp"
    ], 
  "ref":
    [
      "images/img11.bmp", 
      "distorted/img6.bmp"
    ], 
  "score":
    [
      0.5503, 
      0.4312
    ]
}
```
(this has been prettified as everthing actually on one line)

同时，运行完 `TID2013_make_list.py`这一程序后，您应该在TID2013数据集文件夹还能看到 `val_data.json` 和 `test_data.json` 这两个.json文件。 您可以在`main.py`的`argparser`模块中修改`list-dir`,如果代码未指定`list-dir`,`list-dir`的默认值就是 `data-dir`. 


### 运行main.py
转到项目文件夹下

如果您需要训练模型的话，请运行：
```bash
python3 main.py train --resume pretrianed_model_path --data-dir DIR_OF_DATASET
```

如果`pretrianed_model_path`没有被指定清楚，则应当运行如下命令：

```bash
python3 main.py train --resume pretrianed_model_path | tee train.log
```
以获取训练日志. 
  
如果您需要验证模型的话，请运行：
```bash
python3 main.py train --evaluate --resume pretrained_model_path
```
  
如果您需要测试模型的话，请运行：
```bash
python3 main.py test --resume pretrained_model_path
```

  
### Todo

1. 目前我们只完成了代码的训练和测试工作，虽然通过SROCC等指标可以衡量训练效果，但对于真实图片指标设计，目前该模块还未开发
2. 还尚未做好预训练模型
  
## 实验结果
在最好的条件下， `SROCC`指标可以达到约`0.95`（衡量人眼主观评分和模型给出指标分数），甚至更高。 

  
## 致谢
+ 模型复现的论文是 [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment](https://arxiv.org/abs/1612.01697)
+  `MS-SSIM` 的pytorch实现参考了[lizhengwei1992/MS_SSIM_pytorch](https://github.com/lizhengwei1992/MS_SSIM_pytorch.git)的代码。
+ 一部分的代码设计参考了 [fyu/drn](https://github.com/fyu/drn)的工作。

With best thanks!  

  
