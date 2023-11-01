# Getting Started

This page provides basic usage about RotatedRiceSpikeDet. For installation instructions, please see [install.md](./install.md).

# Train a model

**1. Prepare custom dataset files**

1.1 Make sure the labels format is [poly classname diffcult], e.g., You can set **diffcult=0**
```
  x1      y1       x2        y2       x3       y3       x4       y4       classname     diffcult

1686.0   1517.0   1695.0   1511.0   1711.0   1535.0   1700.0   1541.0   large-vehicle      1
```
![image](https://user-images.githubusercontent.com/72599120/159213229-b7c2fc5c-b140-4f10-9af8-2cbc405b0cd3.png)


1.2 Split the dataset. 
```shell
cd RotatedRiceSpikeDet
python DOTA_devkit/ImgSplit_multi_process.py
```
or Use the orignal dataset. 
```shell
cd RotatedRiceSpikeDet
```

1.3 Make sure your dataset structure same as:
```
parent
├── RotatedRiceSpikeDet
└── datasets
    └── DOTAv1.5
        ├── train_split_rate1.0_subsize1024_gap200
        ├── train_split_rate1.0_subsize1024_gap200
        └── test_split_rate1.0_subsize1024_gap200
            ├── images
                 |────1.jpg
                 |────...
                 └────10000.jpg
            ├── labelTxt
                 |────1.txt
                 |────...
                 └────10000.txt

```

**Note:**
* DOTA is a high resolution image dataset, so it needs to be splited before training/testing to get better performance.

**2. Train**

2.1 Train with specified GPUs. (for example with GPU=3)

```shell
python train.py --device 3
```

2.2 Train with multiple(4) GPUs. (DDP Mode)

```shell
python -m torch.distributed.launch --nproc_per_node 4 train.py --device 0,1,2,3
```

2.3 Train the orignal dataset demo.
```shell
python train.py --data 'data/yolov5obb_demo.yaml' --epochs 10 --batch-size 1 --img 1024 --device 0
```

2.4 Train the splited dataset demo.
```shell
python train.py --data 'data/yolov5obb_demo_split.yaml' --epochs 10 --batch-size 2 --img 1024 --device 0
```


