# PointNet.pytorch
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `pointnet.py`.


# Running steps

### 1. Create environment
```
conda env create -f env.yml # create environment
source activate pointnet_pytorch # activate installed environment
```

### 2. Download data
1. [Download data here](https://drive.google.com/open?id=1nlDdKajIjFLqHlMe3_OMrlgUxPT7BFpz)
1. Extract to pointnet.pytorch project folder, we will obtain a folder of data named `shapenetcore_partanno_segmentation_benchmark_v0`

### 3. Train segmentation and classification
```
python train_segmentation.py # train 3D model segmentaion
```

### 4. Visualize result
```
bash build.sh # build C++ code for visualization
python show_seg.py --model seg/seg_model_2.pth  # show segmentation results
```

# Performance
Without heavy tuning, PointNet can achieve 80-90% performance in classification and segmentaion on this [dataset](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html). 

Sample segmentation result:
![seg](https://raw.githubusercontent.com/fxia22/pointnet.pytorch/master/misc/show3d.png?token=AE638Oy51TL2HDCaeCF273X_-Bsy6-E2ks5Y_BUzwA%3D%3D)


# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)
