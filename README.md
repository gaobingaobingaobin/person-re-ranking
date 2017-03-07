# Re-ranking Person Re-identification with k-reciprocal Encoding 
=============
This code was used for experiments for the paper "Re-ranking Person Re-identification with k-reciprocal Encoding".

If you find this code useful in your research, please consider citing:

    @article{zhong2017re,
      title={Re-ranking Person Re-identification with k-reciprocal Encoding},
      author={Zhong, Zhun and Zheng, Liang and Cao, Donglin and Li, Shaozi},
      booktitle={CVPR},
      year={2017}
    }


### Requirements: Caffe

Requirements for `Caffe` and `matcaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

### Installation
1. Build Caffe and matcaffe
    ```Shell
    cd $Re-ranking_ROOT/caffe
    # Now follow the Caffe installation instructions here:
    # http://caffe.berkeleyvision.org/installation.html
    make -j8 && make matcaffe
    ```
2. Download pre-computed imagenet models, Market-1501 dataset and CUHK03 dataset
  ```Shell
  Please download the pre-trained imagenet models and put it in the "data/imagenet_models" folder.
  Please download Market-1501 dataset and unzip it in the "evaluation/data/Market-1501" folder. 
  Please download CUHK03 dataset and unzip it in the "evaluation/data/CUHK03" folder.
  ```
- [Pre-trained imagenet models](https://pan.baidu.com/s/1o7YZT8Y)
  
- [Market-1501](https://pan.baidu.com/s/1ntIi2Op)

- [CUHK03](https://pan.baidu.com/----)

### Training and testing IDE model with our re-ranking method

1. Training 
  ```Shell
  cd $Re-ranking_ROOT
  # train IDE ResNet_50 for Market-1501
  ./experiments/Market-1501/train_IDE_ResNet_50.sh
  
  # train IDE ResNet_50 for CUHK03
  ./experiments/CUHK03/train_IDE_ResNet_50_labeled.sh
  ./experiments/CUHK03/train_IDE_ResNet_50_detected.sh
  ```
2. Feature Extraction
  ```Shell
  cd $Re-ranking_ROOT/evaluation
  # extract feature for Market-1501
  matlab Market_1501_extract_feature.m
  
  # extract feature for CUHK03
  matlab CUHK03_extract_feature.m
  # The IDE features are saved under: "market_evaluation/feat"
  ```
  
3. Evaluation
  ```Shell
    # evaluation for Market-1501
    matlab Market_1501_evaluation.m
    
    # evaluation for CUHK03
    matlab CUHK03_evaluation.m
  ``` 
  
### Results
You can download our pre-trained IDE models and IDE features, and put them in the "out_put/"  and "evaluation/feat" folder, respectively. 

- [IDE models](https://pan.baidu.com/123) 

- [IDE features](https://pan.baidu.com/123)

Using the above models and features with our re-ranking method, you can reproduce the results as follows:


### Contact us

If you have any questions about this code, please do not hesitate to contact us.

[Zhun Zhong](http://zhunzhong.site)

[Liang Zheng](http://liangzheng.com.cn)
