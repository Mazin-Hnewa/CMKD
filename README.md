# Cross Modality Knowledge Distillation for Robust VRU detection in low light and adverse weather condition

The goal of this project is to use knowledge distillation techniques to improve the performance of the object detectors (for VRU and Large animals) in adverse weather and low light conditions.

This implementation is based on [Faster R-CNN](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf) with ResNet50-FPN backone in Pytorch using [Seeing Through Fog](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/) dataset. 
## Usage
- Install [PyTorch](https://pytorch.org/).
- Download the data from [here](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EQiY_z8k_1FOnYtWzN-JljcB0k96HO5azGNu_rZsPq4jIg?e=TUhJtb) and extract the ZIP file in `data/` folder.
- Download the [trained teacher network](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EQbkqtMSPXRHmkirHyYfStUBd5ktb0Mh4Q81noLXhx2tOQ?e=boom6E) or train it by running this comment. The teacher network is trained using both RGB images and 3 Gated slices in the dataset.
```
python train_teacher.py
``` 
- train Cross Modality Knowledge Distillation (CMKD) method based on Mean Squred Error (MSE) of backnone features by running this comment:
```
python train_cmkd_mse.py
```
- train Cross Modality Knowledge Distillation (CMKD) method based adversarial training of backnone features by running this comment:
```
python train_cmkd_adv.py
```
- The trained network can be tested using valand test sets by changing the name of tested weights file in `test.py` line 109 and running this comment:
```
python test.py
```
- The baseline network can be trained by running this comment. Baseline is trained using only RGB images without CMKD. 
```
python train_baseline.py
```
## Results & Pretrained Weights
|Model|COCO mAP val set| COCO mAP test set| Trained model|
|---|---|---|---|
Teacher|25.8|27.5|[download](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EQbkqtMSPXRHmkirHyYfStUBd5ktb0Mh4Q81noLXhx2tOQ?e=boom6E)
|Baseline|22.5|24.2|[download](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EfTjUsojmxJJmSXrIaX7b98Bdv3NmER5iJ6UOG9DV0t8qA?e=FVdD5X)
|CMKD-MSE|23.6|25.4|[download](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EcyNYGUdSVVHmldwy9ytTXABwXw1loMY9uomx4iFRsrFMw?e=hkwPqP)
|CMKD-Adv|24.2|26.0|[download](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EcJ5AiKKSKZGgnR9q2NzmYABnYYqeN9v7gwxfm-0wGGBSA?e=AxDSbc)
