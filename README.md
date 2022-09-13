We here provide our code on the GbP datasets. 
Please install all dependencies in the requirement.txt first.
To install torchxrayvision, using the following:
```
pip install torchxrayvision
```
Please comment line 234-239 in the original package of torchxrayvision/models.py as follow:

```
#if hasattr(self, 'apply_sigmoid') and self.apply_sigmoid:
#    out = torch.sigmoid(out)

#if hasattr(self,"op_threshs") and (self.op_threshs != None):
#    out = torch.sigmoid(out)
#    out = op_norm(out, self.op_threshs)
```
To run the training code:
```
. ./script/train.sh
```
We provide the csv files under ./dataset. 
As all data in GbP are sampled from NIH ChestX-ray14 dataset, 
one can find the images from the original dataset using the csv provided.
The data directories is like follows:
```
Pseudo_bias_balanced_learning
  ｜———dataset
    |———csv
      |———GbP-case1
        |———train.csv
        |———valid.csv
        |———test.csv
      |———GbP-case2
        |———train.csv
        |———valid.csv
        |———test.csv
      |———GbP
        |———00015214_010.png
        |———00015284_002.png
```






