# This repository is mainly used to store youtube teaching files.

## Please subscribe to my channel

[![https://img.youtube.com/vi/erOPkalrq4A/maxresdefault.jpg](http://img.youtube.com/vi/erOPkalrq4A/0.jpg)](http://www.youtube.com/watch?v=erOPkalrq4A "My channel ")


## How to use

1. Download the dataset from the [here](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip), and extract in root folder.
2. If system is Windows, check install the pycocotools Module.
3. After training, you can get a stored neural network test.pyh, and then you can run the test show_eample.py to predict the image.


## dataset like
```
PennFudanPed/
  PedMasks/
    FudanPed00001_mask.png
    FudanPed00002_mask.png
    FudanPed00003_mask.png
    FudanPed00004_mask.png
    ...
  PNGImages/
    FudanPed00001.png
    FudanPed00002.png
    FudanPed00003.png
    FudanPed00004.png
```

![image](https://github.com/kmolLin/yt_code_share/blob/master/torch_maskrcnn/reference/original.png?raw=true)
![image](https://github.com/kmolLin/yt_code_share/blob/master/torch_maskrcnn/reference/result.jpg?raw=true)


## Special

If you have realsense L515 camera, the realsense.py can run real time camera and classifier the result.