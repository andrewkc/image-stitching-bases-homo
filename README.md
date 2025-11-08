# Pasos

## Entorno y dependencias
```
conda env create -f environment.yml
conda activate cg-project
```
## Data pre-processing

* Download raw data
```
# GoogleDriver
https://drive.google.com/file/d/19d2ylBUPcMQBb_MNBBGl9rCAS7SU-oGm/view?usp=sharing
# BaiduYun
# https://pan.baidu.com/s/1Dkmz4MEzMtBx-T7nG0ORqA (key: gvor)
```
* Unzip the data to directory "./dataset"
* Rename "dataset/Train" to "dataset/train" and "dataset/Test" to "dataset/test"
* Run "video2img.py"

## Modelo preentrenado

* Download model 
```
# GoogleDriver
https://1drv.ms/u/s!AglwI3TlqfaZhlU4BXo7iYiyLdI5?e=tCygKh
# BaiduYun
https://pan.baidu.com/s/1v4tF_RPEKgc6YF_mfRl7jQ (key: b3yq)
```
* Save it in the directory: experiments/base_model/

## Prueba
```
python3 evaluate.py
```
