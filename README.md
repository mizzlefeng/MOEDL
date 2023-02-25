# MOEDL

## 1 System requirements:

```
Hardware requirements: 
	Model.py requires a computer with enough RAM to support the in-memory operations.
	Operating system：windows 10

Code dependencies:
	python '3.7.4' (conda install python==3.7.4)
	pytorch-GPU '1.10.1' (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch)
	numpy '1.23.5' (conda install numpy==1.23.5)
	pandas '1.5.2' (conda install pandas==1.5.2)
	scikit-learn '1.2.0' (conda install scikit-learn==1.2.0)
	matplotlib '3.6.2' (conda install matplotlib==3.6.2)
```

## 2 Installation guide:

```
First, install CUDA 10.2 and CUDNN 8.2.0.
Second, install Anaconda3. Please refer to https://www.anaconda.com/distribution/ to install Anaconda3.
Third, install PyCharm. Please refer to https://www.jetbrains.com/pycharm/download/#section=windows.
Fourth, open Anaconda Prompt to create a virtual environment by the following command:
	conda env create -n env_name python=3.7.4
```

## 3 Instructions to run on a small real dataset(demo)

```
Based on a small dataset from BRCA dataset:
	First, put folder demodata, demo.py and Model.py into the same folder.
	Second, use PyCharm to open demo.py and set the python interpreter of PyCharm.
	Third, modify codes in demo.py to set the path for loading data and the path for saving the trained model.
	Fourth, run Demo.py in PyCharm.

	Expected output：
		A txt file with timestamps and results of all evaluation metrics and training curves.
```

## 4 Instructions for use(three benchmark datasets are included in our data):

```
Based on BRCA dataset:
	First, put folder BRCAdata, BRCA10fold.py and Model.py into the same folder.
	Second, use PyCharm to open BRCA10fold.py and set the python interpreter of PyCharm.
	Third, modify codes in BRCA10fold.py to set the path for loading data and the path for saving the trained model.
	Fourth, run BRCA10fold.py in PyCharm.

	Expected output：
		A txt file with timestamps and results of all evaluation metrics and training curves.

Based on KIPAN dataset:
	First, put folder KIPANdata, KIPAN10fold.py and Model.py into the same folder.
	Second, use PyCharm to open KIPAN10fold.py and set the python interpreter of PyCharm.
	Third, modify codes in KIPAN10fold.py to set the path for loading data and the path for saving the trained model.
	Fourth, run KIPAN10fold.py in PyCharm.

	Expected output：
		A txt file with timestamps and results of all evaluation metrics and training curves.

Based on NSCLC dataset:
	First, put folder NSCLCdata, NSCLC10fold.py and Model.py into the same folder.
	Second, use PyCharm to open NSCLC10fold.py and set the python interpreter of PyCharm.
	Third, modify codes in NSCLC10fold.py to set the path for loading data and the path for saving the trained model.
	Fourth, run NSCLC10fold.py in PyCharm.

	Expected output：
		A txt file with timestamps and results of all evaluation metrics and training curves.
```
