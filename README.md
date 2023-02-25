# MOEDL

 A Deep Learning Model for Cancer Subtype Prediction

1.代码文件夹中包含了*10fold.py可执行的程序文件，其中Model.py为MOEDL的模型文件。

2.代码文件夹中的子文件夹*data是*10fold.py程序运行时所需要的数据，包含9个csv文件：
	  *_G feature：基因表达的特征名称，
	  *_G：基因表达矩阵数据，
	  *_M feature：DNA甲基化的特征名称，
	  *_M：DNA甲基化矩阵数据，
	  *_mi feature：miRNA的特征名称，
	  *_mi：miRNA表达矩阵数据，
	  *_P feature：反相蛋白质微阵列的特征名称，
	  *_P：反相蛋白质微阵列数据，
	  *_label：BRCA的标签信息

3.如何运行*10fold.py程序：安装pytorch后，直接运行运行此程序即可。

4.结果：在程序运行完之后自动在当前文件夹中生成MOEDL 10fold.txt作为结果文件，此文件的内容就是10次10折验证结果。  



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

## 3 Instructions to run on a small real dataset(Demo)

```
Based on a small dataset from kiba dataset:
	First, put folder demodata, demo.py, model.py and Demo.py into the same folder.
	Second, use PyCharm to open Demo.py and set the python interpreter of PyCharm.
	Third, modify codes in Demo.py to set the path for loading data and the path for saving the trained model. The details are as follows:
		line 249 in Demo.py
		line 268 in Demo.py
	Fourth, open Anaconda Prompt and enter the following command:
		activate env_name
	Fifth, run Demo.py in PyCharm.

	Expected output：
		The kiba scores between drugs and targets in test set of the small dataset would be output as a csv file.
	
	Expected run time on a "normal" desktop computer:
		The run time in our coumputer (CPU:Xeon 3106, GPU NVIDIA Geforce RTX 2080 Ti, ARM 64G) is about 5 minutes.

	Note: in the csv file, drug SMILES, protein sequences and binding affinity values are displayed in column 1, column 2 and column 3, respectively. 
```

## 4 Instructions for use(two benchmark datasets are included in our data):

```
Based on kiba dataset:
	First, put folder data_kiba, DataHelper.py, emetrics.py and MRBDTA.py into the same folder.
	Second, use PyCharm to open MRBDTA.py and set the python interpreter of PyCharm.
	Third, modify codes in MRBDTA.py to set the path for loading data and the path for saving the trained model. The details are as follows:
		line 287 in MRBDTA.py
		line 384-388 in MRBDTA.py
	Fourth, open Anaconda Prompt and enter the following command:
		activate env_name
	Fifth, run MRBDTA.py in PyCharm.

	Expected output：
		Results (MSE, CI, RM2) predicted by MRBDTA on test set of KIBA dataset for 5 times would be output as three csv files, respectively.

	Expected run time on a "normal" desktop computer:
		The run time in our coumputer (CPU:Xeon 3106, GPU NVIDIA Geforce RTX 2080 Ti, ARM 64G) is about 168 hours (seven days).

Based on davis dataset:
	First, put folder data_davis, DataHelper.py, emetrics.py and MRBDTA.py into the same folder.
	Second, use PyCharm to open MRBDTA.py and set the python interpreter of PyCharm.
	Third, modify codes in MRBDTA.py to set parameters for the davis dataset. The details are as follows:
		line 279-286 in MRBDTA.py: 'max length for drugs, max length for proteins, trian set, test set'.
		line 291 in MRBDTA.py: 'drug, target, affinity = DH.LoadData(fpath_kiba, logspance_trans=False)' -> 'drug, target, affinity = DH.LoadData(fpath_davis, logspance_trans=True)'.
		line 398 in MRBDTA.py: 'EPOCHS, batch_size, accumulation_steps = 600, 32, 32' -> 'EPOCHS, batch_size, accumulation_steps = 300, 32, 8'.
		line 300-346 in MRBDTA.py
	Fourth, modify codes in MRBDTA.py to set the path for loading data and the path for saving the trained model. The details are as follows:
		line 282 in MRBDTA.py
		line 384-388 in MRBDTA.py
	Fifth, open Anaconda Prompt and enter the following command:
		activate env_name
	Sixth, run MRBDTA.py in PyCharm.

	Expected output：
		Results (MSE, CI, RM2) predicted by MRBDTA on test set of davis dataset for 5 times would be output as three csv files, respectively.

	Expected run time on a "normal" desktop computer:
		The run time in our coumputer (CPU:Xeon 3106, GPU NVIDIA Geforce RTX 2080 Ti, ARM 64G) is about 96 hours (four days).
```
