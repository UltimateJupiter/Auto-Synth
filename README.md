## DX-7 Auto Generating

### Updated for final release

---

This is a prooject amining on giving an approxiemate estimation of the synthesizer parameters of mdaDX10.vst to mimic a recorded sound timbre.

#### A Simple Guide

This project mainly consists of three parts

1. Dataset Generating in folder Dataset_generate

2. Network training and testing in folder Regression


---

### To Teammates

**@jeffery @lucas** please fork this repo and change the generator written in [dataset_gen.py](https://github.com/UltimateJupiter/Auto-Synth/blob/master/dataset_gen.py)

```py
def param_gen(length, param_num): 
    
    # TODO: make a better version

    # A simple random version
    rand_array = np.random.rand(length, param_num)
    rand_array = rand_array * 100

    # make integer
    rand_array.astype(np.uint8)

    return rand_array
```

---

### Environments

**This project is based on Python 3.6**

**Please install the following packages under pip3**

| Package Needed  	| Usage                      	|
|-----------------	|----------------------------	|
| pickle          	| data encoding              	|
| scipy           	| scientific computing       	|
| numpy           	| scientific computing       	|
| multiprocessing 	| multi thread computing     	|
| tqdm            	| visualization              	|
| tensorflow      	| machine learning core      	|
| keras           	| machine learning embedding 	|
| IPython         	| debugging                  	|

---

### Notes
**The code for dataset generating part is completed, network is under construction.**

The current version encode the dataset into pure binary code (e.g. [Datasets/RandomTest/RandomTest-params.pkl](https://github.com/UltimateJupiter/Auto-Synth/blob/master/Datasets/RandomTest/RandomTest-params.pkl))

Two files exists in the dataset folder:

> NAME-params.pkl (TRAINING_SET_SIZE * PARAM_NUMBER)
>
> NAME-wavs.pkl (TRAINING_SET_SIZE * FRAMES_PER_FILE)

These can be decoded by [pickle](https://docs.python.org/3/library/pickle.html)

---

### Training Data Generating

1. Change the function param_gen in dataset_gen.py
2. Modify Config.py to change the parameters including parameter number, thread number, and trainig set size
3. Import dataset_gen to any code or in IPython, and call dataset_gen.generate(**NAME**). A folder called NAME will appear in the Datasets Folder.
