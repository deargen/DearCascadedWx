# DearCascadedWx
Public codes for Cascaded Wx algorithm to select genes relative with patients survivals.

A Keras implementation of paper, :   
**[Cascaded Wx: a novel prognosis-related feature selection framework in human lung adenocarcinoma transcroptiomes] Bonggun Shin et al.**

 
**Contacts**
- Your contributions to the repo are always welcome. 
Open an issue or contact me with E-mail `sspark@deargen.me`

## Environment
```
python == 3.6.8
tensorflow-gpu == 1.10.0
keras == 2.2.4
xgboost == 0.90
scikit-learn == 0.21.2
lifelines == 0.21.3
```


## Usage

**Download the TCGA data**

```
$ DearCacadedWx\src]python download_tcga.py
$ DearCacadedWx\src]python split_tcga.py
```
**Get score from elastic-net**

```
$ DearCacadedWx\src]python train_elastic.py
```

**Feature Selection and Performance evaluation**

```
$ DearCacadedWx\src]python run.py
```
