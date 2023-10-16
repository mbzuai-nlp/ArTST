<div align="center">

<h1> ArTST </h1>

<a href=''> <a href=''><img src='https://img.shields.io/badge/paper-ArXiv-red'></a> &nbsp;  <a href='https://artstts.wixsite.com/artsttts'><img src='https://img.shields.io/badge/demo-Page-green'></a> &nbsp;

<div>
    <a href='' target='_blank'>Hawau Olamide Toyin <sup>*,1,2</sup> </a>&emsp;
    <a href='' target='_blank'>Amirbek Djanibekov <sup>*,1,2</a>&emsp;
    <a href='' target='_blank'>Ajinkya Kulkarni <sup>1,2</a>&emsp;
    <a href='' target='_blank'>Hanan Aldarmaki <sup>1,2</a>&emsp;
</div>
<br>
<div>
    <sup>1</sup> MBZUAI &emsp; <sup>2</sup> Speech Lab, MBZUAI &emsp; <sup>*</sup> equal contribution &emsp; 
</div>
<br>
<i><strong><a target='_blank'>ArabicNLP 2023</a></strong></i>
<br>
</div>


# Checkpoints

## Pre-Trained Models

 Model | Pre-train Dataset | Model |
| --- | --- | --- |
| ArTST base | MGB2 | [OneDrive](https://mbzuaiac.sharepoint.com/:u:/s/Interns-Summer23/Eap0It3DUTtIhLnanxJe-SEBeHalIkEoCvJUFB_rARqcdQ?e=HbhV87) |

## Finetuned Models
 Model | FInetune Dataset | Model |
| --- | --- | --- |
| ArTST ASR | MGB2 | [OneDrive](https://mbzuaiac.sharepoint.com/:u:/s/Interns-Summer23/EZhZt4Vs8CFFqLnJ3XeGVZcBgl2aJDcfsbE8q8WrH8HxVA?e=roH9Z2) |
| ArTST TTS | [ClArTTS]() | [Onedrive](https://mbzuaiac.sharepoint.com/:u:/s/Interns-Summer23/EUX97Mhgtm5CizEojxNl2tYB0UFTF4IZ1-OEY1RMdBKZwg?e=PWwc04) |
| ArTST* TTS |  [ClArTTS]() | [Onedrive](https://mbzuaiac.sharepoint.com/:u:/s/Interns-Summer23/EUi9oUDzfy9Ai1zWe428yT4BXXBWlyAJB0MSEG6IoUo01Q?e=1hwh2g)  |



|  |  |  |

# Environment & Installation

Python version == 3.8

1) Clone this repo
```bash
git clone https://github.com/Theehawau/ArTST.git
cd ArTST
conda create -n artst python=3.8
conda activate artst
pip install -r requirements.txt
```
2) Install fairseq
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
python setup.py build_ext --inplace
cd ../
```

# Finetune

## ASR

```bash
bash ASR/finetune.sh
```

## TTS

```bash
bash TTS/finetune.sh
```

# Inference
## ASR

```bash
bash ASR/inference.sh
```

## TTS

```bash
bash TTS/inference.sh
```

# Acknowledgements

ArTST is built on [SpeechT5](https://arxiv.org/abs/2110.07205) Architecture.