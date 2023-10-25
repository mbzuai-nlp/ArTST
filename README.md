<div align="center">

<h1> ArTST </h1>
This repository contains the implementation of the paper:

**ArTST**: **Ar**abic **T**ext and **S**peech **T**ransformer

<a href=''> <a href=''><img src='https://img.shields.io/badge/paper-ArXiv-red'></a> &nbsp;  <a href='https://artstts.wixsite.com/artsttts'><img src='https://img.shields.io/badge/demo-Page-green'></a> &nbsp;

<div>
    <a href='' target='_blank'>Hawau Olamide Toyin <sup>* 1</sup> </a>&emsp;
    <a href='' target='_blank'>Amirbek Djanibekov <sup>* 1</a>&emsp;
    <a href='' target='_blank'>Ajinkya Kulkarni <sup>1</a>&emsp;
    <a href='' target='_blank'>Hanan Aldarmaki <sup>1</a>&emsp;
</div>
<br>
<div>
    <sup>*</sup> equal contribution &emsp; <sup>1</sup> MBZUAI &emsp;
</div>
<br>
<i><strong><a target='_blank'>ArabicNLP 2023</a></strong></i>
<br>
</div>

## ArTST 
ArTST, a pre-trained Arabic text and speech transformer for supporting open-source speech technologies for the Arabic language. The model architecture in this first edition follows the unified-modal framework, SpeechT5, that was recently released for English, and is focused on Modern Standard Arabic (MSA), with plans to extend the model for dialectal and code-switched Arabic in future editions. We pre-trained the model from scratch on MSA speech and text data, and fine-tuned it for the following tasks: Automatic Speech Recognition (ASR), Text-To-Speech synthesis (TTS), and spoken dialect identification. 


## Checkpoints

### Pre-Trained Models

 Model | Pre-train Dataset | Model | Tokenizer |
| --- | --- | --- | --- |
| ArTST base | MGB2 | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/pretrain_checkpoint.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/asr_spm.model)

### Finetuned Models
 Model | FInetune Dataset | Model | Tokenizer |
| --- | --- | --- | --- |
| ArTST ASR | MGB2 | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/MGB2_ASR.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/asr_spm.model)|
| ArTST TTS | [ClArTTS]() | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/CLARTTS_ArTST_TTS.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/tts_spm.model)|
| ArTST* TTS |  [ClArTTS]() | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/CLARTTS_ArTSTstar_TTS.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/tts_spm.model)|


## Environment & Installation

Python version: 3.8+

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
```

## Data Preparation

#### Speech

For pretraining, follow the steps for preparing wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) and preparing HuBERT label [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans).

For finetuning TTS task, an extra column is required in the speech manifest file for speaker embedding. To generate speaker embedding, we use [speech brain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb). 
[Here](./main/TTS/hubert_labels/ASC) is a DATA_ROOT sample folder structure that contains manifest samples.

#### Text 

Pretrain:

Please use [fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess) to generate the index and bin files for the text data. We use sentencepiece to pre-process the text, we've provided our SPM models and [dictionary](./main/dict.txt) in this repo. You need to use the SPM model to process the text and then use [fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess) with the provided dictionary to get the index and bin files. Note that after SPM processes sentences, the resulting text should have individual characters separated by space.

For Finetuning, a simple text file containing corresponding texts on each line suffices. See [here](.main/ASR/labels/ASC/) for sample manifest.

## Training

The bash files contain the parameters and hyperparameters used for pretraining and finetuning. Find more details on training arguments [here](https://fairseq.readthedocs.io/en/latest/)


### Pretrain

``` bash
bash /ArTST/pretrain/train.sh
```

### Finetune

#### ASR

```bash
bash /ArTST/ASR/finetune.sh
```

#### TTS

```bash
bash /ArTST/TTS/finetune.sh
```

## Inference
#### ASR

```bash
bash /ArTST/ASR/inference.sh
```

#### TTS

```bash
bash /ArTST/TTS/inference.sh
```

# Acknowledgements

ArTST is built on [SpeechT5](https://arxiv.org/abs/2110.07205) Architecture.