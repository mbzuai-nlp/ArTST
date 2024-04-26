<div align="center">

<h1> ArTST </h1>
This repository contains the implementation of the paper:

**ArTST**: **Ar**abic **T**ext and **S**peech **T**ransformer

<a href=''> <a href='https://aclanthology.org/2023.arabicnlp-1.5/'><img src='https://img.shields.io/badge/paper-Paper-red'></a> &nbsp;  <a href='https://artstts.wixsite.com/artsttts'><img src='https://img.shields.io/badge/project-Page-green'></a> &nbsp; <a href='https://huggingface.co/spaces/MBZUAI/artst-tts-demo'><img src='https://img.shields.io/badge/demo-Spaces-yellow'></a> &nbsp;

<div>
    <a href='https://www.linkedin.com/in/toyinhawau/' >Hawau Olamide Toyin <sup>* 1</sup> </a>&emsp;
    <a href='https://www.linkedin.com/in/amirbek-djanibekov-a7788b201/' target='_blank'>Amirbek Djanibekov <sup>* 1</a>&emsp;
    <a href='https://www.linkedin.com/in/ajinkya-kulkarni-32b80a130/' target='_blank'>Ajinkya Kulkarni <sup>1</a>&emsp;
    <a href='https://linkedin.com/in/hanan-aldarmaki/' target='_blank'>Hanan Aldarmaki <sup>1</a>&emsp;
</div>
<br>
<div>
    <sup>*</sup> equal contribution &emsp; <sup>1</sup> MBZUAI &emsp;
</div>
<br>
<i><strong><a href='https://aclanthology.org/2023.arabicnlp-1.5/' target='_blank'>ArabicNLP 2023</a></strong></i>
<br>
</div>

## ArTST 
ArTST, a pre-trained Arabic text and speech transformer for supporting open-source speech technologies for the Arabic language. The model architecture in this first edition follows the unified-modal framework, SpeechT5, that was recently released for English, and is focused on Modern Standard Arabic (MSA), with plans to extend the model for dialectal and code-switched Arabic in future editions. We pre-trained the model from scratch on MSA speech and text data, and fine-tuned it for the following tasks: Automatic Speech Recognition (ASR), Text-To-Speech synthesis (TTS), and spoken dialect identification. 

## Update
 * February, 2024: Bug fix with checkpoint loading
 * December, 2023: Released ArTST ASR demo [HF-Spaces](https://huggingface.co/spaces/MBZUAI/artst-demo-asr)
 * November, 2023: Released ArTST TTS demo [HF-Spaces](https://huggingface.co/spaces/MBZUAI/artst-tts-demo)
 * October, 2023: Open-sourced model's weight to HuggingFace
 * October, 2023: ArTST was accepted by EMNLP (ArabicNLP conference) 2023.



## Checkpoints

### Pre-Trained Models

 Model | Pre-train Dataset | Model | Tokenizer |
| --- | --- | --- | --- |
| ArTST base | MGB2 | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/pretrain_checkpoint.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/asr_spm.model)

### Finetuned Models
 Model | FInetune Dataset | Model | Tokenizer |
| --- | --- | --- | --- |
| ArTST ASR | MGB2 | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/MGB2_ASR.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/asr_spm.model)|
| ArTST TTS | ClArTTS | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/CLARTTS_ArTST_TTS.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/tts_spm.model)|
| ArTST* TTS |  ClArTTS | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/CLARTTS_ArTSTstar_TTS.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/tts_spm.model)|


## Environment & Installation

Python version: 3.8+

1) Clone this repo
```bash
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

3) Download Checkpoints
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/MBZUAI/ArTST
```
## Loading Model

```python
import torch
from artst.tasks.artst import ArTSTTask
from artst.models.artst import ArTSTTransformerModel

checkpoint = torch.load('checkpoint.pt')
checkpoint['cfg']['task'].t5_task = 't2s' # or "s2t" for asr
task = ArTSTTask.setup_task(checkpoint['cfg']['task'])

model = ArTSTTransformerModel.build_model(checkpoint['cfg']['model'], task)
model.load_state_dict(checkpoint['model'])
```

## Data Preparation

#### Speech

For pretraining, follow the steps for preparing wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) and preparing HuBERT label [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans).

For finetuning TTS task, an extra column is required in the speech manifest file for speaker embedding. To generate speaker embedding, we use [speech brain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb). 
[Here](./scripts/DATA_ROOT) is a DATA_ROOT sample folder structure that contains manifest samples.

#### Text 

Pretrain:

Please use [fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess) to generate the index and bin files for the text data. We use sentencepiece to pre-process the text, we've provided our SPM models and [dictionary](./scripts/DATA_ROOT/dict.txt) in this repo. You need to use the SPM model to process the text and then use [fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess) with the provided dictionary to get the index and bin files. Note that after SPM processes sentences, the resulting text should have individual characters separated by space.

For Finetuning, a simple text file containing corresponding texts on each line suffices. See [here](./scripts/DATA_ROOT/test.txt) for sample manifest. Normalize the texts as we did for training/evaluation using this [script](./scripts/ASR/normalize_text.py).

## Training

The bash [files](./scripts/) contain the parameters and hyperparameters used for pretraining and finetuning. Find more details on training arguments [here](https://fairseq.readthedocs.io/en/latest/)


### Pretrain

``` bash
bash /scripts/pretrain/train.sh
```

### Finetune

#### ASR

```bash
bash /scripts/ASR/finetune.sh
```

#### TTS

```bash
bash /scripts/TTS/finetune.sh
```

## Inference
#### ASR

```bash
bash /scripts/ASR/inference.sh
```

#### TTS

```bash
bash /scripts/TTS/inference.sh
```

# Acknowledgements

ArTST is built on [SpeechT5](https://arxiv.org/abs/2110.07205) Architecture. If you use any of ArTST models, please cite 

``` 
@inproceedings{toyin2023artst,
  title={ArTST: Arabic Text and Speech Transformer},
  author={Toyin, Hawau and Djanibekov, Amirbek and Kulkarni, Ajinkya and Aldarmaki, Hanan},
  booktitle={Proceedings of ArabicNLP 2023},
  pages={41--51},
  year={2023}
}
```
