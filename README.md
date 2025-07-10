<div align="center">

<h1> ArTST </h1>
This repository contains the implementation of the paper:

**ArTST**: **Ar**abic **T**ext and **S**peech **T**ransformer

<a href=''> <a href='https://aclanthology.org/2023.arabicnlp-1.5/'><img src='https://img.shields.io/badge/paper-Paper(v1)-red'></a> &nbsp; <a href='https://arxiv.org/abs/2411.05872'><img src='https://img.shields.io/badge/paper-Paper(v2 & v3)-red'></a> &nbsp;  <a href='https://artstts.wixsite.com/artsttts'><img src='https://img.shields.io/badge/project-Page-green'></a> &nbsp; <a href='https://huggingface.co/spaces/MBZUAI/artst-tts-demo'><img src='https://img.shields.io/badge/demo-Spaces-yellow'></a> &nbsp;

<div>
    <a href='https://www.linkedin.com/in/toyinhawau/' >Hawau Olamide Toyin <sup>* 1</sup> </a>&emsp;
    <a href='https://www.linkedin.com/in/amir-djanibekov/' target='_blank'>Amirbek Djanibekov <sup>* 1</a>&emsp;
    <a href='https://www.linkedin.com/in/ajinkya-kulkarni-32b80a130/' target='_blank'>Ajinkya Kulkarni <sup>1</a>&emsp;
    <a href='https://www.linkedin.com/in/raghadalshalan/' target='_blank'>Raghad Alshalan <sup>2</a>&emsp;
    <a href='' target='_blank'>Abdullah Alitr <sup>2</a>&emsp;
    <a href='https://linkedin.com/in/hanan-aldarmaki/' target='_blank'>Hanan Aldarmaki <sup>1</a>&emsp;
</div>
<br>
<div>
    <sup>*</sup> equal contribution &emsp; <sup>1</sup> MBZUAI &emsp; <sup>2</sup> STC &emsp;
</div>
<br>
<i><strong><a href='https://aclanthology.org/2023.arabicnlp-1.5/' target='_blank'>ArabicNLP 2023</a></strong></i>, &nbsp;
<i><strong><a href='' target='_blank'>ACL 2025</a></strong></i>
<br>
</div>

## ArTST 
ArTST, a pre-trained Arabic text and speech transformer for supporting open-source speech technologies for the Arabic language. The model architecture in this first edition follows the unified-modal framework, SpeechT5, that was recently released for English, and is focused on Modern Standard Arabic (MSA), with plans to extend the model for dialectal and code-switched Arabic in future editions. We pre-trained the model from scratch on MSA speech and text data, and fine-tuned it for the following tasks: Automatic Speech Recognition (ASR), Text-To-Speech synthesis (TTS), and spoken dialect identification. 


### Latest Highlights from the ArTST
* **June 2024** — *ArTSTv1.5: MSA pre-training with diacritics for Arabic*.
  Pre-trained ArTST from scratch on MGB2 and Tashkeela dataset.

* **May 2024** — *ArTSTv2 and ArTSTv3 make it to ACL (main conference)*.
  Our paper, *"Dialectal Coverage and Generalization in Arabic Speech Recognition"*, was officially accepted at ACL 2025. A significant milestone for dialectal Arabic ASR research.

* **April 2025** — *A milestone in dialectal ASR: 17 fine-tuned checkpoints released*.
  We’ve expanded ArTSTv2’s reach by publishing 17 fine-tuned models across Arabic dialects. Explore them all on [HuggingFace](https://huggingface.co/MBZUAI/ArTSTv2/tree/main/dialectal_checkpoints).

* **April 2025** — *ArTSTv3 goes live on HuggingFace*.
  ArTSTv3 ASR model cards are now available: [MGB2 version](https://huggingface.co/MBZUAI/artst_asr_v3) and [QASR version](https://huggingface.co/MBZUAI/artst_asr_v3_qasr).

* **March 2025** — *ArTSTv3: Multilingual pre-training for Arabic + English, French, Spanish*.
  A major upgrade for ArTST. ArTSTV3 now spans Arabic dialects and adds multilingual support. Available on [HuggingFace](https://huggingface.co/MBZUAI/ArTSTv3).

* **December 2024** — *Fine-tuning made easy*.
  Released a practical notebook for fine-tuning with Hugging Face Trainer—run it on [Google Colab](https://drive.google.com/file/d/1Tp-6BgjmbVh-sh_Oro1xlWhYJm5_8Sfl/view?usp=sharing).

* **October 2024** — *ArTSTv2 released with HuggingFace integration*.
  ArTSTv2 is now live along with model cards: [ASR v2 (MGB2)](https://huggingface.co/MBZUAI/artst_asr_v2) and [ASR v2 (QASR)](https://huggingface.co/MBZUAI/artst_asr_v2_qasr)

* **October 2024** — *ArTSTv2: Dialectal pre-training for Arabic*.
  Pre-trained ArTST from scratch on 17 Arabic dialects

* **October 2024** — *ArTSTv1 joins the HuggingFace*.
  The first version of our ASR model has been made available here: [ASR v1](https://huggingface.co/MBZUAI/artst_asr)

* **February 2024** - *Bug fix*. Addressed key checkpoint-loading issues

* **February 2024** — *TTS support launched*.
  We’ve released ArTST TTS for Arabic via HuggingFace's Transformers: [TTS model](https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar).

* **December 2023** — *Speech-to-Text (ASR) demo now on HuggingFace Spaces*.
  Try the ArTST ASR model in real-time: [Demo here](https://huggingface.co/spaces/MBZUAI/artst-demo-asr)

* **November 2023** — *Text-to-Speech (TTS) demo released*.
  Experience our TTS model in action: [Demo here](https://huggingface.co/spaces/MBZUAI/artst-tts-demo)

* **October 2023** — *ArTST goes open-source*.
  Model weights are now publicly accessible on [HuggingFace](https://huggingface.co/MBZUAI/ArTST).

* **October 2023** — *ArTST recognized at EMNLP 2023*.
  Our work was accepted at the ArabicNLP workshop at EMNLP 2023.




## Checkpoints

### Pre-Trained Models

 Model | Pre-train Dataset | Model | Tokenizer |
| --- | --- | --- | --- |
| ArTST v1 base | MGB2 | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/pretrain_checkpoint.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/asr_spm.model)
| ArTST v1.5 base | MGB2 + Tashkeela| [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv1.5/blob/main/%5Bfairseq%5Dcheckpoint_best.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv1.5/blob/main/sp_model.model)
| ArTST v2 base | Dialects | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv2/blob/main/pretrain_checkpoint.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv2/blob/main/tokenizer_artstv2.model)
| ArTST v3 base | Multilingual | [HuggingFace](https://huggingface.co/MBZUAI/ArTSTv3/blob/main/pretrain_checkpoint.pt) | [HuggingFace](https://huggingface.co/MBZUAI/ArTSTv3/blob/main/tokenizer_artstv3.model) 

### Finetuned Models
 Model | FInetune Dataset | Model | Tokenizer |
| --- | --- | --- | --- |
| ArTST v1 ASR | MGB2 | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/MGB2_ASR.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/asr_spm.model)|
| ArTST v1 TTS | ClArTTS | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/CLARTTS_ArTST_TTS.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/tts_spm.model)|
| ArTST* TTS |  ClArTTS | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/CLARTTS_ArTSTstar_TTS.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTST/blob/main/tts_spm.model)|
| ArTST v2 ASR | QASR | [Hugging Face - safetenors](https://huggingface.co/MBZUAI/artst_asr_v2/blob/main/model.safetensors) | [Hugging Face](https://huggingface.co/MBZUAI/artst-v2-asr/blob/main/spm_char.model) |
| ArTST v2 ASR | MGB2 | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv2/blob/main/ASR_MGB2_best.pt_hf.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv2/blob/main/tokenizer_artstv2.model) |
| ArTST v2 ASR | QASR | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv2/blob/main/ASR_QASR_best.pt_hf.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv2/blob/main/tokenizer_artstv2.model) |
| ArTST v2 ASR | Dialects | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv2/blob/main/ASR_Dialects_MGB2_best.pt_hf.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv2/blob/main/tokenizer_artstv2.model) |
| ArTST v3 ASR | MGB2 | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv3/blob/main/%5Bfairseq%5Dmgb2_checkpoint_best.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv3/blob/main/tokenizer_artstv3.model) |
| ArTST v3 ASR | QASR | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv3/blob/main/%5Bfairseq%5Dqasr_checkpoint_best.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv3/blob/main/tokenizer_artstv3.model) |
| ArTST v3 ASR | Mutlilingual | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv3/resolve/main/%5Bfairseq%5Djoint_multilingual.pt) | [Hugging Face](https://huggingface.co/MBZUAI/ArTSTv3/blob/main/tokenizer_artstv3.model) |

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
### With HuggingFace Transformers
Transformers version: 4.46.3
```python
from transformers import (
    SpeechT5ForSpeechToText,
    SpeechT5Processor,
    SpeechT5Tokenizer,
)

device = "cuda" if torch.cuda.is_available() else "CPU"

model_id = "mbzuai/artst-v2-asr" # or "mbzuai/artst_asr" for v1

tokenizer = SpeechT5Tokenizer.from_pretrained(model_id)
processor = SpeechT5Processor.from_pretrained(model_id , tokenizer=tokenizer)
model = SpeechT5ForSpeechToText.from_pretrained(model_id).to(device)
```

### With Fairseq
```python
import torch
from artst.tasks.artst import ArTSTTask
from artst.models.artst import ArTSTTransformerModel

checkpoint = torch.load('checkpoint.pt')
checkpoint['cfg']['task'].t5_task = 't2s' # or "s2t" for asr
checkpoint['cfg']['task'].data = 'path-to-folder-with-checkpoints'
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

@misc{djanibekov2025dialectalcoveragegeneralizationarabic,
      title={Dialectal Coverage And Generalization in Arabic Speech Recognition}, 
      author={Amirbek Djanibekov and Hawau Olamide Toyin and Raghad Alshalan and Abdullah Alitr and Hanan Aldarmaki},
      year={2025},
      eprint={2411.05872},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.05872}, 
}
```
