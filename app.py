import os
import torch
import gradio as gr
import numpy as np
import os.path as op
import pyarabic.araby as araby

from artst.tasks.artst import ArTSTTask
from transformers import SpeechT5HifiGan
from artst.models.artst import ArTSTTransformerModel
from fairseq.tasks.hubert_pretraining import LabelEncoder
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('ckpts/clartts_tts.pt')
checkpoint['cfg']['task'].t5_task = 't2s'
checkpoint['cfg']['task'].bpe_tokenizer = "utils/arabic.model"
checkpoint['cfg']['task'].data = "utils/"
checkpoint['cfg']['model'].mask_prob = 0.5
checkpoint['cfg']['task'].mask_prob = 0.5
task = ArTSTTask.setup_task(checkpoint['cfg']['task'])

emb_path='embs/clartts.npy'
model = ArTSTTransformerModel.build_model(checkpoint['cfg']['model'], task)
model.load_state_dict(checkpoint['model'])

checkpoint['cfg']['task'].bpe_tokenizer = task.build_bpe(checkpoint['cfg']['model'])
tokenizer = checkpoint['cfg']['task'].bpe_tokenizer

processor = LabelEncoder(task.dicts['text'])

vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(device)

def get_embs(emb_path):
    spkembs = get_features_or_waveform(emb_path)
    spkembs = torch.from_numpy(spkembs).float().unsqueeze(0)
    return spkembs

def process_text(text):
    text = araby.strip_diacritics(text)
    return processor(tokenizer.encode(text)).reshape(1, -1)

net_input = {}

def inference(text, spkr=emb_path):
    if len(text.strip()) == 0:
        return (16000, np.zeros(0).astype(np.int16))
    net_input['src_tokens'] = process_text(text)
    net_input['spkembs'] = get_embs(spkr)
    outs, _, attn = task.generate_speech(
            [model], 
            net_input,
        )
    with torch.no_grad():
        gen_audio = vocoder(outs.to(device))
    speech = (gen_audio.cpu().numpy() * 32767).astype(np.int16)
    return (16000,speech)

text_box = gr.Textbox(max_lines=2, label="Arabic Text", rtl=True)
out = gr.Audio(label="Synthesized Audio", type="numpy")
title="ArTST: Arabic Speech Synthesis"
description="ArTST: Arabic text and speech transformer based on the T5 transformer. This space demonstarates the TTS checkpoint finetuned on \
    the Classical Arabic Text-To-Speech (CLARTTS) dataset. The model is pre-trained on the MGB-2 dataset."

examples=["لأن فراق المألوف في العادة ومجانبة ما صار متفقا عليه بالمواضعة",\
    "ومن لطيف حكمته أن جعل لكل عبادة حالتين",\
    "فمن لهم عدل الإنسان مع من فوقه"]

article = """
<div style='margin:20px auto;'>
<p>References: <a href="https://arxiv.org/abs/2310.16621">ArTST paper</a> |
<a href="https://github.com/mbzuai-nlp/ArTST">GitHub</a> |
<a href="https://huggingface.co/MBZUAI/ArTST">Weights and Tokenizer</a></p>
<pre>
@misc{toyin2023artst,
      title={ArTST: Arabic Text and Speech Transformer}, 
      author={Hawau Olamide Toyin and Amirbek Djanibekov and Ajinkya Kulkarni and Hanan Aldarmaki},
      year={2023},
      eprint={2310.16621},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
</pre>
<p>Speaker embeddings were generated from <a href="http://www.festvox.org/cmu_arctic/">CMU ARCTIC</a>.</p>
<p>ArTST is based on <a href="https://arxiv.org/abs/2110.07205">SpeechT5 architecture</a>.</p>
</div>
"""

demo = gr.Interface(inference, \
    inputs=text_box, outputs=out, title=title, description=description, examples=examples, article=article)

if __name__ == "__main__":
    demo.launch(share=True)
