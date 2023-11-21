import os
import torch
import gradio as gr
import os.path as op

from artst.tasks.artst import ArTSTTask
from transformers import SpeechT5HifiGan
from artst.models.artst import ArTSTTransformerModel
from fairseq.tasks.hubert_pretraining import LabelEncoder
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORK_DIR = os.getcwd()
checkpoint = torch.load('/l/users/speech_lab/_SpeechT5PretrainDataset/Finetune/TTS/models/CLARTTS/checkpoint_last.pt')
checkpoint['cfg']['task'].t5_task = 't2s'
task = ArTSTTask.setup_task(checkpoint['cfg']['task'])

emb_path='/l/users/speech_lab/_SpeechT5PretrainDataset/v2/data/CLARTTS/CLARTTS/speaker_embedding/CLARTTS_speaker_embedding.npy'
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
    return processor(tokenizer.encode(text)).reshape(1, -1)

net_input = {}

def inference(text, spkr=emb_path):
    net_input['src_tokens'] = process_text(text)
    net_input['spkembs'] = get_embs(spkr)
    outs, _, attn = task.generate_speech(
            [model], 
            net_input,
        )
    with torch.no_grad():
        gen_audio = vocoder(outs.to(device))
    return (16000,gen_audio.cpu().numpy())



text_box = gr.Textbox(max_lines=2, label="Arabic Text")
out = gr.Audio(label="Synthesized Audio", type="numpy")
demo = gr.Interface(inference, \
    inputs=text_box, outputs=out, title="ArTST")

if __name__ == "__main__":
    demo.launch(share=True)
