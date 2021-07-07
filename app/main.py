from flask import Flask,request,jsonify
import torch
import sys
import os
import urllib.request
import torchaudio

app= Flask(__name__)

PATH = "app/w2v_pretrain915_finetuned_quantized.pt"
model = torch.jit.load(PATH)

def start_asr(url):
    print(url)
    audio , _ = torchaudio.load(url)

    print(os.path.isfile(PATH))
 
    text = model(audio)
    print(text)
    return text

@app.route('/v1/transcript', methods=['GET'])
def index():
    url = request.args.get('url')
    file_name, headers = urllib.request.urlretrieve(url)
    file_type = headers.get('Content-Type')
    try:
        if file_type == "audio/wav":
            text = start_asr(file_name)
            return jsonify({'status': 1,
                    'result': {
                      'transcription':text
                    }})

        else:
            return jsonify({'status': 0,
                    'message':"Type must be audio/wav"})
    except:
        return jsonify({'error': 'error during prediction'})



