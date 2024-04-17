from flask import Flask, request, jsonify
from funasr import AutoModel

import shutil
import json
import uuid
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type = int, default=20067, help="")
args = parser.parse_args()


app = Flask(__name__)
# model = AutoModel( model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++")

model = AutoModel(model="/data/LZY/model/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="/data/LZY/model/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="/data/LZY/model/punc_ct-transformer_cn-en-common-vocab471067-large",
    spk_model="/data/LZY/model/speech_campplus_sv_zh-cn_16k-common")

@app.route('/api/ping')
def ping():
    return jsonify({"message": "pong"})


@app.route('/api/transcriptions', methods=['POST'])
def create_translation():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        filename = str(uuid.uuid4())  # Generate a unique filename

        # Save the uploaded file to a temporary location
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call FunASR for transcription
        transcription = model.generate(input=file_path, batch_size_s=300)

        # Cleanup the temporary file
        os.remove(file_path)

        res = transcription[0]['sentence_info']
        for key in res:
            key.pop('timestamp')

        # return res
        return jsonify({"segments": res})

    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"message": "An error occurred while processing the request"}), 400


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = '/tmp'  # Temporary folder for file storage
    app.run(debug=True, host='0.0.0.0', port=args.port)