import os
from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import uuid
import argparse
import tempfile
from funasr import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=20067, help="")
args = parser.parse_args()

app = FastAPI()

model = AutoModel(
    model="/data/LZY/model/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="/data/LZY/model/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="/data/LZY/model/punc_ct-transformer_cn-en-common-vocab471067-large",
    spk_model="/data/LZY/model/speech_campplus_sv_zh-cn_16k-common"
)

@app.post('/api/ping')
def ping():
    return {"message": "pong"}

async def process_file(file: UploadFile):
    try:
        # Generate a unique filename
        filename = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, filename)

            # Save the uploaded file to the temporary directory
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            transcription = model.generate(input=file_path, batch_size_s=300)

            res = transcription[0]['sentence_info']
            for key in res:
                key.pop('timestamp')

        return {"segments": res}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while processing the file: {str(e)}")


@app.post("/api/transcriptions")
async def create_transcription(file: UploadFile = File(...)):
    return await process_file(file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
