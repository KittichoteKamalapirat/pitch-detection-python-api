# from fastapi import FastAPI

# app = FastAPI()
print('1')
# audio_path_wav = "./assets/kami-sankaku-native-mono.wav"

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


import torchcrepe

print(2)


# audio, sr = torchcrepe.load.audio("./assets/kami-sankaku-native-mono.wav")
audio, sr = torchcrepe.load.audio("./assets/YIN_pitch_detection_singer.wav")
print(3)
print(f'audio {audio}')
print(f'sr {sr}')

# Here we'll use a 5 millisecond hop length
hop_length = int(sr / 200.)
print(4)
print(f'hop_length {hop_length}')

fmin = 50
fmax = 550
print(5)

model = 'tiny'

print(6)
# device = 'cuda:0'
device = 'cpu'


print(7)
batch_size = 2048

print(8)

pitch = torchcrepe.predict(audio,
                           sr,
                           hop_length,
                           fmin,
                           fmax,
                           model,
                           batch_size=batch_size,
                           device=device)

print(9)
print(pitch)
