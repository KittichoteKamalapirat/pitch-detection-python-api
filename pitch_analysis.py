import torchcrepe

def get_pitch(file_path):
    audio, sr = torchcrepe.load.audio(file_path)

    hop_length = int(sr / 200.)
    fmin = 50
    fmax = 550
    model = 'tiny'
    device = 'cpu'
    # device = 'cuda:0'
    batch_size = 2048

    pitch, confidence = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True
    )

    pitch_list = pitch[0]
    confidence_list = confidence[0]

    return pitch_list, confidence_list
