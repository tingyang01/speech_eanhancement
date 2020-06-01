import librosa
import numpy as np
import IPython.display as ipd

NFFT = 768
HOP_LEN= 384

ground_truth, _ = librosa.load("groundtruth.wav", sr=16000) #pure near-end
mic, _ = librosa.load("mic.wav", sr=16000)    #mix near-end and far-end
speaker, _ = librosa.load("speaker.wav", sr=16000) #far-end (echo)

out = []
cur_frame_mic = np.array([0.0] * (3 * HOP_LEN))
cur_frame_spk = np.array([0.0] * (3 * HOP_LEN))
cur_frame_truth = np.array([0.0] * (3 * HOP_LEN))

for i in range(0, mic.shape[0] - NFFT, HOP_LEN):
    cur_frame_mic[:HOP_LEN] = cur_frame_mic[NFFT:3 * HOP_LEN]
    cur_frame_mic[HOP_LEN:3 * HOP_LEN] = mic[i: i + NFFT]
    cur_stft_mic = librosa.stft(cur_frame_mic, n_fft=NFFT, hop_length=HOP_LEN, center=False)
    debug_pcm = librosa.istft(cur_stft_mic, hop_length=HOP_LEN, win_length=NFFT, center=False)

    cur_frame_spk[:HOP_LEN] = cur_frame_spk[NFFT:3 * HOP_LEN]
    cur_frame_spk[HOP_LEN:3 * HOP_LEN] = speaker[i: i + NFFT]
    cur_stft_spk = librosa.stft(cur_frame_spk, n_fft=NFFT, hop_length=HOP_LEN, center=False)

    mic_mag, mic_phase = librosa.magphase(cur_stft_mic)
    speaker_mag, _ = librosa.magphase(cur_stft_spk)
    concat_mag = np.concatenate((mic_mag, speaker_mag), axis=0)

    cur_frame_truth[:HOP_LEN] = cur_frame_truth[NFFT:3 * HOP_LEN]
    cur_frame_truth[HOP_LEN:3 * HOP_LEN] = ground_truth[i: i + NFFT]
    cur_stft_truth = librosa.stft(cur_frame_truth, n_fft=NFFT, hop_length=HOP_LEN, center=False)
    clean_mag, _ = librosa.magphase(cur_stft_truth)
    mask = np.sqrt(clean_mag ** 2 / (clean_mag ** 2 + speaker_mag ** 2))

    recovered_clean_mag = mic_mag * mask  # elementwise (F, T)

    pred_clean_near = librosa.istft(recovered_clean_mag * mic_phase, hop_length=HOP_LEN, win_length=NFFT, center=False)

    out += pred_clean_near[HOP_LEN:NFFT].tolist()

out_array = np.array(out)
ipd.Audio(out_array, rate = 16000)
librosa.output.write_wav("python_recovered_audio.wav", out_array, 16000)