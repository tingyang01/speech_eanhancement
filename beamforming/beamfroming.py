import numpy as np
import librosa
import IPython.display as ipd


class beamformer:
    def __init__(self, sample_rate, r):
        self.sample_rate = sample_rate
        self.theta_arr = np.linspace(0, np.pi, 13)[1:-1]
        self.C = 343
        self.r = r
        self.previous_angle = None

    def shift_sig(self, sig, p_num):
        aligned_sig = np.zeros(sig.shape)

        p_num = int(p_num)
        if p_num > 0:
            aligned_sig[:-p_num] = sig[p_num:]
        elif p_num < 0:
            aligned_sig[-p_num:] = sig[:p_num]
        else:
            aligned_sig = sig

        return aligned_sig

    def align_signal(self, sigs, x_M, theta):

        source_x = self.r * np.cos(theta)
        source_y = -self.r * np.sin(theta)

        # pth_diff = dis* np.cos(theta)
        distance = np.sqrt((x_M - source_x) ** 2 + source_y ** 2)  # distance between source and each microphone
        distance_diff = distance - distance[6]  # distance difference to central microphone

        tdoa = distance_diff / self.C
        points_to_shift = np.round(tdoa * self.sample_rate)

        aligned_sigs = []
        for sig, p in zip(sigs, points_to_shift):
            aligned_sigs.append(self.shift_sig(sig, p))

        mse = 0
        l = len(aligned_sigs)
        half_l = l // 2
        weight_sum = 0
        for i in range(half_l):
            weight = 1 - 0.1 * i
            weight_sum += weight
            mse += weight * np.sqrt(np.sum((aligned_sigs[i] - aligned_sigs[-(i + 1)]) ** 2))

        mse = mse / weight_sum
        return aligned_sigs, mse

    def beamforming(self, sigs, x_M):
        best_aligned = None
        best_mse = float("inf")
        best_theta = None
        for theta in self.theta_arr:
            aligned_sigs, mse = self.align_signal(sigs, x_M, theta)

            if mse < best_mse:
                best_aligned = aligned_sigs
                best_mse = mse
                best_theta = theta

        print(best_theta / np.pi * 180)

        beamform_res = np.average(np.array(best_aligned), axis=0)

        return beamform_res


def read_signals(angle):
    sigs = []
    file_name = "mic_audiodata_" + str(angle) + "_"
    for i in range(15):
        f_n = file_name + str(i + 1) + ".wav"
        sig, sr = librosa.load(f_n, sr=16000)
        sigs.append(sig)

    mic_order = [6, 5, 4, 15, 11, 13, 14, 12, 10, 9, 8, 7, 3, 2, 1]
    ordered_sigs = []

    for i in range(15):
        ind = mic_order[i]
        sig = sigs[ind - 1]
        ordered_sigs.append(sig)
    ordered_sigs = np.array(ordered_sigs)

    return ordered_sigs


def main_proc():
    x_M = [-0.56, -0.42, -0.28, -0.14, -0.07, -0.035, 0, 0.035, 0.07, 0.105, 0.14, 0.21, 0.28, 0.42, 0.56]
    x_M = np.array(x_M)
    signal_array = read_signals(120)
    BF = beamformer(16000, 2)

    beamform_res_chunks = []
    chunck_len = 32000
    for i in range(0, signal_array.shape[1] - chunck_len, chunck_len // 2):
        sig_chunk = signal_array[:, i:i + chunck_len]
        beamform_res_chunks += (BF.beamforming(sig_chunk, x_M).tolist()[:chunck_len // 2])

    beamform_res_chunks = np.array(beamform_res_chunks)
    ipd.Audio(beamform_res_chunks, rate=16000)

if __name__ == "__main__":
    main_proc()