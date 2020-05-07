from utils.dataset import findFile, load_edf
from preprocessing import nedc_pystream as ned
from scipy import signal
import matplotlib.pyplot as plt

"""
Playground to check edf properties and qualitatively look at data
"""
# check for sampling frequencies and visualize stft to see what noise frequency to remove
tcp_ar_params = ned.nedc_load_parameters('../preprocessing/parameter_files/params_01_tcp_ar.txt')
tcp_le_params = ned.nedc_load_parameters('../preprocessing/parameter_files/params_02_tcp_le.txt')
tcp_ar_a_params = ned.nedc_load_parameters('../preprocessing/parameter_files/params_03_tcp_ar_a.txt')
edf_files = findFile('../data/edf/train', '.edf')
for edf_path in edf_files[100:]:
    print(edf_path)
    if '01_tcp_ar' in edf_path:
        freq, edf, _ = load_edf(tcp_ar_params, edf_path)
    elif '02_tcp_le' in edf_path:
        freq, edf, _ = load_edf(tcp_le_params, edf_path)
    else:
        freq, edf, _ = load_edf(tcp_ar_a_params, edf_path)
    montage_0_freq = freq[0]

    # do stft for 30s
    f, t, stft = signal.stft(edf[0][:montage_0_freq*30], fs=montage_0_freq, nperseg=montage_0_freq)
    f2, t2, stft2 = signal.stft(edf[5][:montage_0_freq*30], fs=montage_0_freq, nperseg=montage_0_freq)

    # visualize
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(abs(stft)[1:126][:])
    fig.add_subplot(1, 2, 2)
    plt.imshow(abs(stft2)[1:126][:])
    plt.colorbar()
    plt.show()

    # check that a single edf file all has the same frequencies, THIS IS TRUE
    # for montage_n_freq in freq[1:]:
    #     assert montage_0_freq == montage_n_freq