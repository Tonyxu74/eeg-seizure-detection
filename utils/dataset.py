from torch.utils import data
import torch
import os
from myargs import args
from preprocessing import nedc_pystream as ned
from scipy import signal
import numpy as np


def findFile(root_dir, contains):
    """
    Finds file with given root directory containing keyword "contains"
    :param root_dir: root directory to search in
    :param contains: the keyword that should be contained
    :return: a list of the file paths of found files
    """

    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if contains in file:
                all_files.append(os.path.join(path, file))

    return all_files


def load_edf(params, edf_path, start, read_len):
    """Loads an EDF file given a path to the EDF file as well as the specifying parameter file path."""
    # loads the Edf into memory
    fsamp, sig, labels = ned.nedc_load_edf(edf_path, start, read_len)

    # select channels from parameter file
    fsamp_sel, sig_sel, labels_sel = ned.nedc_select_channels(params, fsamp, sig, labels)

    # apply a montage
    fsamp_mont, sig_mont, labels_mont = ned.nedc_apply_montage(
        params, fsamp_sel, sig_sel, labels_sel
    )

    # print the values to stdout
    return fsamp_mont, sig_mont, labels_mont


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, datapath, parampath, keep, eval):
        """
        Initializes dataset for given author and datapath
        :param datapath: path to data
        :param parampath: path to parameter files
        :param keep: the array of channels to keep
        :param eval: whether to use val or train set
        """

        self.eval = eval
        self.keep_channels = keep
        self.std = 0

        # get all edf file paths
        if not self.eval:
            edf_files = findFile(datapath + '/edf/train', '.edf')
            label_file = open(datapath + '/_DOCS/ref_train.txt', 'r')

        else:
            edf_files = findFile(datapath + '/edf/dev', '.edf')
            label_file = open(datapath + '/_DOCS/ref_dev.txt', 'r')

        # create a label dictionary for all file paths, every label is added as a array
        label_dict = {}

        for line in label_file:
            parts = line.split(' ')

            # parts[0]: name of the file
            # parts[1]: start time
            # parts[2]: the end time
            # parts[3]: label
            if parts[0] not in label_dict:
                label_dict[parts[0]] = {}
                label_dict[parts[0]]['event_time'] = [(float(parts[1]), float(parts[2]))]
                label_dict[parts[0]]['label'] = [0] if parts[3] == 'bckg' else [1]

            else:
                label_dict[parts[0]]['event_time'].append((float(parts[1]), float(parts[2])))
                label_dict[parts[0]]['label'].append(0 if parts[3] == 'bckg' else 1)

        # actual datalist after split into window_len intervals
        self.datalist = []
        for file in edf_files:
            # get filename and the dictionary of labels and event times
            file_name = file.split('\\')[-1].replace('.edf', '')
            file_dict = label_dict[file_name]

            # get length of the recording of current edf file
            recording_length = int(file_dict['event_time'][-1][-1])
            window_start = 0

            # special case where recording length is less than window length
            if recording_length < args.window_len:
                # though unlikely for such a short snippet, still check for seizure/not-seizure times
                seizure_time = 0
                for (start, end), label in zip(file_dict['event_time'], file_dict['label']):
                    if label == 1:
                        seizure_time += start - end

                window_label = int(seizure_time / args.window_len >= args.seiz_sens)

                self.datalist.append({
                    'filepath': file, 'start_time': window_start, 'label': window_label, 'short': True
                })

            # get all windows and labels of windows
            while window_start + args.window_len <= recording_length:
                # get end of window
                window_end = window_start + args.window_len

                # time where label is 1
                seizure_time = 0

                # get step size if label is zero
                next_step_size = args.window_len - args.label_0_overlap

                # find seizure time for the window selected
                for (start, end), label in zip(file_dict['event_time'], file_dict['label']):
                    if label == 1:
                        # if not at window yet:
                        if end < window_start:
                            continue

                        # if past window:
                        elif window_end < start:
                            break

                        # if window is completely enclosed in single label, more positive examples coming, reduce step
                        elif end >= window_end and start <= window_start:
                            seizure_time += window_end - window_start
                            next_step_size = args.window_len - args.label_1_overlap

                        # if window completely encloses a label
                        elif end < window_end and start > window_start:
                            seizure_time += end - start

                        # if start before window and end in window
                        elif end < window_end and start <= window_start:
                            seizure_time += end - window_start

                        # if start in window and end after window, more positive examples coming, reduce step
                        elif end >= window_end and start > window_start:
                            seizure_time += window_end - start
                            next_step_size = args.window_len - args.label_1_overlap

                # window label is 1 if the percent of seizure time is greater than required sensitivity
                window_label = int(seizure_time / args.window_len >= args.seiz_sens)

                # add datapoint
                self.datalist.append({
                    'filepath': file, 'start_time': window_start, 'label': window_label, 'short': False
                })

                # continue to next window
                window_start += next_step_size

        # find distribution of positive examples
        numpos = 0
        for item in self.datalist:
            numpos += item['label']

        print(
            f"{'train' if not self.eval else 'val'} || "
            f"positive examples: {numpos} || "
            f"total examples: {len(self.datalist)} || "
            f"percent positive: {numpos/len(self.datalist)}"
        )

        # get parameters for the electrode configurations
        self.tcp_ar_params = ned.nedc_load_parameters(parampath + '/params_01_tcp_ar.txt')
        self.tcp_le_params = ned.nedc_load_parameters(parampath + '/params_02_tcp_le.txt')
        self.tcp_ar_a_params = ned.nedc_load_parameters(parampath + '/params_03_tcp_ar_a.txt')

    def __len__(self):
        """
        Denotes the total number of samples
        :return: length of the dataset
        """

        return len(self.datalist)

    def edf_to_tensor(self, edf_data, freq):
        """
        Performs STFT on given edf data, appends montages, and converts it to tensor
        :param edf_data: edf data
        :param start: start time of the window
        :param freq: frequency of sampling edf data
        :return: tensor format
        """

        if self.keep_channels is not None:
            edf_data = [edf_data[i] for i in self.keep_channels]
            freq = [freq[i] for i in self.keep_channels]

        # sampling frequency
        sample_freq = freq[0]

        # ensure all sample frequency for each montage is equal
        assert all([freq[i] == sample_freq for i in range(1, len(freq))])

        # special case for very short recordings
        if len(edf_data[0]) < sample_freq * args.window_len:
            # append zeros at end of array to make it the proper length
            zeros_arr = np.zeros(sample_freq * args.window_len - len(edf_data[0]), dtype=np.float32)
            edf_data = [np.concatenate((data, zeros_arr)) for data in edf_data]

        # stft each montage
        stft_data = []
        for data in edf_data:
            f, t, stft = signal.stft(data, fs=sample_freq, nperseg=sample_freq)
            stft_data.append(abs(stft))

        # filter out 0, 57 - 63 and 117 to 123 Hz
        stft_data = [
            np.concatenate((
                stft_item[1:57],
                stft_item[64: 117],
                stft_item[124:126]
            ), axis=0) for stft_item in stft_data
        ]

        # convert to tensor and return
        stft_data = np.asarray(stft_data)
        tensor = torch.from_numpy(stft_data).float()

        # data augmentation here?
        if not self.eval:
            noise = torch.from_numpy(np.random.normal(
                0, self.std, size=tensor.shape)).type(torch.FloatTensor)
            tensor += noise

        return tensor

    def __getitem__(self, index):
        """
        Generates one sample of data
        :param index: index of the data in the datalist
        :return: returns the data and label in float tensor and long tensor respectively
        """

        edf_path = self.datalist[index]['filepath']
        window_start = self.datalist[index]['start_time']
        label = self.datalist[index]['label']

        # check sampling frequency
        samp_freq = ned.nedc_get_fs(edf_path)

        # get start and length of data
        read_start = window_start * samp_freq
        read_len = args.window_len * samp_freq
        if self.datalist[index]['short']:
            read_start = 0
            read_len = None

        # check which electrode setup is used, labels of montage are not important
        # cut out montages 8 and 13 here
        if '01_tcp_ar' in edf_path:
            freq, edf, _ = load_edf(self.tcp_ar_params, edf_path, start=read_start, read_len=read_len)
            del freq[8]
            del freq[12]
            del edf[8]
            del edf[12]
        elif '02_tcp_le' in edf_path:
            freq, edf, _ = load_edf(self.tcp_le_params, edf_path, start=read_start, read_len=read_len)
            del freq[8]
            del freq[12]
            del edf[8]
            del edf[12]
        else:
            freq, edf, _ = load_edf(self.tcp_ar_a_params, edf_path, start=read_start, read_len=read_len)

        # cut out the correct window segment from each montage, STFT, data augment, append, convert it into a tensor
        output = self.edf_to_tensor(edf, freq)

        return output, label


def GenerateIterator(datapath, parampath, keep=None, eval=False, shuffle=True):
    """
    Creates iterator object
    :param datapath: path to data
    :param parampath: path to parameter files
    :param eval: whether to evaluate or train
    :param shuffle: shuffle data randomly or not
    :return: iterator object
    """

    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(datapath=datapath, parampath=parampath, keep=keep, eval=eval), **params)


class Dataset_eval(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, datapath, parampath, keep, val):
        """
        Initializes dataset for given author and datapath
        :param datapath: path to data
        :param parampath: path to parameter files
        :param keep: the array of channels to keep
        :param val: whether to evaluate the validation set
        """

        self.keep_channels = keep
        self.validate = val

        # get all edf file paths
        if not self.validate:
            edf_files = findFile(datapath + '/edf/eval', '.edf')

        else:
            edf_files = findFile(datapath + '/edf/dev', '.edf')

        # actual datalist after split into window_len intervals
        self.datalist = []
        self.anslist = {}

        # process all files
        for file in edf_files:

            # get length and samp frequency
            edf_len = ned.nedc_get_len(file)
            samp_freq = ned.nedc_get_fs(file)

            recording_length = int(edf_len / samp_freq)
            window_start = 0

            # get one array for each file
            ans_arr = np.zeros(shape=(recording_length), dtype=np.float32)
            avg_arr = np.zeros(shape=(recording_length), dtype=np.uint8)
            self.anslist[file] = {'pred': ans_arr, 'counter': avg_arr}

            # special case where recording length is less than window length
            if recording_length < args.window_len:
                self.datalist.append({
                    'filepath': file, 'start_time': window_start, 'short': True
                })

            # get all windows and labels of windows
            while window_start + args.window_len <= recording_length:

                # add datapoint
                self.datalist.append({
                    'filepath': file, 'start_time': window_start, 'short': False
                })

                # continue to next window
                window_start += args.window_len - args.eval_overlap

        # get parameters for the electrode configurations
        self.tcp_ar_params = ned.nedc_load_parameters(parampath + '/params_01_tcp_ar.txt')
        self.tcp_le_params = ned.nedc_load_parameters(parampath + '/params_02_tcp_le.txt')
        self.tcp_ar_a_params = ned.nedc_load_parameters(parampath + '/params_03_tcp_ar_a.txt')

        print(f'length of dataset is {len(self.datalist)}')

    def get_anslist(self):
        return self.anslist

    def __len__(self):
        """
        Denotes the total number of samples
        :return: length of the dataset
        """

        return len(self.datalist)

    def edf_to_tensor(self, edf_data, freq):
        """
        Performs STFT on given edf data, appends montages, and converts it to tensor
        :param edf_data: edf data
        :param start: start time of the window
        :param freq: frequency of sampling edf data
        :return: tensor format
        """

        if self.keep_channels is not None:
            edf_data = [edf_data[i] for i in self.keep_channels]
            freq = [freq[i] for i in self.keep_channels]

        # sampling frequency
        sample_freq = freq[0]

        # ensure all sample frequency for each montage is equal
        assert all([freq[i] == sample_freq for i in range(1, len(freq))])

        # special case for very short recordings
        if len(edf_data[0]) < sample_freq * args.window_len:
            # append zeros at end of array to make it the proper length
            zeros_arr = np.zeros(sample_freq * args.window_len - len(edf_data[0]), dtype=np.float32)
            edf_data = [np.concatenate((data, zeros_arr)) for data in edf_data]

        # stft each montage
        stft_data = []
        for data in edf_data:
            f, t, stft = signal.stft(data, fs=sample_freq, nperseg=sample_freq)
            stft_data.append(abs(stft))

        # filter out 0, 57 - 63 and 117 to 123 Hz
        stft_data = [
            np.concatenate((
                stft_item[1:57],
                stft_item[64: 117],
                stft_item[124:126]
            ), axis=0) for stft_item in stft_data
        ]

        # convert to tensor and return
        stft_data = np.asarray(stft_data)
        tensor = torch.from_numpy(stft_data).float()

        return tensor

    def __getitem__(self, index):
        """
        Generates one sample of data
        :param index: index of the data in the datalist
        :return: returns the data and label in float tensor and long tensor respectively
        """

        edf_path = self.datalist[index]['filepath']
        window_start = self.datalist[index]['start_time']

        # check sampling frequency
        samp_freq = ned.nedc_get_fs(edf_path)

        # get start and length of data
        read_start = window_start * samp_freq
        read_len = args.window_len * samp_freq
        if self.datalist[index]['short']:
            read_start = 0
            read_len = None

        # check which electrode setup is used, labels of montage are not important
        # cut out montages 8 and 13 here
        if '01_tcp_ar' in edf_path:
            freq, edf, _ = load_edf(self.tcp_ar_params, edf_path, start=read_start, read_len=read_len)
            del freq[8]
            del freq[12]
            del edf[8]
            del edf[12]
        elif '02_tcp_le' in edf_path:
            freq, edf, _ = load_edf(self.tcp_le_params, edf_path, start=read_start, read_len=read_len)
            del freq[8]
            del freq[12]
            del edf[8]
            del edf[12]
        else:
            freq, edf, _ = load_edf(self.tcp_ar_a_params, edf_path, start=read_start, read_len=read_len)

        # cut out the correct window segment from each montage, STFT, data augment, append, convert it into a tensor
        output = self.edf_to_tensor(edf, freq)

        return output, window_start, edf_path


def GenerateIterator_eval(datapath, parampath, keep=None, val=False, shuffle=False):
    """
    Creates iterator object
    :param datapath: path to data
    :param parampath: path to parameter files
    :param shuffle: shuffle data randomly or not
    :return: iterator object
    """

    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset_eval(datapath=datapath, parampath=parampath, keep=keep, val=val), **params)


# check that the data is loading properly
# iter = GenerateIterator('../data/edf', shuffle=False, eval=True)
#
# for stft, label in iter:
#     print(stft.shape, label)
#
#     a = stft[0][0].numpy()
#     b = stft[0][5].numpy()
#
#     fig = plt.figure()
#     fig.add_subplot(1, 2, 1)
#     plt.imshow(abs(a))
#     fig.add_subplot(1, 2, 2)
#     plt.imshow(abs(b))
#     plt.colorbar()
#     plt.show()
#
#     break

# iter = GenerateIterator_eval('../data', '../preprocessing/parameter_files')