from torch.utils import data
import os
from myargs import args
from preprocessing import nedc_pystream as ned


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


def load_edf(params, edf_path):
    """Loads an EDF file given a path to the EDF file as well as the specifying parameter file path."""
    # loads the Edf into memory
    fsamp, sig, labels = ned.nedc_load_edf(edf_path)

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

    def __init__(self, datapath, eval):
        """
        Initializes dataset for given author and datapath
        :param datapath: path to data
        :param eval: whether to use val or train set
        """

        self.eval = eval

        # get all edf file paths
        if not self.eval:
            edf_files = findFile(datapath + '/train', '.edf')
            label_file = open('../data/_DOCS/ref_train.txt', 'r')

        else:
            edf_files = findFile(datapath + '/dev', '.edf')
            label_file = open('../data/_DOCS/ref_dev.txt', 'r')

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

            while window_start + args.window_len <= recording_length:
                # get end of window
                window_end = window_start + args.window_len

                # time where label is 1
                seizure_time = 0
                for (start, end), label in zip(file_dict['event_time'], file_dict['label']):
                    if label == 1:
                        # if window is` completely enclosed in single label
                        if end >= window_end and start <= window_start:
                            seizure_time = window_end - window_start
                            break

                        # if window completely encloses a label
                        elif end < window_end and start > window_start:
                            seizure_time += end - start

                        # if start before window and end in window
                        elif end < window_end and start <= window_start:
                            seizure_time += end - window_start

                        # if start in window and end after window
                        elif end >= window_end and start > window_start:
                            seizure_time += window_end - start

                # window label is 1 if the percent of seizure time is greater than required sensitivity
                window_label = int(seizure_time / args.window_len >= args.seiz_sens)

                # add datapoint
                self.datalist.append({'filepath': file, 'start_time': window_start, 'label': window_label})

                window_start += args.window_len - args.overlap

        # get parameters for the electrode configurations
        self.tcp_ar_params = ned.nedc_load_parameters('../preprocessing/parameter_files/params_01_tcp_ar.txt')
        self.tcp_le_params = ned.nedc_load_parameters('../preprocessing/parameter_files/params_02_tcp_le.txt')
        self.tcp_ar_a_params = ned.nedc_load_parameters('../preprocessing/parameter_files/params_03_tcp_ar_a.txt')

    def __len__(self):
        """
        Denotes the total number of samples
        :return: length of the dataset
        """

        return len(self.datalist)

    def edf_to_tensor(self, edf_data, start, freq=250):
        """
        Performs STFT on given edf data, appends montages, and converts it to tensor
        :param edf_data: edf data
        :param start: start time of the window
        :param freq: frequency of sampling edf data
        :return:
        """

        return 0

    def __getitem__(self, index):
        """
        Generates one sample of data
        :param index: index of the data in the datalist
        :return: returns the data and label in float tensor and long tensor respectively
        """

        edf_path = self.datalist[index]['filepath']
        window_start = self.datalist[index]['start_time']
        label = self.datalist[index]['label']

        # check which electrode setup is used, labels of montage are not important
        # cut out montages 8 and 13 here
        if '01_tcp_ar' in edf_path:
            freq, edf, _ = load_edf(self.tcp_ar_params, edf_path)
        elif '02_tcp_le' in edf_path:
            freq, edf, _ = load_edf(self.tcp_le_params, edf_path)
        else:
            freq, edf, _ = load_edf(self.tcp_ar_a_params, edf_path)

        # cut out the correct window segment from each montage, STFT, data augment, append, convert it into a tensor
        output = self.edf_to_tensor(edf, window_start, freq)

        return output, label


def GenerateIterator(datapath, eval=False, shuffle=True):
    """
    Generates a batch iterator for data
    :param datapath: path to data
    :param txtcode: code that describes the author to load data from
    :param shuffle: whether to shuffle the batches around or not
    :return: a iterator combining the data into batches
    """

    params = {
        'batch_size': args.batch_size,  # batch size must be 1, as data is already separated into batches
        'shuffle': shuffle,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(datapath=datapath, eval=eval), **params)


GenerateIterator('../data/edf')
