from utils.dataset import GenerateIterator_eval
from utils.models import CNN
from myargs import args
import torch
import numpy as np
from scipy.ndimage.morphology import binary_closing, binary_opening
from tqdm import tqdm


def generate(datapath, parampath, keep=None):
    """
    generate hyp.txt for a trained model
    :param datapath: path to data
    :param parampath: path to parameter files
    :param keep: which channels to keep
    :return: none
    """

    print('keeping channels {}'.format(str(keep) if keep is not None else 'all'))

    # create iterators, change val into false for generating on test set
    eval_iter = GenerateIterator_eval(datapath, parampath, keep=keep, val=True, shuffle=True)
    answers = eval_iter.dataset.get_anslist()

    # get model
    model = CNN(keep)

    # load_model
    pretrained_dict = torch.load('{}/models/ch{}_{}_model_{}.pt'.format(
        datapath,
        '-'.join([str(ch) for ch in keep]),
        args.model_name,
        args.eval_epoch
    ))['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()

    pbar = tqdm(eval_iter)

    for stft_items, start_times, file_paths in pbar:

        # move to GPU
        if torch.cuda.is_available():
            stft_items = stft_items.cuda()

        # get predictions
        preds = model(stft_items)

        # set answer arrays
        for pred, st_time, file in zip(preds, start_times, file_paths):
            answers[file]['pred'][st_time: st_time + args.window_len] += pred[1].item()
            answers[file]['counter'][st_time: st_time + args.window_len] += 1

    # create hypothesis text
    for file_path in answers.keys():
        pred_arr = answers[file_path]['pred']
        counter_arr = answers[file_path]['counter']

        final_arr = np.divide(pred_arr, counter_arr)

        # adjust threshold for more/less sensitive
        threshold = 0.9
        final_arr = final_arr > threshold

        # test larger structures and changing order on final output
        final_arr = binary_closing(final_arr, structure=np.ones(3))
        final_arr = binary_opening(final_arr, structure=np.ones(3))

        file_name = file_path.split('\\')[-1].replace('.edf', '')

        # append to file
        with open('hyp.txt', 'a') as fp:
            seiz_start = 0
            for i in range(len(final_arr) - 1):
                if final_arr[i] == 0 and final_arr[i+1] == 1:
                    seiz_start = i+1

                elif final_arr[i] == 1 and final_arr[i+1] == 0:
                    # add start/end time to the file
                    # don't wanna report confidence so use 1.0, and 3 is the number of channels we are using
                    fp.write(f'{file_name} {float(seiz_start)} {float(i+1)} {1.0} {3}\n')


generate('./data', './preprocessing/parameter_files', keep=[13])





