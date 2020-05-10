from utils.dataset import GenerateIterator
from utils.models import CNN
from torch import optim
from torch import nn
import torch
import tqdm
import numpy as np
from myargs import args


def train(datapath, parampath, continue_train=False):

    # create iterators
    train_iter = GenerateIterator(datapath, parampath, eval=False)
    val_iter = GenerateIterator(datapath, parampath, eval=True)

    # get model
    model = CNN()

    # get optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    lossfn = nn.CrossEntropyLoss()

    start_epoch = args.start_epoch

    # if training model from previous saved weights
    if continue_train:
        pretrained_dict = torch.load(
            '{}/models/{}_model_{}.pt'.format(datapath, args.model_name, args.start_epoch
        ))['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    for epoch in range(start_epoch+1, args.num_epochs):

        # ==================== Training set ====================

        # progress bar to view progression of model
        train_pbar = tqdm.tqdm(train_iter)

        # used to check accuracy to gauge model progression on training set
        train_losses_sum = 0
        train_n_total = 1
        train_pred_classes = []
        train_ground_truths = []

        for i, (stft_item, label) in enumerate(train_pbar):

            if args.early_break > 0 and i > args.early_break:
                break

            # move to GPU
            if torch.cuda.is_available():
                stft_item = stft_item.cuda()
                label = label.cuda()

            # get prediction
            prediction = model(stft_item)

            # predictions to check for model progression
            pred_class = torch.argmax(prediction, dim=-1)
            train_pred_classes.extend(pred_class.cpu().data.numpy().tolist())
            train_ground_truths.extend(label.cpu().data.numpy().tolist())

            # get loss
            loss = lossfn(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses_sum += loss
            train_pbar.set_description('Epoch: {} || Loss: {:.5f} '.format(epoch, train_losses_sum / train_n_total))
            train_n_total += 1

        train_pred_classes = np.asarray(train_pred_classes)
        train_ground_truths = np.asarray(train_ground_truths)
        train_accuracy = np.mean((train_pred_classes == train_ground_truths)).astype(np.float)

        # ==================== Validation set ====================

        # change modulo to do validation every few epochs
        if epoch % 1 == 0:

            # evaluate
            with torch.no_grad():
                model.eval()

                # progress bar to view progression of model
                val_pbar = tqdm.tqdm(val_iter)

                # used to check accuracy to gauge model progression on validation set
                val_losses_sum = 0
                val_n_total = 1
                val_pred_classes = []
                val_ground_truths = []

                for i, (stft_item, label) in enumerate(val_pbar):

                    if args.early_break > 0 and i > args.early_break:
                        break

                    # move to GPU
                    if torch.cuda.is_available():
                        stft_item = stft_item.cuda()
                        label = label.cuda()

                    # get prediction
                    prediction = model(stft_item)

                    # predictions to check for model progression
                    pred_class = torch.argmax(prediction, dim=-1)
                    val_pred_classes.extend(pred_class.cpu().data.numpy().tolist())
                    val_ground_truths.extend(label.cpu().data.numpy().tolist())

                    # get loss
                    loss = lossfn(prediction, label)

                    val_losses_sum += loss
                    val_pbar.set_description('Epoch: {} || Loss: {:.5f} '.format(epoch, val_losses_sum / val_n_total))
                    val_n_total += 1

                val_pred_classes = np.asarray(val_pred_classes)
                val_ground_truths = np.asarray(val_ground_truths)
                val_accuracy = np.mean((val_pred_classes == val_ground_truths)).astype(np.float)

                model.train()

            print('Epoch: {} || Train_Acc: {} || Train_Loss: {} || Val_Acc: {} || Val_Loss: {}'.format(
                epoch, train_accuracy, train_losses_sum / train_n_total, val_accuracy, val_losses_sum / val_n_total
            ))

        # change modulo number to save every few epochs
        if epoch % 1 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './data/models/{}_model_{}.pt'.format(args.model_name, epoch))


if __name__ == '__main__':
    train('./data', './preprocessing/parameter_files')
