import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def callbacks_during_train(model, dis_x, dis_y, dis_path, net, epoch, device='cpu'):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(dis_x, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(input_tensor).squeeze().cpu().numpy()

    _, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))

    ax_x_ori.imshow(cv2.cvtColor(cv2.imread(dis_path), cv2.COLOR_BGR2RGB))
    ax_x_ori.set_title('Original Image')

    ax_y.imshow(np.squeeze(dis_y), cmap=plt.cm.jet)
    ax_y.set_title('Ground_truth: ' + str(np.sum(dis_y)))

    ax_pred.imshow(pred, cmap=plt.cm.jet)
    ax_pred.set_title('Prediction: ' + str(np.sum(pred)))

    plt.savefig(f'tmp/{net}_{epoch}-epoch.png')
    plt.close()


def eval_loss(model, x, y, device='cpu', quality=False):
    model.eval()
    preds, DM, GT = [], [], []
    losses_SFN, losses_MAE, losses_MAPE, losses_RMSE = [], [], [], []

    with torch.no_grad():
        for idx_pd in range(x.shape[0]):
            input_tensor = torch.tensor(x[idx_pd], dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(input_tensor).squeeze().cpu().numpy()

            preds.append(pred)
            gt = np.squeeze(y[idx_pd])
            DM.append(gt)
            GT.append(round(np.sum(gt)))  # Integral GT

    for pred, gt_sum, gt_map in zip(preds, GT, DM):
        losses_SFN.append(np.mean((pred - gt_map) ** 2))
        losses_MAE.append(np.abs(np.sum(pred) - gt_sum))
        losses_MAPE.append(np.abs(np.sum(pred) - gt_sum) / gt_sum if gt_sum != 0 else 0)
        losses_RMSE.append((np.sum(pred) - gt_sum) ** 2)

    loss_SFN = np.sum(losses_SFN)
    loss_MAE = np.mean(losses_MAE)
    loss_MAPE = np.mean(losses_MAPE)
    loss_RMSE = np.sqrt(np.mean(losses_RMSE))

    if quality:
        psnr_list, ssim_list = [], []
        for pred, gt in zip(preds, DM):
            data_range = max(np.max(pred), np.max(gt)) - min(np.min(pred), np.min(gt))
            if data_range == 0:
                data_range = 1e-6
            psnr = compare_psnr(gt, pred, data_range=data_range)
            ssim = compare_ssim(gt, pred, data_range=data_range)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        return loss_MAE, loss_RMSE, loss_SFN, loss_MAPE, np.mean(psnr_list), np.mean(ssim_list)

    return loss_MAE, loss_RMSE, loss_SFN, loss_MAPE

print("All function used and done succesfully")