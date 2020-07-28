import torch
import numpy as np

# [1, 7, 7, 5]
def loss_func(outputs, labels):
    loss = 0
    rho1, rho2 = 0.3, 0.3
    batch, w, h, z = outputs.shape[:]
    loss_tmp, loss_conf_tmp = np.zeros((w, h)), np.zeros((w, h))
    for b in range(batch):
        for i in range(w):
            for j in range(h):
                c_l, c_pred = labels[b, i, j, 0], outputs[b, i, j, 0]
                x_l, x_pred = labels[b, i, j, 1], outputs[b, i, j, 1]
                y_l, y_pred = labels[b, i, j, 2], outputs[b, i, j, 2]
                w_l, w_pred = labels[b, i, j, 3], outputs[b, i, j, 3]
                h_l, h_pred = labels[b, i, j, 4], outputs[b, i, j, 4]
                if c_l == 1:
                    loss_conf = (c_l - c_pred) ** 2
                else:
                    loss_conf = rho1 * (c_l - c_pred) ** 2

                loss_conf_tmp[i, j] = loss_conf

                w_l, h_l, w_pred, h_pred = w_l ** 0.5, h_l ** 0.5, w_pred ** 0.5, h_pred ** 0.5
                loss_geo = (x_l - x_pred)**2 + (y_l - y_pred)**2 + (w_l - w_pred)**2 + (h_l - h_pred)**2
                loss_geo *= c_l
                loss_tmp[i, j] = loss_geo + loss_conf * rho2
                loss += loss_tmp[i, j]
    return loss, loss_tmp, loss_conf_tmp


