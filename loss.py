import torch
import numpy as np

# [1, 7, 7, 5]
def loss_func(outputs, labels):
    loss = 0
    rho1, rho2 = 0.3, 0.3
    batch, w, h, z = outputs.shape[:]
    loss_tmp, loss_conf_tmp = torch.zeros((w, h)), torch.zeros((w, h))
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

                w_l, h_l, w_pred, h_pred = w_l ** 0.5, h_l ** 0.5, torch.abs(w_pred) ** 0.5, torch.abs(h_pred) ** 0.5
                loss_geo = (x_l - x_pred)**2 + (y_l - y_pred)**2 + (w_l - w_pred)**2 + (h_l - h_pred)**2
                loss_geo *= c_l
                loss_tmp[i, j] = loss_geo + loss_conf * rho2
                loss += loss_tmp[i, j]
    return loss, loss_tmp, loss_geo, loss_conf_tmp

    # 判断维度
    # assert (outputs.shape == labels.shape), "outputs shape[%s] not equal labels shape[%s]" % (
    # outputs.shape, labels.shape)
    # # import pdb
    # # pdb.set_trace()
    # b, w, h, c = outputs.shape
    # loss = 0
    # # import pdb
    # # pdb.set_trace()
    # conf_loss_matrix = torch.zeros(b, w, h)
    # geo_loss_matrix = torch.zeros(b, w, h)
    # loss_matrix = torch.zeros(b, w, h)
    #
    # for bi in range(b):
    #     for wi in range(w):
    #         for hi in range(h):
    #             # import pdb
    #             # pdb.set_trace()
    #             # detect_vector=[confidence,x,y,w,h]
    #             detect_vector = outputs[bi, wi, hi]
    #             gt_dv = labels[bi, wi, hi]
    #             conf_pred = detect_vector[0]
    #             conf_gt = gt_dv[0]
    #             x_pred = detect_vector[1]
    #             x_gt = gt_dv[1]
    #             y_pred = detect_vector[2]
    #             y_gt = gt_dv[2]
    #             w_pred = detect_vector[3]
    #             w_gt = gt_dv[3]
    #             h_pred = detect_vector[4]
    #             h_gt = gt_dv[4]
    #             loss_confidence = (conf_pred - conf_gt) ** 2
    #             # loss_geo = (x_pred-x_gt)**2 + (y_pred-y_gt)**2 + (w_pred**0.5-w_gt**0.5)**2 + (h_pred**0.5-h_gt**0.5)**2
    #
    #             loss_geo = (x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2 + (w_pred - w_gt) ** 2 + (h_pred - h_gt) ** 2
    #             loss_geo = conf_gt * loss_geo
    #             loss_tmp = loss_confidence + 0.3 * loss_geo
    #             # print("loss[%s,%s] = %s,%s"%(wi,hi,loss_confidence.item(),loss_geo.item()))
    #             loss += loss_tmp
    #             conf_loss_matrix[bi, wi, hi] = loss_confidence
    #             geo_loss_matrix[bi, wi, hi] = loss_geo
    #             loss_matrix[bi, wi, hi] = loss_tmp
    # # 打印出batch中每张片的位置loss,和置信度输出！！！
    # # print(geo_loss_matrix)
    # # print(outputs[0, :, :, 0] > 0.5)
    # return loss, loss_matrix, geo_loss_matrix, conf_loss_matrix