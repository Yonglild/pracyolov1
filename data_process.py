import PIL.Image as Image
import torch
import cv2

def input_process(batch):
    batchsize = len(batch)
    imglist = []
    for i in range(batchsize):
        img = batch[i]['img']
        img = cv2.resize(img, (448, 448))
        img = torch.Tensor(img).permute((2, 0, 1))
        imglist.append(img)
    return torch.stack(imglist, dim=0)

def target_process(batch):
    batchsize = len(batch)
    target = torch.zeros([batchsize, 7, 7, 5])
    for i in range(batchsize):
        w, h = batch[i]['imgwh'][:2]
        wscale, hscale = w / 448, h / 448
        print(wscale, hscale)
        step = 448 / 7
        obj_nums = len(batch[i]['boxs'])
        for j in range(obj_nums):
            xmin, ymin, xmax, ymax = batch[i]['boxs'][j]
            objw, objh = batch[i]['boxwhs'][j]
            xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
            print(xcenter, ycenter)
            xcenter /= wscale
            ycenter /= hscale
            objw /= wscale
            objh /= hscale
            x_w = int(xcenter // step)
            y_h = int(ycenter // step)
            print(xcenter, ycenter)
            print(x_w, y_h)
            target[i, y_h, x_w, :] = torch.Tensor([1, xcenter, ycenter, objw, objh])
            print(target[i, y_h, x_w, 2])
    return target
