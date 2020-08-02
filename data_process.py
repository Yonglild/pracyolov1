import PIL.Image as Image
import torch
import cv2

def input_process(batch):
    batchsize = len(batch)
    imglist = []
    for i in range(batchsize):
        img = batch[i]['img']
        img = cv2.resize(img, (448, 448))
        img = torch.tensor(img).permute((2, 0, 1))
        imglist.append(img)
    return torch.stack(imglist, dim=0).type(torch.FloatTensor)


    #import pdb
    #pdb.set_trace()
    batch_size=len(batch)
    input_batch= torch.zeros(batch_size,3,448,448)
    for i in range(batch_size):
        inputs_tmp = torch.tensor(batch[0]['img'])
        inputs_tmp1=cv2.resize(inputs_tmp.numpy(),(448,448))
        inputs_tmp2=torch.tensor(inputs_tmp1).permute([2,0,1])
        input_batch[i:i+1,:,:,:]= torch.unsqueeze(inputs_tmp2,0)
    return input_batch


def target_process(batch):
    batchsize = len(batch)
    target = torch.zeros([batchsize, 7, 7, 5])
    for i in range(batchsize):
        w, h = batch[i]['imgwh'][:2]
        wscale, hscale = w / 448, h / 448
        # print(wscale, hscale)
        step = 448 / 7
        obj_nums = len(batch[i]['boxs'])
        for j in range(obj_nums):
            xmin, ymin, xmax, ymax = batch[i]['boxs'][j]
            objw, objh = batch[i]['boxwhs'][j]
            xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
            # print(xcenter, ycenter)
            xcenter /= wscale
            ycenter /= hscale
            objw /= wscale
            objh /= hscale
            x_w = int(xcenter // step)
            y_h = int(ycenter // step)
            # print(xcenter, ycenter)
            # print(x_w, y_h)
            target[i, x_w, y_h, :] = torch.Tensor([1, xcenter, ycenter, objw, objh]).float()/torch.Tensor([1., 448., 448., 448., 448.]).float()
            # print(target[i, y_h, x_w, 2])
    return target
