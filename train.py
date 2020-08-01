import torch
from yolov1model import YOLOV1
from loss import loss_func
import torch.optim as optim
from torch.optim import lr_scheduler
# from data_process import input_process, target_process
from data import PennFudanDataset
import torch.utils.data
from PennFudanDataset_main import *
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = YOLOV1().to(device)
print(model)

num_classes = 2
n_class    = 2
batch_size = 6
epochs     = 500
lr         = 1e-3
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5

# 数据加载方式重写
dataset = None
dataset_test = None
collate_fn = None       # 如何将单个样本变成minibatch


# def collate_fn(batch):
#     return tuple(batch)

def collate_fn(batch):
    return tuple(zip(*batch))

datapath = '/home/bubble/wyl/data/PennFudanPed'
# PennFudan = PennFudanDataset(datapath)
dataset = PennFudanDataset(datapath, get_transform(train=False))
dataset = torch.utils.data.Subset(dataset, [0])

train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1,
            collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4)

# 优化backbone 还是 检测器
optimizer = optim.SGD(model.detector.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def input_process(batch):
    # import pdb
    # pdb.set_trace()
    batch_size = len(batch[0])
    input_batch = torch.zeros(batch_size, 3, 448, 448)
    for i in range(batch_size):
        inputs_tmp = batch[0][i]
        inputs_tmp1 = cv2.resize(inputs_tmp.permute([1, 2, 0]).numpy(), (448, 448))
        inputs_tmp2 = torch.tensor(inputs_tmp1).permute([2, 0, 1])
        input_batch[i:i + 1, :, :, :] = torch.unsqueeze(inputs_tmp2, 0)
    return input_batch


# batch[1][0]['boxes'][0]
def target_process(batch, grid_number=7):
    # batch[1]表示label
    # batch[0]表示image
    batch_size = len(batch[0])
    target_batch = torch.zeros(batch_size, grid_number, grid_number, 5)
    # import pdb
    # pdb.set_trace()
    for i in range(batch_size):
        labels = batch[1]
        batch_labels = labels[i]
        # import pdb
        # pdb.set_trace()
        number_box = len(batch_labels['boxes'])
        for wi in range(grid_number):
            for hi in range(grid_number):
                # 便利每个标注的框
                for bi in range(number_box):
                    bbox = batch_labels['boxes'][bi]
                    _, himg, wimg = batch[0][i].numpy().shape
                    bbox = bbox / torch.tensor([wimg, himg, wimg, himg]).float()
                    # import pdb
                    # pdb.set_trace()
                    center_x = (bbox[0] + bbox[2]) * 0.5
                    center_y = (bbox[1] + bbox[3]) * 0.5
                    # print("[%s,%s,%s],[%s,%s,%s]"%(wi/grid_number,center_x,(wi+1)/grid_number,hi/grid_number,center_y,(hi+1)/grid_number))
                    if center_x <= (wi + 1) / grid_number and center_x >= wi / grid_number and center_y <= (
                            hi + 1) / grid_number and center_y >= hi / grid_number:
                        # pdb.set_trace()
                        cbbox = torch.cat([torch.ones(1), bbox])
                        # 中心点落在grid内，
                        target_batch[i:i + 1, wi:wi + 1, hi:hi + 1, :] = torch.unsqueeze(cbbox, 0)
                    # else:
                    # cbbox =  torch.cat([torch.zeros(1),bbox])
                # import pdb
                # pdb.set_trace()
                # rint(target_batch[i:i+1,wi:wi+1,hi:hi+1,:])
                # target_batch[i:i+1,wi:wi+1,hi:hi+1,:] = torch.unsqueeze(cbbox,0)
    return target_batch

def train():
    model.train()
    for epoch in range(epochs):
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # 取图片
            inputs = input_process(batch).to(device)
            # 取标注
            labels = target_process(batch).to(device)

            # 获取得到输出
            outputs = model(inputs)
            # print('outputs', outputs)
            loss, loss_tmp, geo_loss, loss_conf_tmp = loss_func(outputs, labels)
            print('loss', loss)
            # print('loss_tmp:', loss_tmp)
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}, lr: {}".format(epoch, iter, loss.data.item(),
                                                                 optimizer.state_dict()['param_groups'][0]['lr']))
        # if epoch // 30 == 0:
            # torch.save(YOLOV1.state_dict(), '{}_{:.2f}.pth'.format(epoch, loss.item()))
        scheduler.step()

train()