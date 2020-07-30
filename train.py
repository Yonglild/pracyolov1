import torch
from yolov1model import YOLOV1
from loss import loss_func
import torch.optim as optim
from torch.optim import lr_scheduler
from data_process import input_process, target_process
from data import PennFudanDataset
import torch.utils.data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = YOLOV1().to(device)

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


def collate_fn(batch):
    return tuple(batch)

datapath = '/home/wyl/YOLO&SSD/PennFudanPed'
PennFudan = PennFudanDataset(datapath)

train_loader = torch.utils.data.DataLoader(
        PennFudan, batch_size=2, shuffle=False, num_workers=1,
            collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4)
# 优化backbone 还是 检测器
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def train():
    for epoch in range(epochs):
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # 取图片
            inputs = input_process(batch).to(device)
            # 取标注
            labels = target_process(batch).to(device)

            # 获取得到输出
            outputs = model(inputs)

            loss, loss_tmp, loss_conf_tmp = loss_func(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}, lr: {}".format(epoch, iter, loss.data.item(),
                                                                 optimizer.state_dict()['param_groups'][0]['lr']))
        if epoch // 30 == 0:
            torch.save(YOLOV1.state_dict(), '{}_{:.2f}.pth'.format(epoch, loss.item()))
        scheduler.step()

train()