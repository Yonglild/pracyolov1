import torch
from yolov1model import YOLOV1
from loss import loss_func
import torch.optim as optim
from torch.optim import lr_scheduler
from data_process import input_process, target_process
from data import PennFudanDataset
import torch.utils.data
# from PennFudanDataset_main import *
import cv2
from showresults import show_results

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
dataset_test = None
collate_fn = None       # 如何将单个样本变成minibatch


def collate_fn(batch):
    return tuple(batch)

# def collate_fn(batch):
#     return tuple(zip(*batch))

datapath = '/home/wyl/YOLO&SSD/PennFudanPed'
dataset = PennFudanDataset(datapath)
dataset_test = PennFudanDataset(datapath)
# dataset = PennFudanDataset(datapath, get_transform(train=False))
# dataset = torch.utils.data.Subset(dataset, [0])

train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=1,
            collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,  shuffle=False, num_workers=1,
            collate_fn=collate_fn)

# 优化backbone 还是 检测器
optimizer = optim.SGD(model.detector.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


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
        if epoch % 10 == 0:
            torch.save(model.state_dict(), '{}_{:.2f}.pth'.format(epoch, loss.item()))
        scheduler.step()

# 可能是训练时，损失函数的定义出现问题；
def test():
    Path = '/home/wyl/YOLO&SSD/Yolo_pratical/pracyolo1/60_0.03.pth'
    imgpath = '/home/wyl/YOLO&SSD/PennFudanPed/p/FudanPed00001.png'
    model.load_state_dict(torch.load(Path))
    model.eval()
    for iter, batch in enumerate(test_loader):
        img = batch['img']
        # img = cv2.imread(imgpath)
        # labels = target_process(batch).to(device)

        input = torch.tensor(img).permute((0, 3, 1, 2)).type(torch.FloatTensor)
        output = model(input.to(device))
        show_results(img, output)

if __name__ == '__main__':
    # train()
    test()