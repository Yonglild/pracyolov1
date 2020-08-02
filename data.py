from PIL import Image
import os
import re
import numpy as np


class PennFudanDataset():
    def __init__(self, root):
        self.imgdir = os.path.join(root, 'PNGImages')
        self.annodir = os.path.join(root, 'Annotation')
        self.imgs = os.listdir(self.imgdir)


    def __getitem__(self, item):
        imgpath = os.path.join(self.imgdir, self.imgs[item])
        imgname = os.path.splitext(os.path.basename(imgpath))[0]
        img = Image.open(imgpath)
        annoname = self.annodir + '/' + imgname + '.txt'
        boxs, whs = self.getbox(annoname)   # []
        print(annoname, boxs)
        target = {}
        target['img'] = np.array(img)/255
        target['boxs'] = boxs
        target['boxwhs'] = whs
        target['imgwh'] = img.size
        target['name'] = imgpath
        return target


    def __len__(self):
        return len(self.imgs)


    def getbox(self, anno):
        boxs, whs = [], []
        with open(anno, 'r') as f:
            lines = f.readlines()
            objnum = int(re.findall(r"\d+", lines[4])[0])
            objstart = 10   #
            for i in range(objnum):
                boxline = objstart + 5 * i
                boxs_str = re.findall(r"\d+", lines[boxline])
                xmin, ymin, xmax, ymax = int(boxs_str[1])-1, int(boxs_str[2])-1, int(boxs_str[3])-1, int(boxs_str[4])-1
                boxs.append([xmin, ymin, xmax, ymax])
                whs.append([xmax - xmin, ymax - ymin])
        return boxs, whs