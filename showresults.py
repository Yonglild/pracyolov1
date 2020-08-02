import torch
import cv2

def show_results(img, results):
    h, w = img.shape[:2]
    tmpresults = []
    wscale, hscale = w/448, h/448
    for i in range(7):
        for j in range(7):
            if results[0, i, j, 0] > 0.5:
                tmpresults.append(results[0, i, j, :])
    for result in tmpresults:
        conf, xcenter, ycenter, w, h = result
        w, h = w * 448 * wscale, h * 448 * hscale
        xcenter, ycenter = xcenter * 448 * wscale, ycenter * 448 * hscale
        xmin, ymin = xcenter - w/2, ycenter - h/2
        xmax, ymax = xcenter + w/2, ycenter + h/2
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        img = cv2.putText(img, str(conf), (xmin, ymin), cv.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    cv2.imshow(' ', img)
    cv2.waitKey()