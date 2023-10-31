# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def fitness(x):
    # Model fitness as a weighted combination of metrics  以矩阵的加权组合作为模型的适应度
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]  每个变量对应的权重 [P, R, mAP@0.5, mAP@0.5:0.95]
    # (torch.tensor).sum(1) 每一行求和tensor为二维时返回一个以每一行求和为结果的行向量
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.  计算平均精度（AP），并绘制P-R曲线
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).  预测目标类别
        target_cls:  True object classes (nparray).  真实目标类别
        plot:  Plot precision-recall curve at mAP@0.5  在mAP@0.5的情况下  是否绘制P-R曲线
        save_dir:  Plot save directory  P-R曲线图的保存路径
    # Returns
        像faster-rcnn那种方式计算AP （这里涉及计算AP的两种不同方式 建议查询）
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness  # 将目标进行排序
    # np.argsort(-conf)函数返回一个索引数组 其中每一个数按照conf中元素从大到小 置为 0,1...n
    i = np.argsort(-conf)
    # tp conf pred_cls 三个矩阵均按照置信度从大到小进行排列
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes  # 找到各个独立的类别
    # np.unique()会返回输入array中出现至少一次的变量 这里返回所有独立的类别
    # print(tp, conf, pred_cls)
    unique_classes, nt = np.unique(target_cls, return_counts=True)  # for plotting

    nc = unique_classes.shape[0]  # number of classes

    # Create Precision-Recall curve and compute AP for each class  创建P-R曲线 并 计算每一个类别的AP
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    # 初始化 对每一个类别在每一个IOU阈值下面 计算P R AP参数, tp.shape[1]: IOU loss 阈值的类别的 (i.e. 10 for mAP0.5...0.95), tp: true positive
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):  # ci为类别对应索引 c为具体的类别
        i = pred_cls == c
        n_l = nt[ci]  # number of labels  # ground truth中 类别c 的个数 all_results
        n_p = i.sum()  # number of predictions    # 预测类别中为 类别c 的个数

        if n_p == 0 or n_l == 0:  #如果没有预测到 或者 ground truth没有标注 则略过类别c
            continue
        else:
            """ 
                计算 FP（False Positive） 和 TP(Ture Positive)
                tp[i] 会根据i中对应位置是否为False来决定是否删除这一位的内容，如下所示：
                a = np.array([0,1,0,1]) i = np.array([True,False,False,True]) b = a[i]
                则b为：[0 1]
                而.cumsum(0)函数会 按照对象进行累加操作，如下所示：
                 a = np.array([0,1,0,1]) b = a.cumsum(0)
                则b为：[0,1,1,2]   0:0  1:0+1  1:0+1+0  2:0+1+0+1
                （FP + TP = all_detections   所以有 fp[i] = 1 - tp[i]）
                所以fpc为 类别c 按照置信度从大到小排列 截止到每一位的FP数目
                tpc为 类别c 按照置信度从大到小排列 截止到每一位的TP数目
                recall 和 precision 均按照元素从小到大排列
            """
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            # Recall = TP / (TP + FN) = TP / all_results = TP / n_l
            recall = tpc / (n_l + eps)  # recall curve  # 加一个1e-16的目的是防止n_l为0 时除不开
            """
                np.interp() 函数第一个输入值为数值 第二第三个变量为一组x y坐标 返回结果为一个数值
                这个数值为 找寻该数值左右两边的x值 并将两者对应的y值取平均 如果在左侧或右侧 则取 边界值
                如果第一个输入为数组 则返回一个数组 其中每一个元素按照上述计算规则产生
            """

            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases,  pr_score 处的y值

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:  # iou_thres为IOU loss的阈值
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """
        params nc: 数据集类别个数
        params conf: 预测框置信度阈值
        Params iou_thres: iou阈值
        """
        self.matrix = np.zeros((nc + 1, nc + 1))  # +1: add background class
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        返回 各个box之间的交并比(iou)
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        每一个box的集合都被期望使用(x1,y1,x2,y2)的形式 这两个点为box的对角顶点
        Arguments:  detections 和 labels的数据结构
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
            无返回 更新混淆矩阵
        """
        # detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        detections = detections[detections[:, 4] > self.conf]  # 返回检测大于阈值的预测框（和nms差不多）
        # gt_classes (Array[M, 1]), ground_truth class
        gt_classes = labels[:, 0].int()  # # 返回ground truth的类别
        # detection_classes (Array[M, 1]), predicted class
        detection_classes = detections[:, 5].int()  # 返回检测到的类别
        # iou计算	box1 (Array[N, 4]), x1, y1, x2, y2
        #           box2 (Array[M, 4]), x1, y1, x2, y2
        # iou (Tensor[N, M]) NxM矩阵包含了 box1中每一个框和box2中每一个框的iou值
        # 非常重要！ iou中坐标 (n1,m1) 代表 第n1个ground truth 框 和 第m1个 预测框的
        iou = box_iou(labels[:, 1:], detections[:, :4])  #调用general中计算iou的方式计算iou
        # x为一个含有两个tensor的tuple表示iou中大于阈值的值的坐标，第一个tensor为第几行，第二个为第几列
        x = torch.where(iou > self.iou_thres)  #找到iou中大于阈值的那部分并提取
        if x[0].shape[0]:   # 当大于阈值的坐标不止一个的时候
            """
            torch.cat(inputs,dimension=0) 为在指定的维度对 张量inputs进行堆叠 
            二维情况下 0代表按照行 1代表按照列 0时会增加行 1时会增加列
            torch.stack(x,1) 当x为二维张量的时候 本质上是对x做转置操作
            .cpu()是将变量转移到cpu上进行运算.numpy()是转换为numpy数组
            matches (Array[N, 3]), row,col,iou_value ！！！
            row为大于阈值的iou张量中点的横坐标 col为纵坐标 iou_value为对应的iou值
            """
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:  # 当box个数大于1时进行以下过程 此处matches的过滤过程见下文 补充部分
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按第三列iou从大到小重排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 按顺序取第二列中各个框首次出现(不同预测的框)的行(即每一种预测的框中iou最大的那个)
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 按顺序取第一列中各个框首次出现(不同gt的框)的行(即每一种gt框中iou最大的那个)
        else:
            matches = np.zeros((0, 3))  # 这里返回一个0行3列全0的二维数组 ？因为没有一个例子满足这个要求

        n = matches.shape[0] > 0  #这里n为 True 或 False 用于判断是否存在满足阈值要求的对象是否至少有一个
        """
              a.transpose()是numpy中轮换维度索引的方法 对二维数组表示为转置
              此处matches (Array[N, 3]), row,col,iou_value
              物理意义：在大于阈值的前提下，N*M种label与预测框的组合可能下，每一种预测框与所有label框iou值最大的那个
              m0，m1  (Array[1, N])
              m0代表 满足上述条件的第i个label框   （也即类别）
              m1代表 满足上述条件的第j个predict框 （也即类别）
        """
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):   #解析ground truth 中的类别
            j = m0 == i
            if n and sum(j) == 1:  # 检测到的目标至少有1个 且 ground truth对应只有一个
                #  如果sum(j)=1 说明gt[i]这个真实框被某个预测框检测到了
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct  # TP 判断正确的数目加1
            else:
                #  如果sum(j)=0 说明gt[i]这个真实框没被任何预测框检测到 也就是说这个真实框被检测成了背景框
                self.matrix[self.nc, gc] += 1  # background FP  # 背景 FP（false positive） 个数加1 背景被误认为目标

        if n:  # 当目标不止一个时
            for i, dc in enumerate(detection_classes):  # i为索引 dc为每一个目标检测到的类别
                if not any(m1 == i):  # 检测到目标 但是目标与groundtruth的iou小于之前要求的阈值则
                    self.matrix[dc, self.nc] += 1  # background FN  # 背景 FN 个数加1 （目标被检测成了背景）

    def matrix(self):  #返回matrix变量 该matrix为混淆矩阵
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn  #seaborn 为易于可视化的一个模块
            # 按照每一列进行归一化
            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)  # 小于0.005的值被认为NaN

            fig = plt.figure(figsize=(12, 9), tight_layout=True)  #初始化画布
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size  设置标签的尺寸
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels  用于绘制过程中判断是否应用names
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):  # 打印出每一个元素对应的数据
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:
                return iou - rho2 / c2  # DIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()
