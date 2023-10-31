# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 180  # number of outputs per anchor            # 每一个 预选框预测输出，前nc个01字符对应类别，后5个对应：是否有目标，目标框的中心，目标框的宽高
        self.nl = len(anchors)  # number of detection layers            # 表示预选层数，yolov5是3层预选
        self.na = len(anchors[0]) // 2  # number of anchors            # 预选框数量，anchors数据中每一对数据表示一个预选框的宽高
        self.grid = [torch.zeros(1)] * self.nl  # init grid                # 初始化grid列表大小，空列表
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid                # 初始化anchor_grid列表大小，空列表
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)    # 注册常量anchor，并将预选框（尺寸）以数对形式存入 ---- 实际存的是框的宽高
        # 每一张进行三次预测，每一个预测结果包含nc+5+180个值
        # (n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20) --> ch=(255, 255, 255)
        # 255 -> (nc+5)*3 ===> 为了提取出预测框的位置信息以及预测框尺寸信息
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        """
        Args:
            x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)

        Return：
            if train:
                x (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            else:
                inference (tensor): (b, n_all_anchors, self.no)
                x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)
        """
        z = []  # inference output
        # 输入的x是来自三层金字塔的预测结果(n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20)

        # https://blog.csdn.net/weixin_43799388/article/details/126207632
        # logits_ = []  # redutu修改---1

        for i in range(self.nl):
            # 下面3行代码的工作：
            # (n, 255, _, _) -> (n, 3, nc+5+180, ny, nx) -> (n, 3, ny, nx, nc+5+180)
            # 相当于三层分别预测了80*80、40*40、20*20次，每一次预测都包含3个框
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)

            # contiguous 将数据保证内存中位置连续
            # view()变换形状，数据不变, x(bs,255,20,20) to x(bs,3,85,20,20),将一个预测层里的3个anchor的信息分出来，每个预测框预测信息数量为self.no(这里为85)
            # permute(0, 1, 3, 4, 2)，x[i]有5个维度，（2，3，4）变成（3，4，2）,x(bs,3,85,20,20)to x(bs,3,20,20,85)

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # self.training 作为nn.Module的参数，默认是True，因此下方代码先不考虑
            '''
            网络的for loop次数为3，也就是依次在这3个特征图上进行网格化预测，利用卷积操作得到通道数为no×nl的特征输出。
            拿128x80x80举例，在nc=15的情况下经过卷积得到60x80x80的特征图，这个特征图就是后续用于格点检测的特征图。
            '''
            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # 为每一层划分网格
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # logits = x[i][..., 5:]  # redutu修改---2
                '''
                随后就是基于经过检测器卷积后的特征图划分网格，网格的尺寸是与输入尺寸相同的，如20x20的特征图会变成20x20的网格，那么一个
                网格对应到原图中就是32x32像素；40x40的一个网格就会对应到原图的16x16像素，以此类推。
                '''
                y = x[i].sigmoid() # (tensor): (b, self.na, h, w, self.no)
                # 改变原数据
                if self.inplace:
                    # grid[i] = (3, 20, 20, 2), y = [n, 3, 20, 20, nc+5]
                    # grid实际是 位置基准 或者理解为 cell的预测初始位置，而y[..., 0:2]是作为在grid坐标基础上的位置偏移
                    # anchor_grid实际是 预测框基准 或者理解为 预测框的初始位置，而 y[..., 2:4]是作为预测框位置的调整
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    # stride应该是一个grid cell的实际尺寸
                    # 经过sigmoid，值范围变成了(0-1),下一行代码将值变成范围（-0.5，1.5），
                    # 相当于预选框上下左右都扩大了0.5倍的移动区域，不易大于0.5倍，否则就重复检验了其他网格的内容了
                    # 此处的1表示一个grid cell的尺寸，尽量让预测框的中心在grid cell中心附近
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # 范围变成(0-4)倍，设置为4倍的原因是下层的感受野是上层的2倍
                    # 因下层注重检测大目标，相对比上层而言，计算量更小，4倍是一个折中的选择
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1) 
                z.append(y.view(bs, -1, self.no)) # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)

                # logits_.append(logits.view(bs, -1, self.no - 5))  # redutu修改---3

        return x if self.training else (torch.cat(z, 1), x)  # return if no redutu
        # return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)  # redutu修改---4

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            # 网格标尺坐标
            # indexing='ij' 表示的是i是同一行，j表示同一列
            # indexing='xy' 表示的是x是同一列，y表示同一行
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        # grid --> (20, 20, 2), 拓展（复制）成3倍，因为是三个框 -> (3, 20, 20, 2)
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        # 与grid同理
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1) # featuremap pixel
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Args:
            x (tensor): (b, 3, height, width), RGB

        Return：
            if not augment:
                x (list[P3_out, ...]): tensor.Size(b, self.na, h_i, w_i, c), self.na means the number of anchors scales
            else:
                
        """
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
        Args:
            x (tensor): (b, 3, height, width), RGB

        Return：
            x (list[P3_out, ...]): tensor.Size(b, self.na, h_i, w_i, c), self.na means the number of anchors scales
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    no = na * (nc + 185)  # number of outputs = anchors * (classes + 185)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, GSConv, VoVGSCSP, VoVGSCSPC, ShuffleAttention,
                 C3CBAM, CBAM, C2f, nn.ConvTranspose2d]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, VoVGSCSP, VoVGSCSP, C2f]:
                args.insert(2, n)  # number of repeats
                n = 1
            # 迪导, Bifusion
            elif m is nn.ConvTranspose2d:
                if len(args) >= 7:
                    args[6] = make_divisible(args[6] * gw, 8)
        # # by CSDN 迪菲赫尔曼 请勿散播转发，仅供学习交流
        # # ------------Attention ↆ------------
        # elif m in [SimAM, ECA, ParNetAttention, SpatialGroupEnhance,
        #            TripletAttention]:
        #     args = [*args[:]]
        # elif m in [CoordAtt, GAMAttention]:
        #     c1, c2 = ch[f], args[0]
        #     if c2 != no:
        #         c2 = make_divisible(c2 * gw, 8)
        #     args = [c1, c2, *args[1:]]
        # elif m in [SE, ShuffleAttention, CBAM, SKAttention, DoubleAttention, CoTAttention, EffectiveSEModule,
        #            GlobalContext, GatherExcite, MHSA]:
        #     c1 = ch[f]
        #     args = [c1, *args[0:]]

        elif m is GatherExcite:  # 一个注意力机制用is， 一堆用in
            c1 = ch[f]
            args = [c1, *args[0:]]
        # elif m in [S2Attention, NAMAttention, CrissCrossAttention, SequentialPolarizedSelfAttention,
        #            ParallelPolarizedSelfAttention]:
        #     c1 = ch[f]
        #     args = [c1]
        # # ------------Attention ↑--------------
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        # add ShuffleAttention:
        elif m is ShuffleAttention:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        # S2-MLPv2注意力机制 芒果哥
        elif m is S2Attention:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)

        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--cfg', type=str, default='yolov5mtest.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
