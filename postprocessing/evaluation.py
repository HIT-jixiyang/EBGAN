import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
plt.switch_backend('agg')

import config as c
# from utils import denormalize_frames, normalize_frames


def pixel_to_dBZ(img):
    """

    Parameters
    ----------
    img : np.ndarray or float

    Returns
    -------

    """
    return img * 70.0 - 10.0


def dBZ_to_pixel(dBZ_img):
    """

    Parameters
    ----------
    dBZ_img : np.ndarray

    Returns
    -------

    """
    return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)


def pixel_to_rainfall(img, a=None, b=None):
    """Convert the pixel values to real rainfall intensity

    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    if a is None:
        a = c.ZR_a
    if b is None:
        b = c.ZR_b
    dBZ = pixel_to_dBZ(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity


def rainfall_to_pixel(rainfall_intensity, a=None, b=None):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    if a is None:
        a = c.ZR_a
    if b is None:
        b = c.ZR_b
    dBR = np.log10(rainfall_intensity) * 10.0
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1,img2):

    seq_length = img1.get_shape().as_list()[1]
    loss_ssim = []
    for t in range(seq_length):
        loss_ssim.append(ssim(img1[:,t],img2[:,t]))
    return 1-tf.reduce_mean(tf.stack(loss_ssim))


def ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def image_to_4d(image):
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, -1)
    return image

def get_loss_weight_symbol(data):
    if c.USE_BALANCED_LOSS:
        balancing_weights = c.BALANCING_WEIGHTS
        weights = tf.ones_like(data) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in c.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * tf.to_float(data >= threshold)
        weights = weights
    else:
        weights = tf.ones_like(data)
    if c.TEMPORAL_WEIGHT_TYPE == "same":
        return weights
    else:
        raise NotImplementedError


def weighted_mse(pred, gt, weight):
    return weighted_l2(pred, gt, weight)


def weighted_l2(pred, gt, weight):
    """

    Parameters
    ----------
    pred : tensor
        Shape: (batch_size, seq_len, H, W, C)
    gt : tensor
        Shape: (batch_size, seq_len, H, W, C)
    weight : tensor
        Shape: (batch_size, seq_len, H, W, C)

    Returns
    -------
    l2 : Value
    """
    l2 = weight * tf.square(pred - gt)
    l2 = tf.reduce_sum(l2)
    return l2


def weighted_l1(pred, gt, weight):
    l1 = weight * tf.abs(pred - gt)
    l1 = tf.reduce_sum(l1)
    return l1


def weighted_mae(pred, gt, weight):
    return weighted_l1(pred, gt, weight)


def one_step_diff(dat, axis):
    """

    Parameters
    ----------
    dat : tensor (b, length, h, w, c)
    axes : int 2, 3

    Returns
    -------

    """
    if axis == 2:
        return dat[:, :, :-1, :, :] - dat[:, :, 1:, :, :]
    elif axis == 3:
        return dat[:, :, :, :-1, :] - dat[:, :, :, 1:, :]
    else:
        raise NotImplementedError
def gdl_loss(pred, gt):
    """

    Parameters
    ----------
    pred : tensor
        Shape: (b, length, h, w, c)
    gt : tensor
        Shape: (b, length, h, w, c)
    Returns
    -------
    gdl : value
    """
    pred_diff_h = tf.abs(one_step_diff(pred, axis=2))
    pred_diff_w = tf.abs(one_step_diff(pred, axis=3))
    gt_diff_h = tf.abs(one_step_diff(gt, axis=2))
    gt_diff_w = tf.abs(one_step_diff(gt, axis=3))
    gd_h = tf.abs(pred_diff_h - gt_diff_h)
    gd_w = tf.abs(pred_diff_w - gt_diff_w)
    gdl = tf.reduce_sum(gd_h) + tf.reduce_sum(gd_w)
    return gdl


plt.switch_backend('agg')

EVALUATION_THRESHOLDS = [12.9777173087837, 28.577717308783704, 33.27378524114181, 40.71687681476854]


class Evaluator(object):
    def __init__(self, save_path, seq=10):
        self.metric = {}
        for threshold in EVALUATION_THRESHOLDS:
            self.metric[threshold] = {
                "pod": np.zeros((seq, 1), np.float32),
                "far": np.zeros((seq, 1), np.float32),
                "csi": np.zeros((seq, 1), np.float32),
                "hss": np.zeros((seq, 1), np.float32)
            }
        self.seq = seq
        self.save_path = save_path
        self.total = 0
        print(self.metric.keys())

    def get_metrics(self, gt, pred, threshold):
        b_gt = gt > threshold
        b_pred = pred > threshold
        b_gt_n = np.logical_not(b_gt)
        b_pred_n = np.logical_not(b_pred)

        summation_axis = (0, 2, 3)

        hits = np.logical_and(b_pred, b_gt).sum(axis=summation_axis)
        misses = np.logical_and(b_pred_n, b_gt).sum(axis=summation_axis)
        false_alarms = np.logical_and(b_pred, b_gt_n).sum(axis=summation_axis)
        correct_negatives = np.logical_and(b_pred_n, b_gt_n).sum(axis=summation_axis)

        a = hits
        b = false_alarms
        c = misses
        d = correct_negatives

        pod = a / (a + c)
        # pod = np.divide(a, (a+c), out=np.zeros_like(a), where=(a+c)!=0)
        far = b / (a + b)
        # far = np.divide(a, (a+b), out=np.zeros_like(a), where=(a+b)!=0)
        csi = a / (a + b + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref)
        hss = 2 * (a * d - b * c) / ((a + c) * (c + d) + (a + b) * (b + d))
        self.check(pod, a, b, c, d)
        self.check(far, a, b, c, d)
        self.check(csi, a, b, c, d)
        self.check(hss, a, b, c, d)
        pod[pod == np.inf] = 0
        pod = np.nan_to_num(pod)
        far[far == np.inf] = 0
        far = np.nan_to_num(far)
        csi[csi == np.inf] = 0
        csi = np.nan_to_num(csi)
        hss[hss == np.inf] = 0
        hss = np.nan_to_num(hss)
        return pod, far, csi, hss

    def check(self, data, a, b, c, d):
        nans = np.argwhere(np.isnan(data))
        infs = np.argwhere(np.isnan(data))
        if len(nans) != 0 or len(infs) != 0:
            print("fuck!")
            print(data.reshape(1, -1))
            print("hits", a, "far", b, "misses", c, "TF", d)

    def evaluate(self, gt, pred):
        self.total += 1
        for threshold in EVALUATION_THRESHOLDS:
            pod, far, csi, hss = self.get_metrics(gt, pred, threshold)
            self.metric[threshold]["pod"] += pod
            self.metric[threshold]["far"] += far
            self.metric[threshold]["csi"] += csi
            self.metric[threshold]["hss"] += hss

    def done(self):
        thresholds = EVALUATION_THRESHOLDS
        pods = []
        fars = []
        csis = []
        hsss = []
        save_path = self.save_path
        if not exists(save_path):
            makedirs(save_path)
        # draw line chart
        for threshold in thresholds:
            metrics = self.metric[threshold]
            pod = metrics["pod"].reshape(-1) / self.total
            pods.append(np.average(pod))
            far = metrics["far"].reshape(-1) / self.total
            fars.append(np.average(far))
            csi = metrics["csi"].reshape(-1) / self.total
            csis.append(np.average(csi))
            hss = metrics["hss"].reshape(-1) / self.total
            hsss.append(np.average(hss))

            x = list(range(len(pod)))
            plt.plot(x, pod, "r--", label='pod')
            plt.plot(x, far, "g--", label="far")
            plt.plot(x, csi, "b--", label="csi")
            plt.plot(x, hss, "k--", label="hss")
            for a, p, f, cs, h in zip(x, pod, far, csi, hss):
                plt.text(a, p + 0.005, "%.4f" % p, ha='center', va='bottom', fontsize=7)
                plt.text(a, f + 0.005, "%.4f" % f, ha='center', va='bottom', fontsize=7)
                plt.text(a, cs + 0.005, "%.4f" % cs, ha='center', va='bottom', fontsize=7)
                plt.text(a, h + 0.005, "%.4f" % h, ha='center', va='bottom', fontsize=7)

            plt.title(f"Threshold {threshold}")
            plt.xlabel("Time step")
            plt.ylabel("Rate")
            plt.legend()
            plt.gcf().set_size_inches(4.8 + (4.8 * self.seq // 10), 4.8)
            plt.savefig(join(save_path, f"{threshold}.jpg"))
            plt.clf()
        # draw bar chart
        x = np.array(range(len(thresholds)))
        total_width, n = 0.8, 4
        width = total_width / n
        plt.bar(x, pods, width=width, label='pod', fc='r')
        plt.bar(x + 0.2, fars, width=width, label='far', fc='g', tick_label=thresholds)
        plt.bar(x + 0.4, csis, width=width, label='csi', fc='b')
        plt.bar(x + 0.6, hsss, width=width, label='hss', fc='k')
        for a, p, f, cs, h in zip(x, pods, fars, csis, hsss):
            plt.text(a, p + 0.005, "%.4f" % p, ha='center', va='bottom', fontsize=7)
            plt.text(a + 0.2, f + 0.005, "%.4f" % f, ha='center', va='bottom', fontsize=7)
            plt.text(a + 0.4, cs + 0.005, "%.4f" % cs, ha='center', va='bottom', fontsize=7)
            plt.text(a + 0.6, h + 0.005, "%.4f" % h, ha='center', va='bottom', fontsize=7)
        plt.xlabel("Thresholds")
        plt.ylabel("Rate")
        plt.title(f"Average metrics")
        plt.legend()
        plt.gcf().set_size_inches(9.6, 4.8)
        plt.savefig(join(save_path, f"average.jpg"))
        plt.clf()



if __name__ == '__main__':
    # from cv2 import imread
    # # e = Evaluator(100)
    # gt = np.zeros((1, 10, 900, 900, 1))
    # pred = np.zeros((1, 10, 900, 900, 1))
    # for i in range(1, 11):
    #     gt[:, i-1, :,:,0] = imread(f'/extend/gru_tf_data/0409_small1/Valid/49/201803200006/out/{i}.png', 0)
    #     pred[:, i-1, :,:,0] = imread(f'/extend/gru_tf_data/0409_small1/Valid/49/201803200012/out/{i}.png', 0)
    #
    # print(ssim(pred,gt))
    # for i in range(1, 11):
    #     gt[:, i-1, :,:,0] = imread(f'/extend/results/gru_tf/3_99999_h/20180319232400/out/{i}.png', 0)
    #     pred[:, i-1, :,:,0] = imread(f'/extend/results/gru_tf/3_99999_h/20180319232400/pred/{i}.png', 0)
    # e.evaluate(gt, pred)
    # e.done()
    import pandas as pd
    import os
    import numpy as np
    from PIL import Image

    path='/extend/gru_tf_data/5l_stlstm_incept_ebgan/Test/199999'
    length=10
    or_metric_path=os.path.join(path,'metrics-{}'.format(length))

    evaluator = Evaluator(or_metric_path,seq=length)
    if not os.path.exists(or_metric_path):
        os.mkdir(or_metric_path)
    time=['201801010000','201809302300']
    time_range=pd.date_range(start=time[0],end=time[1],freq='6min')
    for i in range(len(time_range)):
        date = time_range[i].strftime("%Y%m%d%H%M")
        save_path = os.path.join(path, date)
        if os.path.exists(save_path):
            print(save_path)
            out = np.zeros((1, length, 700, 900, 1))
            pred = np.zeros((1, length, 700, 900, 1))
            for out_png in range(1, length + 1):
                out[0, out_png - 1, :, 2:-2, 0] = np.array(
                    Image.open(save_path + '/out/{}.png'.format(out_png)).convert("L"))[98:-98, :]
            for pred_png in range(1, length + 1):
                pred[0, pred_png - 1, :, :, 0] = np.array(
                    Image.open(save_path + '/pred/{}.png'.format(pred_png)).convert("L"))[100:-100, :]
            evaluator.evaluate(out, pred)
    evaluator.done()
