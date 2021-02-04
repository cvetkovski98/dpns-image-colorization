import cv2
import numpy as np


def colorize_with_caffee(image_path):
    W_in = 224
    H_in = 224

    # Select desired model
    net = cv2.dnn.readNetFromCaffe('./resources/colorization_deploy_v2.prototxt',
                                   './resources/colorization_release_v2.caffemodel')

    pts_in_hull = np.load('./resources/pts_in_hull.npy')  # load cluster centers

    # populate cluster centers as 1x1 convolution kernel
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    frame = cv2.imread(image_path)

    img_rgb = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_l = img_lab[:, :, 0]  # pull out L channel
    (H_orig, W_orig) = img_rgb.shape[:2]  # original image size

    # resize image to network input size
    img_rs = cv2.resize(img_rgb, (W_in, H_in))  # resize image to network input size
    img_lab_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2Lab)
    img_l_rs = img_lab_rs[:, :, 0]
    img_l_rs -= 50  # subtract 50 for mean-centering

    net.setInput(cv2.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))  # this is our result

    ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)  # concatenate with original image L
    img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

    return cv2.cvtColor(img_bgr_out, cv2.COLOR_BGR2RGB)
