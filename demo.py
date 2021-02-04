import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from colorize import *
from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str)
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

my_model = load_model('models/model.h5')
my_model.load_weights('models/model_weights.h5')

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

orig = img_to_array(load_img(opt.img_path))
l_orig = rgb2lab(orig)[:, :, 0]
test = resize(orig, (224, 224), anti_aliasing=True)
test *= 1.0 / 255
lab = rgb2lab(test)
l = lab[:, :, 0]
L = gray2rgb(l)
L = L.reshape((1, 224, 224, 3))
ab = my_model.predict(L)
ab = ab * 128
cur = np.zeros((224, 224, 3))
cur[:, :, 0] = l
cur[:, :, 1:] = ab
fin = lab2rgb(cur)
fin = tf.image.resize(fin, l_orig.shape[:2], method='bilinear')

caffe_colorized = colorize_with_caffee(opt.img_path)
caffe_colorized = tf.image.resize(caffe_colorized, l_orig.shape[:2], method='bilinear')

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(fin)
plt.title('Output (VGG16 encoder)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(caffe_colorized)
plt.title('Output (Caffee model)')
plt.axis('off')
plt.show()
