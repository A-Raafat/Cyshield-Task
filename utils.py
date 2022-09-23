import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny
from scipy.stats import mode
import tensorflow as tf
import easyocr

IMG_SIZE = (420, 540)

class Denoiser():
  def __init__(self, model_path):
    self.model = tf.saved_model.load(model_path)
    
  def process_image(self,im):
    img = np.asarray(im, dtype="float32")
    img = cv2.resize(img, IMG_SIZE[::-1])
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (*IMG_SIZE, 1))
    
    return img, im.shape

  def denoise(self, image):
    input_im, img_shape = self.process_image(image)
    decoded_img = np.squeeze(self.model.decoder(self.model.encoder(np.expand_dims(input_im,0)).numpy()).numpy())
    preds_reshaped = cv2.resize(decoded_img, (img_shape[1], img_shape[0]))*255
    preds_reshaped = preds_reshaped.astype('uint8')
    preds_reshaped[preds_reshaped<100]=0
    #f= cv2.detailEnhance(preds_reshaped, sigma_s=10, sigma_r=0.15)
    return preds_reshaped

class Deskew():
  def __init__(self):
    pass

  def deskew(self,image):
    # convert to edges
    edges = canny(image.copy())
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    new_img = rotate(image, skew_angle, cval=1)*255
    return new_img.astype("uint8")

class Segment():
  def __init__(self):
    pass

  def segment_lines(self,img):
    im = img.copy()
    im_Z = cv2.bitwise_not(img.copy())

    h, w = im_Z.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2*w, 1))
    img_dilation = cv2.cvtColor(im_Z,cv2.COLOR_GRAY2RGB)
    img_dilation = cv2.dilate(img_dilation, kernel)

    bordersize = 10
    border = cv2.copyMakeBorder(
        img_dilation,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    new_im = cv2.copyMakeBorder(
        im,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    imgray = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    imgs_list = []
    for c in contours:
        rect = cv2.boundingRect(c)
        if cv2.contourArea(c) < 1000: continue
        x,y,w,h = rect
        imgs_list.append(new_im[y-3:y+h+3, x:x+w])

    return imgs_list[::-1]

  def display_lines(self, ll):
    ll = ll[::-1]
    for idx, img in enumerate(ll):
      plt.figure(figsize=(15,10))
      plt.subplot(len(ll),1,idx+1)
      plt.grid(False)
      plt.title(f"Line {idx+1}")
      plt.imshow(img, 'gray')

class Text_remover():
  def __init__(self):
    self.reader = easyocr.Reader(['en'])
     
  def extract_text(self,image):
    mask = image.copy()*0
    print(mask.shape)
    print(image.shape)
    print(type(mask))
    print(type(image))
    res = self.reader.readtext(image) 
    for (bbox, text, prob) in res: 
      # unpack the bounding box
      (tl, tr, br, bl) = bbox
      tl = (int(tl[0]), int(tl[1]))
      tr = (int(tr[0]), int(tr[1]))
      br = (int(br[0]), int(br[1]))
      bl = (int(bl[0]), int(bl[1]))
      cv2.rectangle(mask, tl, br, (255, 255, 255), -1)
    txt = cv2.bitwise_and(image, image, mask=mask)
    h,w = txt.shape
    vert_S = np.vstack([txt, np.zeros((2,w))])
    Final_mask = np.hstack([vert_S, np.zeros((h+2,2))]).astype("uint8")

    filling_bg = cv2.floodFill(txt, Final_mask, (0,0),255)[1]

    return filling_bg
