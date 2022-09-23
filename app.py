import utils
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny
from scipy.stats import mode
import tensorflow as tf
import easyocr

@st.cache
def init():

  DESKEW       = utils.Deskew()
  DENOISE      = utils.Denoiser("denoise_mdl_Final2")
  SEGMENTOR    = utils.Segment()
  TEXT_REMOVER = utils.Text_remover()

  return DESKEW, DENOISE, SEGMENTOR, TEXT_REMOVER

DESKEW, DENOISE, SEGMENTOR, TEXT_REMOVER = init()

DS = st.checkbox('Deskew')
DN = st.checkbox('Denoise')
SG = st.checkbox('Segment')
TR = st.checkbox('Image remover')
TR2 = st.checkbox('Better Image remover')

def run ():
  if DN or TR2:
    st.write("Activating denoise")
    new_img = DENOISE.denoise(im)
  else:
    new_img = im

  if DS:  
    st.write("Activating deskew")
    new_img = DESKEW.deskew(new_img)
  
  if TR:  
    st.write("Activating image remover")
    new_img = TEXT_REMOVER.extract_text(new_img)

  if SG:  
    st.write("Activating segmentation")
    new_imgs = SEGMENTOR.segment_lines(new_img)

  try:
    for I in new_imgs:
      st.image(I)
  except:
    st.image(new_img)

def load_image(image_file,save=False):
  img = Image.open(image_file)
  if save:
    img.save("Train_pipeline_results/Input/test.jpg")
  else:
    pass
  try:
    return np.array(img)[:,:,0]
  except:
    return np.array(img)
		
st.write('Please upload an image and choose from the checkboxes')
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
  im = load_image(image_file)
  st.image(im)

  st.button(label="Start Processing", on_click=run)
