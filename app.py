import streamlit as st 
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model

def get_opened_image(image):
    return Image.open(image).convert('RGB')

def _loadmodel():
    return load_model('./softmax_rms_new.h5')


def difference(org):
    # filename = path
    # # print(path)
    # resaved_name = filename.split('.')[-2]+'_resaved.jpg'
    # # print(resaved_name)
    # resaved_name = resaved_name.split('/')[-1]
    # org = Image.open(filename).convert('RGB')
    resaved_name = 'temp.jpg'
    org.save(resaved_name, 'JPEG', quality=92)
    resaved = Image.open(resaved_name)
    diff = ImageChops.difference(org, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    diff = ImageEnhance.Brightness(diff).enhance(scale)
    # diff
    return diff

def pred(img):
    model = _loadmodel()
    diff = np.array(difference(img).resize((128, 128))).flatten()/255.0
    diff = diff.reshape(-1, 128, 128, 3)
    pred= model.predict(diff)[0]
    print("================= pred ==================")
    print(pred)
    if pred[0] > pred[1]:
        return "Not Forged"
    else:
        return 'Forged'


st.sidebar.title("Image Forgery Detection")

image_file = st.sidebar.file_uploader('Upload an image', type='jpg')
image =""

if image_file and st.sidebar.button('Load'):
    image = get_opened_image(image_file)
    with st.expander('Selected Image', expanded=True):
        st.image(image, use_column_width=True)
        prediction = pred(image)

        st.subheader('Prediction')
        st.markdown(f'The predicted label is: **{prediction}**')