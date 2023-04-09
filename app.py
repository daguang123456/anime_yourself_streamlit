import streamlit as st
# from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


import os
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.utils as vutils

from model import Generator


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# def load_image(image_path, x32=False):
#     img = Image.open(image_path).convert("RGB")

#     if x32:
#         def to_32s(x):
#             return 256 if x < 256 else x - x % 32
#         w, h = img.size
#         img = img.resize((to_32s(w), to_32s(h)))

#     return img


# def test(args):
#     device = args.device
    
#     net = Generator()
#     net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
#     net.to(device).eval()
#     print(f"model loaded: {args.checkpoint}")
    
#     os.makedirs(args.output_dir, exist_ok=True)

#     for image_name in sorted(os.listdir(args.input_dir)):
#         if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
#             continue
            
#         image = load_image(os.path.join(args.input_dir, image_name), args.x32)

#         with torch.no_grad():
#             image = to_tensor(image).unsqueeze(0) * 2 - 1
#             out = net(image.to(device), args.upsample_align).cpu()
#             out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
#             out = to_pil_image(out)

#         out.save(os.path.join(args.output_dir, image_name))
#         print(f"image saved: {image_name}")


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--checkpoint',
#         type=str,
#         default='./weights/paprika.pt',
#     )
#     parser.add_argument(
#         '--input_dir', 
#         type=str, 
#         default='./samples/inputs',
#     )
#     parser.add_argument(
#         '--output_dir', 
#         type=str, 
#         default='./samples/results',
#     )
#     parser.add_argument(
#         '--device',
#         type=str,
#         default='cuda:0',
#     )
#     parser.add_argument(
#         '--upsample_align',
#         type=bool,
#         default=False,
#         help="Align corners in decoder upsampling layers"
#     )
#     parser.add_argument(
#         '--x32',
#         action="store_true",
#         help="Resize images to multiple of 32"
#     )
#     args = parser.parse_args()
    
#     test(args)

with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'欢迎 *{name}*')
    st.title("动漫自己AnimeGanV2")

    
    st.write("教程[link](https://github.com/bryandlee/animegan2-pytorch")

    SIZES = ('paprika', 'celeba_distill', 'face_paintv1','face_paintv2')
    sizz = st.select_slider("大小", options=(['paprika', 'celeba_distill', 'face_paintv1','face_paintv2']))

    st.text("上传图片")



    device = "cpu"
    net = Generator()
    modelpath = "weights/"+sizz+".pt"
    net.load_state_dict(torch.load(modelpath, map_location="cpu"))
    net.to(device).eval()

    uploaded_file = st.file_uploader("选择..", type=["jpg","png","jpeg"])
    if uploaded_file is not None:


        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='上传了图片。', use_column_width=True)
        with st.spinner("请稍候..."):
            with torch.no_grad():
                image = to_tensor(image).unsqueeze(0) * 2 - 1
                out = net(image.to(device), False).cpu()
                out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
                # out = to_pil_image(out)
                fig1 = plt.figure(figsize=(14,8))

                fig1.suptitle("随机生成的动漫脸")
                plt.imshow(np.transpose(vutils.make_grid(out, padding=2, normalize=True), (1, 2, 0)))  
                st.pyplot(fig1)

        # out.save(os.path.join(args.output_dir, image_name))
        # print(f"image saved: {image_name}")


elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')



