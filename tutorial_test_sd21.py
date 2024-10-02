from share import *
import config
import json
import os
import matplotlib.pyplot as plt
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from tutorial_dataset_bev import MyDataset
from torch.utils.data import DataLoader


model = create_model('/home/zc/Diffusion/src/config/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('/home/zc/Downloads/epoch=98-step=7028.ckpt', location='cuda'))
model = model.cuda()
# model = model.half()
ddim_sampler = DDIMSampler(model)

# Misc
dataset = MyDataset(data_split='test')
dataloader = DataLoader(dataset, num_workers=0, batch_size=5, shuffle=False)


def process(bev_feat, prompt, a_prompt, n_prompt, 
            num_samples, ddim_steps, guess_mode, 
            strength, scale, seed, eta):
    with torch.no_grad():
        # H, W, C = img.shape
        H, W = 128, 128
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {"c_concat": [bev_feat], "c_crossattn": [model.get_learned_conditioning(prompt)]}
        un_cond = {"c_concat": [bev_feat], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 2, W // 2)


        model.control_scales = [strength] * 13
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)


        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    # return [255 - detected_map] + results
    return results


def combine_img(gt_tensor, pred_imgs):

    # Restore gt bev images from tensor
    gt_imgs = []
    for i in range(gt_tensor.size(0)):  # 遍历batch
        gt_img = gt_tensor[i].numpy()  # 将torch tensor转换为numpy数组
        gt_img = ((gt_img + 1.0) * 127.5).astype(np.uint8)
        gt_imgs.append(gt_img)

    combined_images = []
    gap_size=10
    height, width, channels = 512, 512, 3
    
    for gt, pred in zip(gt_imgs, pred_imgs):
        # 上下拼接 GT 和预测图片，并加上中间的垂直空隙
        combined = np.vstack((gt, np.ones((gap_size, width, channels), dtype=np.uint8) * 255, pred))
        combined_images.append(combined)

    return combined_images


def main():
    out_folder = "inference_result_sd_lock"
    if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    idx = 0
    for i, data in enumerate(dataloader):
        # source = data['hint']
        source = einops.rearrange(data['hint'], 'b h w c-> b c h w')
        source = source.float()
        target = data['jpg']
        prompt = data['txt']

        pred_img_list = process(
                bev_feat = source,
                prompt = prompt,
                a_prompt = '',
                n_prompt = '',
                num_samples = 5,
                ddim_steps = 20,
                guess_mode = False,
                strength = 1,
                scale = 9.0,
                seed = -1,
                eta = 0.0,
            )
        result_img = combine_img(target, pred_img_list)
        for img in result_img:
            save_path = os.path.join(out_folder, f"{idx:04d}.jpg")
            # img.save(save_path)
            cv2.imwrite(save_path, img)
            idx += 1


if __name__=="__main__":
    main()
# block = gr.Blocks().queue()
# with block:
#     with gr.Row():
#         gr.Markdown("## Control Stable Diffusion with Canny Edge Maps")
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(source='upload', type="numpy")
#             prompt = gr.Textbox(label="Prompt")
#             run_button = gr.Button(label="Run")
#             with gr.Accordion("Advanced options", open=False):
#                 num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
#                 image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
#                 strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
#                 guess_mode = gr.Checkbox(label='Guess Mode', value=False)
#                 low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
#                 high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
#                 ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
#                 scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
#                 seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
#                 eta = gr.Number(label="eta (DDIM)", value=0.0)
#                 a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
#                 n_prompt = gr.Textbox(label="Negative Prompt",
#                                       value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
#         with gr.Column():
#             result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
#     ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold]
#     run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


# block.launch(server_name='0.0.0.0')
