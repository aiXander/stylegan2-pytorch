import argparse, time
import torch
import cv2
from torchvision import utils
from model import Generator
from tqdm import tqdm
import numpy as np
from collections import deque

'''

cd /home/rednax/Desktop/GitHub_Projects/stylegan2-pytorch
python3 simple_generate.py

'''

def load_pytorch_generator(path, img_res = 1024, _N = 512, n_mapping_layers = 8):
    g_ema = Generator(img_res, _N, n_mapping_layers, channel_multiplier=2).to('cuda')
    checkpoint = torch.load(path)
    g_ema.load_state_dict(checkpoint['g_ema'])
    return g_ema

def postprocess_tensor(tensor):
    tensor = tensor.squeeze()            
    tensor.clamp_(min=-1, max=1)
    tensor.add_(1).div_(1 + 1 + 1e-5)
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)

def save_cuda_tensor_as_img(tensor):
    img = tensor.to('cpu', torch.uint8).numpy()
    cv2.imwrite('sample/test.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def sample_random(g_ema, _N = 512, truncation=1):

    '''
    if transform_dict['layerID'] == self.layerID:
        out = self.layer_options[transform_dict['transformID']](out, transform_dict['params'], transform_dict['indicies'])
    '''

    horizontal_flip = {}
    horizontal_flip['layerID'] = 2
    horizontal_flip['transformID'] = 'translate'
    horizontal_flip['params'] = [0.0]
    horizontal_flip['indicies'] = range(0, 512)

    transform_dict_list = [horizontal_flip]

    with torch.no_grad():
        if truncation < 1: mean_latent = g_ema.mean_latent(4096)
        else: mean_latent = None

        g_ema.eval()
        sample_z = torch.randn(1, _N, device='cuda')
        sample, _ = g_ema(sample_z, truncation=truncation, truncation_latent=mean_latent, transform_dict_list=transform_dict_list)
        sample = postprocess_tensor(sample)
        save_cuda_tensor_as_img(sample)


path = "/home/rednax/Desktop/music_vr/Great_Models/PyTorch_Models/stylegan2-ffhq-config-f.pt"
torch.manual_seed(1)

g_ema = load_pytorch_generator(path)

print_f = 25
check_time = time.time()

for i in range(1):
    sample_random(g_ema)

    if i%print_f == 0:
        fps = print_f / (time.time() - check_time)
        print("Running at %.2f fps" %fps)
        check_time = time.time()
