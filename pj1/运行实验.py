"""Project 1: reproducible ultrasound contrast enhancement and denoising.

Run: python 运行实验.py
Train again: python 运行实验.py --retrain --steps 1800
The source images are never overwritten. No network access is required.
"""
from __future__ import annotations

import os
# Some Windows Anaconda installations preload two Intel OpenMP runtimes.
# Keep this opt-in; set KMP_DUPLICATE_LIB_OK=TRUE externally if needed.
import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from skimage.metrics import structural_similarity
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / '原始数据'
OUT = ROOT / '实验结果'
FIG = OUT / '图表'
IMG = OUT / '处理图像'
MODELS = ROOT / '模型'
FIGURE_NAMES = {
    'contrast_high.png':'高端设备处理对比.png', 'contrast_low.png':'低端设备处理对比.png',
    'histograms_high.png':'高端设备灰度直方图与累计分布.png',
    'histograms_low.png':'低端设备灰度直方图与累计分布.png',
    'clahe_parameters_high.png':'高端设备CLAHE参数对比.png',
    'clahe_parameters_low.png':'低端设备CLAHE参数对比.png',
    'original_histograms.png':'原始灰度直方图.png', 'spatial_split.png':'训练验证测试空间划分.png',
    'training_curve.png':'训练验证曲线.png', 'synthetic_comparison.png':'人工噪声降噪结果对比.png',
    'synthetic_test_crop.png':'独立测试区域局部放大.png', 'synthetic_errors.png':'人工噪声绝对误差图.png',
    'real_comparison.png':'真实噪声降噪结果对比.png', 'real_crop.png':'真实噪声局部放大.png',
}
IMAGE_NAMES = {
    'evaluation_test_mask.png':'独立测试区域掩膜.png', 'evaluation_fan_mask.png':'超声扇形区域掩膜.png',
    'synthetic_noisy_sigma15.png':'人工加噪标准差15.png', 'synthetic_gaussian.png':'高斯滤波结果.png',
    'synthetic_cnn.png':'残差卷积网络降噪结果.png', 'real_original.png':'真实含噪原图.png',
    'real_gaussian.png':'真实图高斯滤波结果.png', 'real_cnn.png':'真实图卷积网络降噪结果.png',
    'high_0.png':'高端设备增强_原图.png', 'high_1.png':'高端设备增强_全局均衡化.png',
    'high_2.png':'高端设备增强_直方图规定化.png', 'high_3.png':'高端设备增强_CLAHE.png',
    'low_0.png':'低端设备增强_原图.png', 'low_1.png':'低端设备增强_全局均衡化.png',
    'low_2.png':'低端设备增强_直方图规定化.png', 'low_3.png':'低端设备增强_CLAHE.png',
}
for directory in [OUT, FIG, IMG, MODELS]:
    directory.mkdir(parents=True, exist_ok=True)

for font in ['Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']:
    if any(f.name == font for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [font]
        break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                     'figure.dpi': 140, 'savefig.dpi': 180})


def load_gray(path):
    """Composite alpha over black before converting to 8-bit luminance."""
    image = Image.open(path)
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image).convert('RGB')
    return np.asarray(image.convert('L'), dtype=np.uint8)


def save_image(name, array):
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(IMG / IMAGE_NAMES.get(name, name))


def localize_result_names():
    for directory, mapping in [(FIG,FIGURE_NAMES),(IMG,IMAGE_NAMES)]:
        for source,destination in mapping.items():
            old,new=directory/source,directory/destination
            if old.exists() and old != new:
                old.replace(new)


def hist(array):
    return np.bincount(array.ravel(), minlength=256)


def hist_stats(array):
    h = hist(array).astype(float)
    p = h[h > 0] / array.size
    return {'mean': float(array.mean()), 'std': float(array.std()),
            'p05': float(np.percentile(array, 5)), 'median': float(np.median(array)),
            'p95': float(np.percentile(array, 95)), 'min': int(array.min()),
            'max': int(array.max()), 'entropy_bits': float(-(p*np.log2(p)).sum()),
            'occupied_bins': int((h > 0).sum()),
            'endpoint_pct': float(((array == 0) | (array == 255)).mean()*100)}


def match_histogram(source, reference):
    """Monotone nearest-CDF mapping on occupied reference gray levels.

    A discrete image cannot generally reproduce every reference bin exactly.
    Mapping to occupied bins avoids falsely mapping CDF=0 to an absent gray.
    """
    src_values, inverse, src_counts = np.unique(source, return_inverse=True, return_counts=True)
    ref_values, ref_counts = np.unique(reference, return_counts=True)
    scdf = np.cumsum(src_counts)/source.size
    rcdf = np.cumsum(ref_counts)/reference.size
    right = np.searchsorted(rcdf, scdf, side='left').clip(0, len(rcdf)-1)
    left = np.maximum(right-1, 0)
    chosen = np.where(abs(rcdf[left]-scdf) <= abs(rcdf[right]-scdf), left, right)
    lut_values = ref_values[chosen]
    return lut_values[inverse].reshape(source.shape).astype(np.uint8)


def make_figure(name, labels, arrays, cols=4, crops=None):
    rows = math.ceil(len(arrays)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0*cols, 3.25*rows), squeeze=False)
    for ax, label, array in zip(axes.ravel(), labels, arrays):
        ax.imshow(array, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(label, fontsize=11)
        ax.axis('off')
    for ax in axes.ravel()[len(arrays):]:
        ax.axis('off')
    fig.tight_layout(pad=1)
    fig.savefig(FIG / name, bbox_inches='tight')
    plt.close(fig)


def contrast_experiment(images):
    rows = []
    for key, image in images.items():
        refkey = 'low' if key == 'high' else 'high'
        variants = {'原图': image, '全局均衡化': cv2.equalizeHist(image),
                    '规定化到另一图': match_histogram(image, images[refkey]),
                    'CLAHE': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(image)}
        make_figure(f'contrast_{key}.png', list(variants), list(variants.values()))
        fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
        for col, (method, result) in enumerate(variants.items()):
            save_image(f'{key}_{col}.png', result)
            h = hist(result)/result.size
            axes[0,col].plot(np.arange(256), h, lw=1, color='#23678e')
            axes[0,col].set_title(method)
            axes[0,col].set_xlim(0,255)
            axes[0,col].set_xlabel('灰度值')
            axes[0,col].set_ylabel('像素比例')
            axes[1,col].plot(np.arange(256), np.cumsum(h), lw=1.4, color='#23678e')
            axes[1,col].set_xlim(0,255)
            axes[1,col].set_ylim(0,1.01)
            axes[1,col].set_xlabel('灰度值')
            axes[1,col].set_ylabel('累计比例')
            rows.append({'image':key, 'method':method, **hist_stats(result)})
        fig.tight_layout()
        fig.savefig(FIG / f'histograms_{key}.png', bbox_inches='tight')
        plt.close(fig)
        sweep = [(1,(4,4)),(2,(4,4)),(3,(4,4)),(2,(8,8))]
        make_figure(f'clahe_parameters_{key}.png',
                    [f'clip={c}, grid={g[0]}×{g[1]}' for c,g in sweep],
                    [cv2.createCLAHE(clipLimit=c,tileGridSize=g).apply(image) for c,g in sweep])
    fig, ax = plt.subplots(figsize=(8,3.1))
    for key, image in images.items():
        ax.plot(np.arange(256), hist(image)/image.size, label=key, alpha=.9)
    ax.set(xlim=(0,255), xlabel='灰度值', ylabel='像素比例', title='两幅原始图像的灰度直方图')
    ax.legend(); fig.tight_layout(); fig.savefig(FIG/'original_histograms.png'); plt.close(fig)
    write_csv(OUT/'增强指标.csv', rows)
    return rows


class ResidualCNN(nn.Module):
    """8-layer residual denoiser, 32 features; no batch normalization."""
    def __init__(self, channels=32, depth=8):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels,1,3,padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, noisy):
        return self.net(noisy)  # predicted additive residual; clean = noisy - residual


TRAIN_TILES = [(0,0),(0,1),(0,2),(1,0),(2,1),(2,2)]
VAL_TILES = [(1,2)]
TEST_TILES = [(1,1),(2,0)]
MARGIN = 12  # greater than CNN receptive-field radius 8 and SSIM radius 3


def tile_masks(shape):
    h,w = shape
    ys, xs = np.linspace(0,h,4,dtype=int), np.linspace(0,w,4,dtype=int)
    masks = {}
    for name, tiles in [('train',TRAIN_TILES),('validation',VAL_TILES),('test',TEST_TILES)]:
        mask = np.zeros(shape, bool)
        for r,c in tiles:
            y0,y1,x0,x1 = ys[r],ys[r+1],xs[c],xs[c+1]
            if name != 'train':
                y0+=MARGIN;y1-=MARGIN;x0+=MARGIN;x1-=MARGIN
            mask[y0:y1,x0:x1] = True
        masks[name] = mask
    return masks,ys,xs


def fan_mask(image):
    binary = np.uint8(image > 8)*255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n < 2:
        raise RuntimeError('Ultrasound support could not be identified.')
    component = np.uint8(labels == 1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))*255
    contours,_ = cv2.findContours(component, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(contours))
    mask = np.zeros_like(image); cv2.fillConvexPoly(mask,hull,255)
    return cv2.erode(mask,np.ones((7,7),np.uint8)) > 0


def add_noise(image, seed, sigma=15):
    noise = np.random.default_rng(seed).normal(0,sigma,image.shape)
    return np.clip(np.rint(image.astype(np.float32)+noise),0,255).astype(np.uint8),noise


def predict(model, image, device):
    tensor = torch.from_numpy(np.array(image,copy=True,order='C')).float().to(device)[None,None]/255
    tensor = nn.functional.pad(tensor,(12,12,12,12),mode='reflect')
    with torch.inference_mode():
        result = (tensor-model(tensor))[:,:,12:-12,12:-12].clamp(0,1)
    return (result[0,0].cpu().numpy()*255).round().astype(np.uint8)


def train_or_load(clean, args, device):
    checkpoint = MODELS/'残差卷积网络权重.pt'
    model = ResidualCNN().to(device)
    if checkpoint.exists() and not args.retrain:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state['training']['seed'] != args.seed:
            raise ValueError('Saved training seed differs from --seed; use --retrain to make a matching checkpoint.')
        model.load_state_dict(state['state_dict']); model.eval()
        print('Loaded saved weights.',flush=True)
        return model,state['training']
    _,ys,xs = tile_masks(clean.shape)
    tiles = [clean[ys[r]:ys[r+1],xs[c]:xs[c+1]].astype(np.float32)/255 for r,c in TRAIN_TILES]
    rng = np.random.default_rng(args.seed+100)
    val_noisy,_ = add_noise(clean,args.seed+1000)
    # Validation is performed only on its held-out tile, including local context
    # from that same tile. This never uses test scores for model selection.
    r,c = VAL_TILES[0]
    val_x = val_noisy[ys[r]:ys[r+1],xs[c]:xs[c+1]]
    val_y = clean[ys[r]:ys[r+1],xs[c]:xs[c+1]]
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                  milestones=[int(args.steps*.6),int(args.steps*.85)], gamma=.3)
    best = -float('inf'); best_state=None; best_step=0; history=[]
    start = time.perf_counter()
    model.train()
    for step in range(1,args.steps+1):
        patches=[]
        for _ in range(16):
            # Sample only within a training tile; no patch crosses tile boundaries.
            tile=tiles[int(rng.integers(len(tiles)))]
            y=int(rng.integers(0,tile.shape[0]-48+1));x=int(rng.integers(0,tile.shape[1]-48+1))
            patch=tile[y:y+48,x:x+48]
            patch=np.rot90(patch,int(rng.integers(4)))
            if rng.random()<.5:patch=np.fliplr(patch)
            patches.append(patch.copy())
        target=torch.from_numpy(np.stack(patches)[:,None]).to(device)
        noisy=(target+torch.randn_like(target)*(15/255)).clamp(0,1)
        residual=model(noisy)
        # Ignore artificial patch boundary context during training.
        loss=nn.functional.mse_loss(residual[:,:,8:-8,8:-8],(noisy-target)[:,:,8:-8,8:-8])
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();scheduler.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            model.eval();den=predict(model,val_x,device)
            a=den[MARGIN:-MARGIN,MARGIN:-MARGIN].astype(float)
            b=val_y[MARGIN:-MARGIN,MARGIN:-MARGIN].astype(float)
            score=float(10*np.log10(255**2/np.mean((a-b)**2)))
            history.append({'step':step,'train_loss':float(loss.item()),'validation_psnr':score})
            if score>best:
                best=score;best_step=step
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f'Train {step}/{args.steps}: loss={loss.item():.6f}, validation PSNR={score:.3f} dB',flush=True)
            model.train()
    elapsed=time.perf_counter()-start
    metadata={'seed':args.seed,'steps':args.steps,'best_step':best_step,'best_validation_psnr':best,
              'training_seconds':elapsed,'parameters':sum(p.numel() for p in model.parameters()),
              'device':str(device),'device_name':torch.cuda.get_device_name(0) if device.type=='cuda' else platform.processor(),
              'architecture':'Residual CNN 8 conv layers, 32 features, ReLU, no BN',
              'patch_size':48,'batch_size':16,'sigma':15,'train_tiles':TRAIN_TILES,
              'validation_tiles':VAL_TILES,'test_tiles':TEST_TILES,'evaluation_border':MARGIN,
              'history':history}
    torch.save({'state_dict':best_state,'training':metadata},checkpoint)
    model.load_state_dict(best_state);model.eval()
    return model,metadata


def quality(clean, candidate, mask):
    error=candidate.astype(float)-clean.astype(float)
    mse=float(np.mean(error[mask]**2))
    _, ssim_map=structural_similarity(clean,candidate,data_range=255,full=True)
    return {'PSNR_dB':float(10*np.log10(255**2/mse)) if mse else 100.,
            'SSIM':float(ssim_map[mask].mean()),'MAE':float(np.abs(error[mask]).mean()),
            'RMSE':float(np.sqrt(mse)),'pixels':int(mask.sum())}


def write_csv(path, rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def denoising_experiment(clean, real, args, device):
    masks,ys,xs=tile_masks(clean.shape)
    assert not np.any(masks['train'] & masks['test'])
    assert not np.any(masks['validation'] & masks['test'])
    support=fan_mask(clean)
    save_image('evaluation_test_mask.png',masks['test'].astype(np.uint8)*255)
    save_image('evaluation_fan_mask.png',support.astype(np.uint8)*255)
    fig,ax=plt.subplots(figsize=(6,6))
    ax.imshow(clean,cmap='gray',vmin=0,vmax=255)
    colors={'训练':'#72c18e','验证':'#edb24c','测试':'#e66d64'}
    for label,tiles in [('训练',TRAIN_TILES),('验证',VAL_TILES),('测试',TEST_TILES)]:
        for r,c in tiles:
            ax.add_patch(plt.Rectangle((xs[c],ys[r]),xs[c+1]-xs[c],ys[r+1]-ys[r],
                         fill=False,edgecolor=colors[label],linewidth=2.2))
            ax.text((xs[c]+xs[c+1])/2,(ys[r]+ys[r+1])/2,label,color=colors[label],
                    ha='center',va='center',fontsize=14,backgroundcolor='black')
    ax.axis('off');fig.tight_layout();fig.savefig(FIG/'spatial_split.png');plt.close(fig)
    synthetic,raw_noise=add_noise(clean,args.seed)
    validation_noisy,_=add_noise(clean,args.seed+1000)
    # Compare a small, specified parameter grid on validation pixels only.
    sweep=[]
    for sigma in [.5,.7,1.,1.2,1.5]:
        k=2*math.ceil(3*sigma)+1
        den=cv2.GaussianBlur(validation_noisy,(k,k),sigmaX=sigma,borderType=cv2.BORDER_REFLECT_101)
        sweep.append({'sigma':sigma,'kernel':k,**quality(clean,den,masks['validation'])})
    selected=max(sweep,key=lambda row:row['PSNR_dB'])
    model,training=train_or_load(clean,args,device)
    k=selected['kernel'];sigma=selected['sigma']
    def gaussian(image):
        return cv2.GaussianBlur(image,(k,k),sigmaX=sigma,borderType=cv2.BORDER_REFLECT_101)
    methods={'人工加噪':synthetic,'高斯滤波':gaussian(synthetic),'残差 CNN':predict(model,synthetic,device)}
    rows=[]
    allmask=np.ones(clean.shape,bool);allmask[:5,:]=False;allmask[-5:,:]=False;allmask[:,:5]=False;allmask[:,-5:]=False
    for name,array in methods.items():
        for scope,mask in [('独立测试区域',masks['test']),('测试区域内超声扇形',masks['test']&support),
                           ('全图内区 仅作补充',allmask)]:
            rows.append({'method':name,'scope':scope,**quality(clean,array,mask)})
    save_image('synthetic_noisy_sigma15.png',synthetic)
    save_image('synthetic_gaussian.png',methods['高斯滤波'])
    save_image('synthetic_cnn.png',methods['残差 CNN'])
    labels=['参考干净图','人工加噪 σ=15','高斯滤波','残差 CNN']
    arrays=[clean,*methods.values()]
    make_figure('synthetic_comparison.png',labels,arrays)
    make_figure('synthetic_test_crop.png',labels,[a[215:380,215:380] for a in arrays])
    errors=[np.abs(a.astype(float)-clean.astype(float)) for a in methods.values()]
    fig,axes=plt.subplots(1,3,figsize=(10,3.4))
    for ax,label,error in zip(axes,methods,errors):
        im=ax.imshow(error,cmap='inferno',vmin=0,vmax=40);ax.set_title(label+' 绝对误差');ax.axis('off')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='绝对灰度误差')
    fig.savefig(FIG/'synthetic_errors.png',bbox_inches='tight');plt.close(fig)
    real_methods={'真实含噪原图':real,'高斯滤波':gaussian(real),'残差 CNN':predict(model,real,device)}
    make_figure('real_comparison.png',list(real_methods),list(real_methods.values()),cols=3)
    make_figure('real_crop.png',list(real_methods),[a[55:185,75:195] for a in real_methods.values()],cols=3)
    real_rows=[]
    # A fixed rectangular inspection ROI avoids most external background and ECG.
    for name,array in real_methods.items():
        save_image('real_'+{'真实含噪原图':'original','高斯滤波':'gaussian','残差 CNN':'cnn'}[name]+'.png',array)
        patch=array[40:205,55:215].astype(float)
        grad=np.hypot(cv2.Sobel(patch,cv2.CV_64F,1,0,ksize=3)/8,
                      cv2.Sobel(patch,cv2.CV_64F,0,1,ksize=3)/8)
        residual=array[40:205,55:215].astype(float)-real[40:205,55:215].astype(float)
        real_rows.append({'method':name,'ROI_mean':float(patch.mean()),'ROI_std':float(patch.std()),
                          'mean_gradient':float(grad.mean()),'change_RMS':float(np.sqrt(np.mean(residual**2)))})
    history=training['history']
    fig,ax=plt.subplots(figsize=(7,3))
    ax.plot([x['step'] for x in history],[x['validation_psnr'] for x in history],marker='o',ms=3)
    ax.set(xlabel='训练迭代数',ylabel='验证 PSNR / dB',title='固定验证区域上的训练进度')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/'training_curve.png');plt.close(fig)
    write_csv(OUT/'人工噪声降噪指标.csv',rows)
    write_csv(OUT/'真实噪声描述统计.csv',real_rows)
    write_csv(OUT/'高斯参数验证.csv',sweep)
    write_csv(OUT/'训练记录.csv',history)
    return {'synthetic_metrics':rows,'real_descriptive':real_rows,'gaussian_validation':sweep,
            'gaussian_selected':selected,'training':training,
            'noise':{'requested_sigma':15,'raw_sample_std':float(raw_noise.std()),
                     'after_clipping_error_std':float((synthetic.astype(float)-clean).std()),
                     'after_clipping_error_mean':float((synthetic.astype(float)-clean).mean())},
            'mask_counts':{**{k:int(v.sum()) for k,v in masks.items()},'test_fan':int((masks['test']&support).sum())}}


def md_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def make_report(contrast, denoising, info):
    raw={r['image']:r for r in contrast if r['method']=='原图'}
    d=denoising
    t=d['training'];g=d['gaussian_selected']
    metric={(r['method'],r['scope']):r for r in d['synthetic_metrics']}
    cnn=metric[('残差 CNN','独立测试区域')];gf=metric[('高斯滤波','独立测试区域')];noisy=metric[('人工加噪','独立测试区域')]
    cfan=metric[('残差 CNN','测试区域内超声扇形')];gfan=metric[('高斯滤波','测试区域内超声扇形')]
    best='残差 CNN' if cnn['PSNR_dB']>gf['PSNR_dB'] else '高斯滤波'
    parts=[]
    def p(text):parts.append(text.strip())
    def image(name,caption):p(f'![{caption}](实验结果/图表/{FIGURE_NAMES.get(name,name)})\n\n{caption}')
    p('# 超声图像增强与降噪实验报告')
    p(f'本实验使用题目提供的 4 张图像，完成灰度直方图分析、全局均衡化、直方图规定化、CLAHE，以及高斯滤波与自行训练残差 CNN 的降噪比较。人工加噪 σ=15 时，独立空间测试区域中，{best}取得更高 PSNR；CNN 为 {cnn["PSNR_dB"]:.2f} dB，高斯滤波为 {gf["PSNR_dB"]:.2f} dB。真实噪声图没有配对干净参考，只作视觉与描述性分析。')
    p('## 1 数据与实验约定')
    p(md_table(['文件','尺寸','原始模式','用途'],[[x['file'],f"{x['width']}×{x['height']}",x['mode'],x['use']] for x in info['inputs']]))
    p('所有图像转为 0～255 的 8 位灰度；带透明通道的 PNG 先合成到黑底，原文件保持不变。Figure2_clean 的 alpha 实际全部为 255。增强实验不做缩放；降噪也分别保留 600×600 和 256×256 的原始像素尺寸。所有对比图统一显示范围 0～255，避免逐图自动拉伸造成误判。')
    p('Figure2_clean 是题目指定的“干净图”，它本身仍有超声组织纹理和斑点，实验中的真值仅指这一输入。Figure2_noisy 与它不是同尺寸、同内容的已配准图像，不能把前者和后者直接配对计算 PSNR 或 SSIM，也不能通过缩放来伪造配对。')
    p('## 2 灰度直方图与图像增强')
    p('### 2.1 原始灰度分布')
    p(md_table(['图像','均值','标准差','P5','中位数','P95','熵 bit','有效灰度级'],[[r['image'],f"{r['mean']:.2f}",f"{r['std']:.2f}",f"{r['p05']:.0f}",f"{r['median']:.0f}",f"{r['p95']:.0f}",f"{r['entropy_bits']:.3f}",r['occupied_bins']] for r in raw.values()]))
    image('original_histograms.png','图 1 两张原图的归一化灰度直方图')
    p(f"高端设备图的灰度均值为 {raw['high']['mean']:.2f}、标准差为 {raw['high']['std']:.2f}；低端设备图分别为 {raw['low']['mean']:.2f}、{raw['low']['std']:.2f}。结合图像可见，两图均有暗部和明显散斑纹理，但低端图的纹理更粗、更杂。这里的 high 与 low 来自题目标签，仅凭两张图的灰度统计不能证明设备性能，也不能把较高均值、标准差或熵等同于较高成像质量。")
    p('### 2.2 方法与参数')
    p('全局均衡化调用 cv2.equalizeHist，以整幅图的累计分布建立单调灰度映射。它扩大常见灰度区间的差异，但不能保证离散输出的直方图完全平坦，也可能抬高暗部斑点。')
    p('直方图规定化采用另一张未处理原图作为参考：high→low、low→high 分别处理。源图每个已出现灰度的累计概率映射到参考图累计概率最接近的已出现灰度级。参考图已经存在的结构不会被复制；该方法只对齐分布，不能修复局部结构或噪声。两图内容并非严格配准，故这是双向分布迁移实验，不把 high 的分布直接定义为“正确分布”。')
    p('CLAHE 使用 clipLimit=2.0、tileGridSize=(4,4)。在 150×150 图上每个局部网格约为 37～38 像素；该参数在局部增强与斑点放大之间折中。另展示 clipLimit=1、2、3（4×4）以及 clipLimit=2（8×8）的敏感性对比。参数按视觉稳定性选择，缺少干净参考，不能声称为客观最优。')
    for key,label in [('high','高端设备图'),('low','低端设备图')]:
        image(f'contrast_{key}.png',f'图 {2 if key=="high" else 4} {label}的四种处理结果')
        image(f'histograms_{key}.png',f'图 {3 if key=="high" else 5} {label}处理前后的直方图与累计分布')
    p('### 2.3 对比分析与方法选择')
    p('全局均衡化将两图的亮度跨度显著扩展，同时暗部纹理更突出，整体灰白感和散斑对比也随之增强；不宜仅因“看起来更亮”就认定图像更好。规定化使 high 向 low 的分布靠近、low 向 high 的分布靠近，但低端图原有的粗糙纹理仍存在，说明全局灰度迁移无法恢复组织细节。')
    p('结合局部组织层次与噪声放大程度，本实验更倾向采用适度 CLAHE 作为两图的显示增强方法：它能补偿局部明暗不均，且不必把整幅图的分布强制拉到参考图。high 原图已显示较清楚的组织边界，增强宜保守；low 的斑点较明显，实际可考虑先降噪再做适度 CLAHE。CLAHE 同样可能放大局部噪声，这一选择是显示效果上的判断，不构成诊断质量或设备质量结论。')
    image('clahe_parameters_high.png','图 6 高端图的 CLAHE 参数对比')
    image('clahe_parameters_low.png','图 7 低端图的 CLAHE 参数对比')
    p(md_table(['图像','处理','均值','标准差','熵 bit','0或255占比'],[[r['image'],r['method'],f"{r['mean']:.2f}",f"{r['std']:.2f}",f"{r['entropy_bits']:.3f}",f"{r['endpoint_pct']:.2f}%"] for r in contrast]))
    p('上表用于解释灰度映射变化。熵、标准差、极端灰度占比都不是独立的图像质量评分；显示增强没有干净参考，因此不使用 PSNR 或 SSIM 给增强方法排名。')
    p('## 3 人工高斯噪声下的降噪')
    p('### 3.1 噪声与空间划分')
    p(f"采用 NumPy default_rng(seed={info['seed']}) 生成均值 0、标准差 15 灰度级的独立加性高斯噪声，然后叠加、四舍五入并截断到 0～255。截断前样本标准差为 {d['noise']['raw_sample_std']:.3f}；截断后相对原图误差的标准差为 {d['noise']['after_clipping_error_std']:.3f}、均值为 {d['noise']['after_clipping_error_mean']:.3f}。黑色背景中负噪声被截断，会造成实际误差分布偏离零均值高斯；报告保留这一影响。")
    p('600×600 干净图划为 3×3 个 200×200 空间块，行列索引从 0 开始。训练块为 (0,0)、(0,1)、(0,2)、(1,0)、(2,1)、(2,2)，验证块为 (1,2)，测试块为 (1,1)、(2,0)。训练 patch 完全位于训练块内，不跨越边界；验证与测试区域各向内去除 12 像素边界，大于 CNN 的 8 像素感受野半径和 SSIM 的 3 像素计算半径。测试像素的卷积和 SSIM 邻域均不依赖训练块。')
    p(f"该划分能够避免同一像素参加训练和测试，但训练与测试仍来自同一幅图，存在同图相关性，不能替代跨患者、跨设备测试。模型选择使用固定独立验证噪声 seed={info['seed']+1000}；主实验 seed={info['seed']} 只用于最终评估，不据其测试分数调参。")
    image('spatial_split.png','图 8 训练 验证 测试空间块划分')
    p('### 3.2 传统高斯滤波与残差 CNN')
    p(f"传统方法选择高斯滤波。在验证区域比较空间标准差 σs∈{{0.5,0.7,1.0,1.2,1.5}}，核宽为 2×ceil(3σs)+1；按验证 PSNR 选择 σs={g['sigma']}、{g['kernel']}×{g['kernel']} 核。σs 的单位是像素，与加性噪声的 σ=15 灰度级不是同一参数。边界采用反射方式。")
    p(md_table(['空间 σs','核大小','验证 PSNR dB','验证 SSIM'],[[x['sigma'],f"{x['kernel']}×{x['kernel']}",f"{x['PSNR_dB']:.3f}",f"{x['SSIM']:.4f}"] for x in d['gaussian_validation']]))
    p(f"深度学习方法自行训练 8 层 3×3 残差 CNN：首层 1→32 通道，中间 6 层 32→32，末层 32→1；前 7 层使用 ReLU，无批归一化，共 {t['parameters']:,} 个参数。网络预测噪声残差，去噪图等于含噪输入减去预测残差。该网络借鉴残差降噪思想，并非原版 DnCNN，也没有使用或冒充预训练权重。")
    p(f"训练 patch 为 48×48，batch=16，随机旋转和翻转；每次迭代重新采样 σ=15 噪声并做同样的范围截断，以干净 patch 为监督目标，仅在中心 32×32 计算残差 MSE。Adam 初始学习率 0.001，在训练进度 60% 和 85% 处各乘 0.3，共 {t['steps']} 次迭代。每 100 步评估固定验证块，保存验证 PSNR 最佳权重，最佳迭代 {t['best_step']}。设备为 {t['device_name']}，本次训练耗时 {t['training_seconds']:.1f} 秒。推理时反射填充 12 像素后裁回原图，输出截断并量化为 uint8。")
    image('training_curve.png','图 9 固定验证块上的 PSNR 曲线')
    p('### 3.3 指标与结果')
    p('PSNR 用 255 作为峰值，由评价区域内的均方误差计算；数值越高表示对指定参考图的像素误差越小。SSIM 使用 scikit-image 默认 7×7 窗口，data_range=255，对 SSIM 局部图在指定区域取均值。MAE 为平均绝对灰度误差。所有表格均依据保存后的 8 位图计算。')
    p('为减少黑色背景抬高分数的影响，除完整独立测试内区，还报告其中超声扇形部分：干净图阈值>8，经 15×15 闭运算，取最大连通分量凸包并向内腐蚀 3 像素，再与测试内区求交集。此掩膜只定义评价范围，不用于训练或选择模型。掩膜 PNG 已随项目保存。')
    p(md_table(['评价区域','方法','PSNR dB','SSIM','MAE','像素数'],[[r['scope'],r['method'],f"{r['PSNR_dB']:.3f}",f"{r['SSIM']:.4f}",f"{r['MAE']:.3f}",r['pixels']] for r in d['synthetic_metrics'] if r['scope']!='全图内区 仅作补充']))
    image('synthetic_comparison.png','图 10 干净参考 人工加噪 高斯滤波与残差 CNN')
    image('synthetic_test_crop.png','图 11 中央独立测试块的局部放大 统一灰度范围')
    image('synthetic_errors.png','图 12 相对干净参考的绝对误差 色标统一为 0～40 灰度级')
    p(f"独立测试内区，人工加噪基线为 {noisy['PSNR_dB']:.3f} dB、SSIM={noisy['SSIM']:.4f}；高斯滤波为 {gf['PSNR_dB']:.3f} dB、{gf['SSIM']:.4f}；残差 CNN 为 {cnn['PSNR_dB']:.3f} dB、{cnn['SSIM']:.4f}。CNN 相对高斯滤波的 PSNR 差值为 {cnn['PSNR_dB']-gf['PSNR_dB']:+.3f} dB。测试扇形区域中，两者分别为 {cfan['PSNR_dB']:.3f} dB 和 {gfan['PSNR_dB']:.3f} dB；该范围更能反映超声区域而非大面积黑底的恢复误差。")
    p('高斯滤波实现简单、无需训练，在近似平坦区域能稳定压低独立高斯噪声；但其固定线性平滑同时削弱边缘和细纹理。残差 CNN 可从训练样例学习局部噪声与结构之间的关系，倾向于保留明显组织边界并抑制随机扰动，但结果依赖有限训练数据和噪声设定，不能保证保留所有微小结构。应结合独立区域指标与局部图一起判断。')
    p('本例黑色背景和心腔暗区较多，范围截断带来的正偏置会被普通高斯平滑保留下来，而 CNN 可以学习补偿部分偏置，因此 CNN 的指标优势不能全部归因于组织细节恢复。扇形区域指标减少了外部黑底的影响，但仍包含真实心腔暗区；单次固定噪声实验也不代表多次独立采样的统计显著性。')
    p('全图指标含训练区域，只作图像整体重建的补充描述，不作为独立泛化成绩。')
    p(md_table(['方法','全图内区 PSNR dB','全图内区 SSIM'],[[r['method'],f"{r['PSNR_dB']:.3f}",f"{r['SSIM']:.4f}"] for r in d['synthetic_metrics'] if r['scope']=='全图内区 仅作补充']))
    p('## 4 真实噪声图像的处理与讨论')
    p('将上述验证选择的高斯核和人工高斯噪声上训练的 CNN 原样用于 Figure2_noisy，不另行利用测试图调参，也不缩放它以制造参考配对。不同图像的像素采样尺度没有物理标定，因此相同像素核宽并不代表相同物理平滑尺度。')
    image('real_comparison.png','图 13 真实含噪图的同参数降噪比较')
    image('real_crop.png','图 14 真实含噪图局部放大 x为75到195 y为55到185 左闭右开')
    p('真实图中的细密颗粒与组织纹理交织。高斯滤波使颗粒和组织边缘一起变软；CNN 也压低了细颗粒，但局部纹理明显减少，部分低亮度区域变暗，低对比边界可能被削弱。此图上 CNN 的平滑较强，不能据人工噪声的高分认定其在真实图上优于高斯滤波。局部看起来更平滑仅说明输出发生平滑，不能据此认定临床细节更真实。')
    p('人工加噪属于可控的独立加性高斯模型，而实际超声斑点往往与局部信号强度相关，也可能受设备处理链、压缩和重采样影响。用高斯噪声训练的网络在真实图上存在分布差异；训练仅来自单张 600×600 图，真实图为另一内容的 256×256 图，纹理尺度也不同，因此人工实验中的分数优势不能直接推广到真实噪声。')
    p('下表仅描述固定 ROI（x=55:215、y=40:205）的灰度与变化。平均梯度基于归一化 Sobel，变化 RMS 相对真实含噪输入计算。这些值衡量平滑和改动程度，不是无参考质量排名；它们不证明噪声被正确去除。没有配对真值，因此不报告该真实图的 PSNR、SSIM 或“降噪提升百分比”。')
    p(md_table(['方法','ROI 均值','ROI 标准差','平均梯度','变化 RMS'],[[r['method'],f"{r['ROI_mean']:.3f}",f"{r['ROI_std']:.3f}",f"{r['mean_gradient']:.3f}",f"{r['change_RMS']:.3f}"] for r in d['real_descriptive']]))
    p('若进一步改进，可获取多幅独立图像及真实噪声配对数据，在患者层面划分数据；比较适合超声的斑点模型、对数域处理或真实噪声自监督方法。此处未实施这些扩展，不将其描述为已完成实验。')
    p('## 5 结论与复现')
    p(f'增强部分，适度 CLAHE 在本组图上较适合保留局部层次，同时需要控制散斑放大；规定化适用于灰度分布对齐，不能替代结构恢复。降噪部分，在固定人工高斯噪声和独立空间测试范围下，{best} 的 PSNR 更高。两种方法在真实噪声上均有分布和尺度限制，缺少参考时应保留定性判断的边界。')
    p('项目已包含原图、全部处理图、直方图、训练划分、训练曲线、逐项 CSV 指标、JSON 记录、模型权重和完整代码。运行 python 运行实验.py 直接加载随附权重重算；运行 python 运行实验.py --retrain --steps 1800 可重新训练。推荐依照 运行依赖.txt 在独立环境安装依赖；CPU 可运行但训练较慢。具体版本、种子、输入 SHA256 记录在 实验结果/实验记录.json。GPU 或库版本不同可能产生微小数值差异，不应把同一随机种子理解为跨硬件逐位一致。')
    p('Windows 的某些 Anaconda 环境可能出现重复 Intel OpenMP 运行库错误；建议优先在独立环境安装依赖。本机为完成运行通过环境变量 KMP_DUPLICATE_LIB_OK=TRUE 绕过，该变量由启动脚本设置，不会修改系统配置；该兼容性措施不能替代验证数值结果。')
    p('## 6 方法参考')
    p('参考资料用于说明方法背景，实验数值全部来自本项目的实际运行。网络为自行编写与训练的小型残差 CNN，不声称复现论文原版模型。')
    p('1. OpenCV 官方教程 Histogram Equalization 与 CLAHE：https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html\n2. OpenCV 官方说明 GaussianBlur：https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html\n3. Zhang K 等 Beyond a Gaussian Denoiser Residual Learning of Deep CNN for Image Denoising，2017，残差降噪思想参考：https://arxiv.org/abs/1608.03981\n4. scikit-image 官方指标文档 structural_similarity：https://scikit-image.org/docs/stable/api/skimage.metrics.html')
    markdown='\n\n'.join(parts)+'\n'
    (ROOT/'实验报告.md').write_text(markdown,encoding='utf-8')
    # Self-contained HTML, requiring neither a server nor linked image files.
    blocks=[]
    for part in parts:
        if part.startswith('!['):
            first,caption=part.split('\n\n',1)
            path=first.split('](',1)[1][:-1]
            b64=base64.b64encode((ROOT/path).read_bytes()).decode('ascii')
            compact = ' class="compact"' if path.endswith('spatial_split.png') else ''
            blocks.append(f'<figure{compact}><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        elif part.startswith('#'):
            level=len(part)-len(part.lstrip('#'))
            blocks.append(f'<h{level}>{html.escape(part[level:].strip())}</h{level}>')
        elif part.startswith('|'):
            lines=part.splitlines(); cells=lambda line:[x.strip() for x in line.strip('|').split('|')]
            head=''.join(f'<th>{html.escape(x)}</th>' for x in cells(lines[0]))
            body=''.join('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in cells(line))+'</tr>' for line in lines[2:])
            blocks.append(f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
        else:
            blocks.append('<p>'+html.escape(part).replace('\n','<br>')+'</p>')
    css='''body{max-width:1040px;margin:40px auto;padding:0 32px;color:#20272e;font:16px/1.85 "Microsoft YaHei","Noto Sans CJK SC",sans-serif}h1,h2,h3{color:#000;line-height:1.45}h1{font-size:30px;margin-bottom:24px}h2{font-size:24px;margin-top:40px}h3{font-size:19px;margin-top:28px}p{margin:16px 0}figure{margin:28px 0;break-inside:avoid}img{display:block;width:100%;height:auto}figcaption{text-align:center;font-size:14px;color:#53616a;margin-top:9px}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.6}th,td{border:1px solid #d9d9d9;padding:9px 11px;text-align:center}th{background:#dce8ef;color:#000}tbody tr:nth-child(even){background:#f5f8fa}.table{margin:22px 0;overflow-x:auto}@media print{body{margin:0;padding:0;font-size:10.5pt;max-width:none}h1{font-size:22pt}h2{font-size:17pt;break-after:avoid}h3{font-size:13pt;break-after:avoid}figure{margin:14px 0}p{orphans:3;widows:3}thead{display:table-header-group}tr{break-inside:avoid}table{font-size:9pt}th,td{padding:5px} @page{size:A4;margin:18mm}}'''
    css += '@media print{.table{break-inside:avoid}.compact{width:62%;margin-left:auto;margin-right:auto}figcaption{font-size:9pt}table{table-layout:auto}}'
    content='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>超声图像增强与降噪实验报告</title><style>'+css+'</style><body>'+''.join(blocks)+'</body></html>'
    (ROOT/'实验报告.html').write_text(content,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=1800)
    parser.add_argument('--retrain',action='store_true')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    print('Running on',device,flush=True)
    contrast_inputs={k:load_gray(DATA/f'Figure1_{k}.jpg') for k in ['high','low']}
    clean=load_gray(DATA/'Figure2_clean.png');real=load_gray(DATA/'Figure2_noisy.png')
    contrast=contrast_experiment(contrast_inputs)
    denoising=denoising_experiment(clean,real,args,device)
    uses={'Figure1_high.jpg':'高端设备增强','Figure1_low.jpg':'低端设备增强',
          'Figure2_clean.png':'人工加噪参考与空间划分训练','Figure2_noisy.png':'真实噪声独立应用'}
    inputs=[]
    for name,use in uses.items():
        path=DATA/name;im=Image.open(path)
        inputs.append({'file':name,'width':im.width,'height':im.height,'mode':im.mode,
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'use':use})
    info={'seed':args.seed,'inputs':inputs,'versions':{'python':sys.version.split()[0],
          'numpy':np.__version__,'opencv':cv2.__version__,'torch':torch.__version__,
          'matplotlib':matplotlib.__version__,'scikit_image':__import__('skimage').__version__,
          'pillow':Image.__version__},'contrast':contrast,**denoising}
    (OUT/'实验记录.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    localize_result_names()
    make_report(contrast,denoising,info)
    print('FINISHED. See 实验报告.html and 实验结果/实验记录.json',flush=True)
    print(json.dumps([r for r in denoising['synthetic_metrics'] if r['scope']=='独立测试区域'],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
