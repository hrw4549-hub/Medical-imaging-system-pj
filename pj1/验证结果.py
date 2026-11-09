"""Check source hashes, saved image metrics, masks and model output.

Run after 运行实验.py: python 验证结果.py
No training or file modification is performed.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import cv2
import 运行实验 as e

root=Path(__file__).resolve().parent
metrics=json.loads((root/'实验结果/实验记录.json').read_text(encoding='utf-8'))
checks=[]
for record in metrics['inputs']:
    path=root/'原始数据'/record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['file']
checks.append('Four original input SHA256 hashes match.')
clean=e.load_gray(root/'原始数据/Figure2_clean.png')
real=e.load_gray(root/'原始数据/Figure2_noisy.png')
assert clean.shape==(600,600) and real.shape==(256,256)
masks,_,_=e.tile_masks(clean.shape)
support=e.fan_mask(clean)
assert not np.any(masks['train']&masks['test'])
assert not np.any(masks['validation']&masks['test'])
assert np.array_equal(e.load_gray(e.IMG/e.IMAGE_NAMES['evaluation_test_mask.png'])>0,masks['test'])
assert np.array_equal(e.load_gray(e.IMG/e.IMAGE_NAMES['evaluation_fan_mask.png'])>0,support)
checks.append('Spatial partitions are disjoint and saved evaluation masks match.')
noisy,_=e.add_noise(clean,metrics['seed'])
assert np.array_equal(noisy,e.load_gray(e.IMG/e.IMAGE_NAMES['synthetic_noisy_sigma15.png']))
g=metrics['gaussian_selected']
assert g['PSNR_dB']==max(row['PSNR_dB'] for row in metrics['gaussian_validation'])
gaussian=cv2.GaussianBlur(noisy,(g['kernel'],g['kernel']),g['sigma'],borderType=cv2.BORDER_REFLECT_101)
assert np.array_equal(gaussian,e.load_gray(e.IMG/e.IMAGE_NAMES['synthetic_gaussian.png']))
checks.append('Noise regeneration and validation-selected Gaussian output match byte for byte.')
methods=['人工加噪','高斯滤波','残差 CNN']
files=[e.IMAGE_NAMES[x] for x in ['synthetic_noisy_sigma15.png','synthetic_gaussian.png','synthetic_cnn.png']]
scopes=['独立测试区域','测试区域内超声扇形','全图内区 仅作补充']
full=np.zeros(clean.shape,bool);full[5:-5,5:-5]=True
scope_masks=[masks['test'],masks['test']&support,full]
for method,file in zip(methods,files):
    image=e.load_gray(e.IMG/file)
    for scope,mask in zip(scopes,scope_masks):
        computed=e.quality(clean,image,mask)
        saved=next(row for row in metrics['synthetic_metrics'] if row['method']==method and row['scope']==scope)
        for key,value in computed.items():
            assert math.isclose(value,saved[key],rel_tol=1e-10,abs_tol=1e-10),(method,scope,key)
checks.append('All 9 rows of PSNR, SSIM, MAE, RMSE and pixel counts recompute exactly.')
torch.set_num_threads(4)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state=torch.load(root/'模型/残差卷积网络权重.pt',map_location=device,weights_only=True)
model=e.ResidualCNN().to(device);model.load_state_dict(state['state_dict']);model.eval()
recomputed=e.predict(model,noisy,device)
saved=e.load_gray(e.IMG/e.IMAGE_NAMES['synthetic_cnn.png'])
max_difference=int(np.abs(recomputed.astype(int)-saved.astype(int)).max())
assert max_difference<=1,max_difference
checks.append(f'Saved checkpoint inference matches quantized output (max difference {max_difference}).')
for row in metrics['contrast']:
    base=e.load_gray(root/'原始数据'/f"Figure1_{row['image']}.jpg")
    reference=e.load_gray(root/'原始数据'/f"Figure1_{'low' if row['image']=='high' else 'high'}.jpg")
    variants=[base,cv2.equalizeHist(base),e.match_histogram(base,reference),
              cv2.createCLAHE(clipLimit=2.,tileGridSize=(4,4)).apply(base)]
    index=['原图','全局均衡化','规定化到另一图','CLAHE'].index(row['method'])
    item_name=f"{'high' if row['image']=='high' else 'low'}_{index}.png"
    assert np.array_equal(variants[index],e.load_gray(e.IMG/e.IMAGE_NAMES[item_name]))
    for key,value in e.hist_stats(variants[index]).items():
        assert math.isclose(value,row[key],rel_tol=1e-10,abs_tol=1e-10)
checks.append('All 8 enhancement outputs and histogram statistics recompute exactly.')
html=next(root.glob('*.html'),None)
if html:
    text=html.read_text(encoding='utf-8')
    assert text.count('<figure')==14
    assert text.count('data:image/png;base64,')==14
    assert '\ufffd' not in text
docx=root/'超声图像增强与降噪实验报告.docx'
assert docx.exists()
from zipfile import ZipFile
with ZipFile(docx) as archive:
    assert len([n for n in archive.namelist() if n.startswith('word/media/')])==14
assert len(list(csv.DictReader((root/'实验结果/人工噪声降噪指标.csv').open(encoding='utf-8-sig'))))==9
checks.append('Editable DOCX has 14 embedded figures; UTF-8 text and CSV row count are valid.')
for check in checks:print('PASS:',check)
print('ALL CHECKS PASSED')
