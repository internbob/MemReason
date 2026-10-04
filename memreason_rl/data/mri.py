from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
import torch

def load_mri(path,spacing=(2.,2.,2.),shape=(96,112,96)):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    img=nib.as_closest_canonical(nib.load(path))
    if len(img.shape)!=3:
        raise ValueError(f'Expected 3D T1: {img.shape}')
    img=resample_to_output(img,voxel_sizes=spacing,order=1)
    array=np.asarray(img.dataobj,dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite MRI voxels; requires QC')
    foreground=np.abs(array)>1e-6
    if foreground.sum()<100 or array[foreground].std()<1e-6:
        raise ValueError('MRI is empty/constant')
    values=array[foreground]
    array[foreground]=(values-values.mean())/values.std()
    output=np.zeros(shape,dtype=np.float32)
    src=[]; dst=[]
    for actual,target in zip(array.shape,shape):
        size=min(actual,target)
        src.append(slice((actual-size)//2,(actual-size)//2+size))
        dst.append(slice((target-size)//2,(target-size)//2+size))
    output[tuple(dst)]=array[tuple(src)]
    return torch.from_numpy(output[None])
