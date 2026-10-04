import numpy as np

def gaussian_calibration(mean,std,target,observed_mask):
    mean,std,target=np.asarray(mean),np.asarray(std),np.asarray(target)
    valid=np.asarray(observed_mask,dtype=bool)&np.isfinite(mean)&np.isfinite(std)&np.isfinite(target)&(std>0)
    if not valid.any():
        return {'n':0,'nll':None,'coverage_95':None,'mean_interval_width_95':None}
    error=target[valid]-mean[valid]
    sigma=std[valid]
    return dict(n=int(valid.sum()),nll=float(np.mean(.5*np.log(2*np.pi*sigma**2)+.5*(error/sigma)**2)),
        coverage_95=float(np.mean(np.abs(error)<=1.96*sigma)),mean_interval_width_95=float(np.mean(3.92*sigma)))
