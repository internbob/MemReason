import numpy as np

def temporal_reward(times,predicted,observed,scale=1.):
    t,p,y=map(lambda x:np.asarray(x,dtype=float),(times,predicted,observed))
    mask=np.isfinite(t)&np.isfinite(p)&np.isfinite(y)
    if mask.sum()<2 or np.unique(t[mask]).size<2:
        return None
    bp=np.polyfit(t[mask],p[mask],1)[0]
    by=np.polyfit(t[mask],y[mask],1)[0]
    return float(np.exp(-abs(bp-by)/max(scale,1e-8)))

def trend_label(values,direction,epsilon=.01):
    if direction not in {'higher=better','higher=worse'} or len(values)<2:
        return None
    delta=values[-1]-values[0]
    if abs(delta)<=epsilon:
        return 'stable'
    improves=delta>0 if direction=='higher=better' else delta<0
    return 'improve' if improves else 'decline'
