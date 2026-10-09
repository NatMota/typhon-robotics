"""M6-only 6D ordering bridge. M0-M5 arrays and the physical OSC decoder stay unchanged."""
import numpy as np
DOMAIN=29
ORDER=[0,3,1,4,2,5]
INVERSE=[0,2,4,1,3,5]
def reorder(value, inverse=False):
    out=value.clone() if hasattr(value,'clone') else np.array(value,copy=True)
    order=INVERSE if inverse else ORDER
    for start in (3,13):out[...,start:start+6]=value[..., [start+i for i in order]]
    return out

def self_check():
    r,_=np.linalg.qr(np.random.default_rng(17).normal(size=(100,3,3)))
    r[:,:,2]*=np.linalg.det(r)[:,None]
    a=np.zeros((100,20));a[:,3:9]=np.concatenate([r[:,:,0],r[:,:,1]],axis=1)
    b=reorder(a)
    assert np.array_equal(b[:,3:9],r[:,:,:2].reshape(100,6))
    assert np.array_equal(reorder(b,True),a)
    return {'passed':True,'random_rotations':100,'physical_contract':'absolute world XYZ metres; generic interleaved first two rotation columns; 20 values; dummy second arm','source_storage':'M0-M5 unchanged; permutation only at M6 boundary'}
if __name__=='__main__':print(self_check())
