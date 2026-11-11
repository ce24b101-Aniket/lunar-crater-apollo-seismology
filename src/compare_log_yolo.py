"""Scale-space Laplacian-of-Gaussian crater candidate baseline; tune on val, test once."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_laplace, maximum_filter

ROOT=Path('work/data/Sombench-NAC-Crater-Detection'); IMGS=Path('work/data/sombench_nac_yolo/images')
OUT=Path('results'); OUT.mkdir(exist_ok=True)
SIGMAS=np.geomspace(1.7,42.,14)
THRESHOLDS=[0.008,0.012,0.018,0.027,0.04]

def load(split):
 d=json.loads((ROOT/f'annotations_{split}.json').read_text(encoding='utf-8'))
 ids={i['id']:[] for i in d['images'] if i.get('label_set')=='standard'}
 for a in d['annotations']:
  if a.get('truncated',False) or float(a.get('diam_px',0))<5: continue
  if a['image_id'] not in ids: continue
  x,y,w,h=map(float,a['bbox']);ids[a['image_id']].append([x,y,x+w,y+h])
 imgs=[i for i in d['images'] if i.get('label_set')=='standard']
 return [(i,np.asarray(ids[i['id']],dtype=float).reshape(-1,4)) for i in imgs]

def log_candidates(img,threshold):
 gray=np.array(Image.open(img).convert('L'))
 # Local contrast normalization limits across-image illumination variation.
 x=gray.astype(np.float32)/255.; x=cv2.GaussianBlur(x,(3,3),0.65)
 lo,hi=np.percentile(x,[1,99]); x=np.clip((x-lo)/max(hi-lo,1e-6),0,1)
 # Positive scale-normalized LoG lobes mark dark interiors; use the absolute response
 # to allow both bright and dark crater morphology, with scale-space NMS.
 stack=np.stack([np.abs((s*s)*gaussian_laplace(x,sigma=float(s))) for s in SIGMAS])
 peaks=(stack==maximum_filter(stack,size=(3,5,5),mode='nearest'))&(stack>=threshold)
 si,yy,xx=np.where(peaks);cand=[]
 for k,y,x0 in zip(si,yy,xx):
  diam=2*np.sqrt(2)*SIGMAS[k]
  if diam<5 or diam>252: continue
  r=diam/2
  cand.append([max(0,x0-r),max(0,y-r),min(256,x0+r),min(256,y+r),float(stack[k,y,x0])])
 # Greedy suppression across adjacent scales/nearby peaks, highest response first.
 cand.sort(key=lambda z:z[4],reverse=True);kept=[]
 for b in cand:
  def separated(q):
   dx=(b[0]+b[2]-q[0]-q[2])*.5;dy=(b[1]+b[3]-q[1]-q[3])*.5
   return dx*dx+dy*dy>(min(b[2]-b[0],q[2]-q[0])*.35)**2
  if all(separated(q) for q in kept):
   kept.append(b)
 return np.asarray([b[:4] for b in kept],dtype=float).reshape(-1,4)

def iou(a,b):
 if not len(a) or not len(b):return np.zeros((len(a),len(b)))
 x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1]);x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3])
 inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1);aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);bb=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
 return inter/(aa[:,None]+bb[None,:]-inter+1e-9)
def score(items,threshold):
 tp=fp=fn=0
 for im,g in items:
  f=IMGS/'val' if im['split']=='val' else IMGS/'test'
  p=log_candidates(f/(Path(im['file_name']).stem+'.png'),threshold)
  m=iou(p,g); used=set()
  for k in range(len(p)):
   j=int(m[k].argmax()) if m.shape[1] else -1
   if j>=0 and m[k,j]>=.5 and j not in used:tp+=1;used.add(j)
   else:fp+=1
  fn+=len(g)-len(used)
 pr=tp/(tp+fp+1e-12);re=tp/(tp+fn+1e-12)
 return {'TP':tp,'FP':fp,'FN':fn,'precision':pr,'recall':re,'F1':2*pr*re/(pr+re+1e-12)}
val=load('val'); test=load('test')
vm=[(score(val,t),t) for t in THRESHOLDS]
best,thr=max(vm,key=lambda z:z[0]['F1'])
tm=score(test,thr)
result={'method':'single-band scale-normalized Laplacian-of-Gaussian blob candidates; square bounding boxes; one-to-one greedy IoU>=0.5','sigmas_px':SIGMAS.tolist(),'threshold_grid_validation_only':THRESHOLDS,'selected_threshold':thr,'validation':best,'test':tm,'test_set':'official standard-label test split, truncated and <5px reference labels removed; 46 tiles','limits':'Threshold chosen by validation F1; this is a simple classical baseline, not a crater-specific trained model. No test retuning. Scale and threshold choices affect the result.'}
(OUT/'LoG_vs_YOLO_same_test.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
