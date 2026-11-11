"""Validation-tuned classical Hough-circle baseline vs YOLO on same tiles."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ANN=Path('work/data/Sombench-NAC-Crater-Detection')
IMG=Path('work/data/sombench_nac_yolo/images')
BEST=Path('work/runs/yolo11n_sombench_nac/weights/best.pt')
OUT=Path('outputs')

def load(split):
 d=json.loads((ANN/f'annotations_{split}.json').read_text(encoding='utf-8'))
 by={i['id']:i for i in d['images']}; aa={}
 for a in d['annotations']:
  if a.get('truncated',False) or float(a.get('diam_px',0))<5: continue
  x,y,w,h=map(float,a['bbox']); aa.setdefault(a['image_id'],[]).append([x,y,x+w,y+h])
 return [(i,np.asarray(aa.get(i['id'],[]),dtype=float).reshape(-1,4)) for i in d['images'] if i.get('label_set')=='standard']

def iou_matrix(a,b):
 if not len(a) or not len(b): return np.zeros((len(a),len(b)))
 x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1]);x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3])
 inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1); area1=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);area2=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
 return inter/(area1[:,None]+area2[None,:]-inter+1e-9)

def score(gt,pred):
 tp=fp=fn=0
 for g,p in zip(gt,pred):
  m=iou_matrix(p,g); used=set()
  for k in range(len(p)):
   j=int(m[k].argmax()) if m.shape[1] else -1
   if j>=0 and m[k,j]>=.5 and j not in used: tp+=1;used.add(j)
   else: fp+=1
  fn+=len(g)-len(used)
 prec=tp/(tp+fp+1e-12);rec=tp/(tp+fn+1e-12);f1=2*prec*rec/(prec+rec+1e-12)
 return {'TP':tp,'FP':fp,'FN':fn,'precision':prec,'recall':rec,'F1':f1}

def hough_predictions(items,param2):
 pred=[]
 for im,_ in items:
  path=IMG/'val'/Path(im['file_name']).with_suffix('.png').name if im['split']=='val' else IMG/'test'/Path(im['file_name']).with_suffix('.png').name
  gray=np.array(Image.open(path).convert('L'))
  gray=cv2.equalizeHist(gray);gray=cv2.GaussianBlur(gray,(5,5),1.2)
  circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.5,minDist=10,param1=80,param2=param2,minRadius=2,maxRadius=126)
  boxes=[]
  if circles is not None:
   for x,y,r in np.round(circles[0]).astype(int): boxes.append([max(0,x-r),max(0,y-r),min(256,x+r),min(256,y+r)])
  pred.append(np.asarray(boxes,dtype=float).reshape(-1,4))
 return pred

val=load('val'); vg=[g for _,g in val]
grid=[]
for p2 in [18,26,35]:
 pp=hough_predictions(val,p2);grid.append((score(vg,pp),p2))
best_metric,best_p2=max(grid,key=lambda q:q[0]['F1'])
test=load('test');tg=[g for _,g in test]
hp=hough_predictions(test,best_p2)
model=YOLO(str(BEST)); yp=[]
paths=[str(IMG/'test'/Path(im['file_name']).with_suffix('.png').name) for im,_ in test]
results=model.predict(paths,imgsz=640,conf=.25,iou=.5,device='cpu',verbose=False)
for r in results:
 b=r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4)); yp.append(b)
idx=int(np.argmax([len(g) for g in tg]))
im,gt=test[idx]; image=np.array(Image.open(paths[idx]).convert('L'))
fig,ax=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
ax[0].imshow(image,cmap='gray');ax[0].set_title('(a) NAC test tile')
ax[1].imshow(image,cmap='gray');ax[1].set_title(f'(b) Reference labels (n={len(gt)})')
for x1,y1,x2,y2 in gt:ax[1].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec='#ffd400',lw=.7))
ax[2].imshow(image,cmap='gray');ax[2].set_title(f'(c) YOLO11n predictions (n={len(yp[idx])})')
for x1,y1,x2,y2 in yp[idx]:ax[2].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec='#e53935',lw=.7))
for a in ax:a.set_xlim(0,256);a.set_ylim(256,0);a.axis('off')
fig.suptitle('Held-out crater detector example',fontweight='bold');fig.savefig(OUT/'YOLO11n_test_tile_example.png',dpi=600);plt.close(fig)
result={'matching_rule':'greedy one-to-one box IoU >= 0.5; same filtered standard-label test set (>=5 px, non-truncated)',
 'hough_validation_param2_grid':{'selected_param2':best_p2,'validation_metrics_at_selected':best_metric},
 'test':{'HoughCircles':score(tg,hp),'YOLO11n_conf0.25':score(tg,yp)},
 'test_example':{'image':im['file_name'],'reference_boxes':len(gt),'YOLO_boxes_conf_0_25':len(yp[idx]),'figure':'YOLO11n_test_tile_example.png'},
 'limitations':'Classical Hough output has no calibrated confidence; precision/recall/F1 at fixed thresholds are not COCO mAP. Hough settings selected on validation only.'}
(OUT/'Hough_vs_YOLO_same_test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
