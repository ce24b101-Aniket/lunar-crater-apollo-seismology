"""QC and exploratory response correction for four known Apollo impacts at S14."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from obspy import read,read_inventory,UTCDateTime

ROOT=Path('work/data/apollo_impacts');OUT=Path('outputs');OUT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'download_manifest.json').read_text(encoding='utf-8'))
PREF=(.4,.7,11.,14.);FMIN,FMAX=1.,10.;items=[];traces=[]
def max_run(mask):
 edges=np.diff(np.r_[False,mask,False].astype(np.int8));s=np.flatnonzero(edges==1);e=np.flatnonzero(edges==-1);return int(np.max(e-s)) if len(s) else 0
for ev in manifest:
 path=Path(ev['SHZ_file']);tr=read(str(path))[0]
 raw=tr.data.astype(float).copy();miss=(raw==-1);valid=raw[~miss]
 s14inv=read_inventory(ev['response_file'])
 attpath=Path(ev.get('ATT_file',''));att=read(str(attpath)) if attpath.exists() else None
 q={'event_id':ev['id'],'event_source':ev['source'],'event_utc_earth_received':ev['origin_utc'],'station_channel':tr.id,
    'sampling_rate_hz':float(tr.stats.sampling_rate),'npts':len(raw),'trace_start_utc':str(tr.stats.starttime),'trace_end_utc':str(tr.stats.endtime),
    'missing_fraction':float(miss.mean()),'missing_max_run_samples':max_run(miss),'min_raw_du_excluding_sentinel':float(valid.min()) if len(valid) else None,
    'max_raw_du':float(valid.max()) if len(valid) else None,'n_at_0':int(np.sum(valid<=0)),'n_at_or_above_1023':int(np.sum(valid>=1023)),
    'fraction_at_or_above_1023_of_all_samples':float(np.mean(raw>=1023)),'max_run_at_or_above_1023':max_run(raw>=1023),
    'impact_time_window_s':[-300,7200], 'timing_correction':'ATT waveform downloaded but not applied; precise onset and sub-second timing are not interpreted.'}
 if att:
  q['ATT']={'id':att[0].id,'npts':len(att[0].data),'sampling_rate_hz':float(att[0].stats.sampling_rate),'start':str(att[0].stats.starttime),'end':str(att[0].stats.endtime),
           'min':float(np.min(att[0].data)),'max':float(np.max(att[0].data))}
  ev['ATT_inspected']=True
 # Interpolate sentinel runs only for an exploratory filtered trace; retain raw rail statistics.
 fill=raw.copy()
 if miss.any():
  ix=np.arange(len(fill));fill[miss]=np.interp(ix[miss],ix[~miss],fill[~miss])
 proc=tr.copy();proc.data=fill.astype(float);proc.detrend('demean');proc.detrend('linear')
 vel=proc.copy()
 try:
  vel.remove_response(inventory=s14inv,output='VEL',pre_filt=PREF,water_level=60)
  vel.filter('bandpass',freqmin=FMIN,freqmax=FMAX,corners=4,zerophase=True)
  q['peak_abs_filtered_velocity_m_per_s_unvalidated']=float(np.max(np.abs(vel.data)))
  q['filter_hz']=[FMIN,FMAX];q['response_prefilter_hz']=list(PREF)
  traces.append((ev,q,vel,raw))
 except Exception as exc:
  q['response_processing_error']=repr(exc);traces.append((ev,q,None,raw))
 items.append(q)
(OUT/'Apollo_Artificial_Impact_QC.json').write_text(json.dumps(items,indent=2,ensure_ascii=False),encoding='utf-8')
fig,ax=plt.subplots(4,1,figsize=(10,11),constrained_layout=True)
for k,(ev,q,vel,raw) in enumerate(traces):
 rel=np.arange(len(raw))/q['sampling_rate_hz']-300
 ax[k].plot(rel/60,raw,color='#4d4d4d',lw=.4)
 ax[k].axvline(0,color='#b22222',ls='--',lw=1)
 ax[k].set_xlim(-2,20);ax[k].set_ylabel('Counts (DU)',fontweight='bold');ax[k].grid(alpha=.2)
 ax[k].text(.01,.94,f"({chr(97+k)}) {ev['id']} | max={q['max_raw_du']:.0f} DU | rail samples={q['n_at_or_above_1023']}",transform=ax[k].transAxes,va='top',fontweight='bold',fontsize=10)
 ax[k].tick_params(labelsize=9)
ax[-1].set_xlabel('Minutes from catalogued Earth-received impact time',fontweight='bold')
fig.savefig(OUT/'Apollo_Artificial_Impact_Records.png',dpi=600,bbox_inches='tight');plt.close(fig)
print(json.dumps(items,indent=2,ensure_ascii=False))
