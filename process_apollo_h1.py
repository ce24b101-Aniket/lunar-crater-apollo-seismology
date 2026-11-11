"""Process two published Apollo SP waveforms from PDS MiniSEED."""
from pathlib import Path
import json
import numpy as np
from obspy import read, read_inventory, UTCDateTime
from scipy.signal import correlate, correlation_lags
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

ROOT=Path('work/data/apollo_s15'); OUT=Path('outputs'); OUT.mkdir(exist_ok=True)
INV=read_inventory(str(ROOT/'stationxml.xa.0.sxml'))
EVENTS=[
 {'id':'KO-26','utc':'1975-03-23T21:10:00','file':'xa.s15..shz.1975.082.0.mseed'},
 {'id':'KO-40','utc':'1976-12-02T12:45:00','file':'xa.s15..shz.1976.337.0.mseed'},
]
FMIN,FMAX=1.,10.; PREF=(.4,.7,11.,14.); BEFORE,AFTER=60.,660.

def window(ev, output='VEL'):
    tr=read(str(ROOT/ev['file']))[0]; p=UTCDateTime(ev['utc'])
    tr.trim(p-BEFORE,p+AFTER,pad=True,fill_value=0)
    # Nunn et al. encode missing Apollo samples as -1 in MiniSEED. These are
    # sentinels, not measurements. Here every sentinel in these windows is an
    # isolated one-sample gap, so interpolate only these single samples.
    missing=tr.data == -1
    ev['missing_sample_fraction']=float(missing.mean())
    if np.any(missing):
        ix=np.arange(tr.stats.npts)
        good=~missing
        tr.data=np.interp(ix,ix[good],tr.data[good]).astype(tr.data.dtype)
    if output=='VEL':
        ev['_raw_counts']=tr.data.copy()
    tr.detrend('demean'); tr.detrend('linear')
    tr.remove_response(inventory=INV,output=output,pre_filt=PREF,water_level=60)
    tr.filter('bandpass',freqmin=FMIN,freqmax=FMAX,corners=4,zerophase=True)
    return tr

def psa_newmark(ag,dt,periods,damping=.05):
    beta,gamma=.25,.5; result=[]
    for period in periods:
        w=2*np.pi/period; c=2*damping*w; k=w*w
        a0=1/(beta*dt*dt); a1=gamma/(beta*dt); a2=1/(beta*dt); a3=1/(2*beta)-1
        a4=gamma/beta-1; a5=dt*(gamma/(2*beta)-1); keff=k+a0+c*a1
        u=v=arel=umax=0.
        for ag1 in ag[1:]:
            peff=-ag1+(a0*u+a2*v+a3*arel)+c*(a1*u+a4*v+a5*arel)
            un=peff/keff; an=a0*(un-u)-a2*v-a3*arel
            vn=v+dt*((1-gamma)*arel+gamma*an)
            u,v,arel=un,vn,an; umax=max(umax,abs(u))
        result.append(w*w*umax)
    return np.asarray(result)

def ko_smooth(freq,amp,bandwidth=40,ncenters=180):
    centers=np.geomspace(max(FMIN,freq[1]),FMAX,ncenters); vals=[]
    for fc in centers:
        z=np.log10(np.maximum(freq[1:],1e-12)/fc); u=bandwidth*z
        w=np.ones_like(u); nz=np.abs(u)>1e-12; w[nz]=(np.sin(u[nz])/u[nz])**4
        vals.append(np.sum(w*amp[1:])/(np.sum(w)+1e-30))
    return centers,np.asarray(vals)

processed=[(ev,window(ev,'VEL')) for ev in EVENTS]
traces=[tr for _,tr in processed]
assert all(abs(tr.stats.sampling_rate-53.)<1e-6 for tr in traces)
assert len({tr.stats.npts for tr in traces})==1
x=traces[0].data.astype(float); y=traces[1].data.astype(float); x-=x.mean(); y-=y.mean()
c=correlate(x,y,mode='full',method='fft'); lags=correlation_lags(len(x),len(y),mode='full')
c/=np.sqrt((x*x).sum()*(y*y).sum()); valid=np.abs(lags/traces[0].stats.sampling_rate)<=5
idx=np.flatnonzero(valid)[np.argmax(np.abs(c[valid]))]; cc=float(c[idx]); lag=float(lags[idx]/traces[0].stats.sampling_rate)
freq=np.fft.rfftfreq(len(x),d=1/traces[0].stats.sampling_rate); specs=[]
for tr in traces:
    z=tr.data.astype(float); amp=np.abs(np.fft.rfft(z))*2/len(z)
    centers,smooth=ko_smooth(freq,amp); specs.append((amp,centers,smooth))
acc=window(EVENTS[0],'ACC'); periods=np.logspace(-1,0,45); psa=psa_newmark(acc.data,acc.stats.delta,periods)

fig,ax=plt.subplots(4,1,figsize=(10,12),constrained_layout=True)
t=traces[0].times()-BEFORE; ev0,tr0=processed[0]
ax[0].plot(t,ev0['_raw_counts'],color='#555555',lw=.55)
ax[0].set(xlim=(-60,120),xlabel='Time from published event time (s)',ylabel='Raw counts (DU)',title='(a) KO-26 raw Apollo 15 SPZ counts'); ax[0].grid(alpha=.25)
ax[1].plot(t,tr0.data*1e6,color='#1f4e79',lw=.55)
ax[1].axvline(0,color='black',ls='--',lw=.8,label='published event time; not a picked P arrival')
ax[1].set(xlim=(-60,120),xlabel='Time from published event time (s)',ylabel='Velocity (µm/s)',title='(b) Response-removed velocity, 1-10 Hz'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
for (ev,tr),(amp,centers,sm),color in zip(processed,specs,['#1f4e79','#c55a11']):
    q=(freq>=FMIN)&(freq<=FMAX); ax[2].loglog(freq[q],amp[q]*1e6,color=color,alpha=.22,lw=.45); ax[2].loglog(centers,sm*1e6,color=color,lw=1.5,label=f"{ev['id']} Konno-Ohmachi")
ax[2].set(xlabel='Frequency (Hz)',ylabel='Fourier amplitude (µm/s)',title='(c) Fourier amplitude spectra'); ax[2].legend(fontsize=8); ax[2].grid(which='both',alpha=.2)
ax[3].loglog(periods,psa/9.80665,color='#1f4e79',label='KO-26, 5% damping'); ax[3].set(xlabel='Oscillator period (s)',ylabel='Pseudo-spectral acceleration (g)',title='(d) Response spectrum from KO-26 acceleration'); ax[3].legend(fontsize=8); ax[3].grid(which='both',alpha=.2)
fig.suptitle('Apollo short-period records: exploratory processing',fontsize=14,fontweight='bold')
fig.savefig(OUT/'Apollo_SP_event_processing.png',dpi=600); fig.savefig(OUT/'Apollo_SP_event_processing.pdf'); plt.close(fig)

# H1-pair diagnostic is separate because the event-time proxies are not picks.
fig,ax=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
for (ev,tr),color in zip(processed,['#1f4e79','#c55a11']):
    a=tr.data.astype(float); ax[0].plot(tr.times()-BEFORE,a/(np.max(np.abs(a))+1e-30),lw=.55,alpha=.8,color=color,label=ev['id'])
ax[0].set(xlabel='Time from published event-time proxy (s)',ylabel='Normalized velocity',title='12-minute H1 event windows, 1-10 Hz'); ax[0].legend(); ax[0].grid(alpha=.25)
ax[1].plot(lags[valid]/traces[0].stats.sampling_rate,c[valid],color='#7030a0')
ax[1].axhline(0,color='black',lw=.6); ax[1].scatter([lag],[cc],color='red',zorder=3,label=f'max |CC|={cc:.3f} at {lag:.3f} s')
ax[1].set(xlabel='Lag (s)',ylabel='Normalized cross-correlation',title='Exploratory full-window cross-correlation (±5 s search)'); ax[1].legend(); ax[1].grid(alpha=.25)
fig.suptitle('H1 waveform comparison: event-time proxies, not P-arrival alignment',fontsize=13,fontweight='bold')
fig.savefig(OUT/'Apollo_H1_pair_compare.png',dpi=600); fig.savefig(OUT/'Apollo_H1_pair_compare.pdf'); plt.close(fig)
result={
 'data_source':'NASA PDS Geosciences Apollo PSE MiniSEED and StationXML','archive_doi':'10.17189/9ykc-er91',
 'channel':'XA.S15..SHZ','sample_rate_hz':float(traces[0].stats.sampling_rate),'filter_hz':[FMIN,FMAX],
 'pre_filter_hz':list(PREF),'window_s':{'before':BEFORE,'after':AFTER,'total':BEFORE+AFTER},
 'events':[{'id':ev['id'],'utc':ev['utc'],'npts':tr.stats.npts,'duration_s':float(tr.stats.endtime-tr.stats.starttime),'missing_sample_fraction':ev['missing_sample_fraction']} for ev,tr in processed],
 'max_abs_normalized_cc_within_5s':cc,'lag_s':lag,
 'missing_data':'PDS MiniSEED missing-sample sentinel -1 was identified and isolated one-sample gaps were linearly interpolated before filtering; missing fraction is reported per event.',
 'timing_caveat':'Published event times are not P-arrival picks. This windowed cross-correlation is exploratory, is sensitive to non-event background, and is not evidence of a repeating source.',
 'response_spectrum':'5% damped PSA via Newmark-beta average acceleration, 0.1-1.0 s periods, 1-10 Hz filtered ground acceleration; interpretation beyond band-supported periods is excluded.',
 'figure':'Apollo_H1_pair_compare.png','processing_figure':'Apollo_SP_event_processing.png'}
(OUT/'Apollo_H1_processing_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
