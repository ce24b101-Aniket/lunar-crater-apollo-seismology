"""Fetch known artificial-impact windows and event-day StationXML through EarthScope FDSN."""
from pathlib import Path
import requests,json
from obspy import UTCDateTime
ROOT=Path('work/data/apollo_impacts');ROOT.mkdir(parents=True,exist_ok=True)
base='https://service.earthscope.org/fdsnws/dataselect/1/query';meta='https://service.earthscope.org/fdsnws/station/1/query'
# Event times and impact sources are from NASA NSSDC's Apollo Impact Sites table.
events=[
 {'id':'Apollo14_LM','origin_utc':'1971-02-07T00:45:25.7','source':'Apollo 14 LM ascent stage','lat':-3.42,'lon':-19.67},
 {'id':'Apollo15_SIVB','origin_utc':'1971-07-29T20:58:42.9','source':'Apollo 15 S-IVB','lat':-1.2897,'lon':-11.8245},
 {'id':'Apollo16_SIVB','origin_utc':'1972-04-19T21:02:04.0','source':'Apollo 16 S-IVB','lat':1.921,'lon':-24.623,'time_uncertainty_s':4},
 {'id':'Apollo17_SIVB','origin_utc':'1972-12-10T20:32:42.3','source':'Apollo 17 S-IVB','lat':-4.1681,'lon':-12.3307},
]
S=requests.Session(); log=[]
for e in events:
 t=UTCDateTime(e['origin_utc']); start=t-300; end=t+7200
 e['station']='S14'; e['channel']='SHZ'; e['window_utc']=[str(start),str(end)]
 for cha,suffix in [('SHZ','mseed'),('ATT','att.mseed')]:
  params={'net':'XA','sta':'S14','loc':'*','cha':cha,'starttime':str(start),'endtime':str(end),'format':'miniseed'}
  r=S.get(base,params=params,timeout=180)
  if r.status_code==200 and r.content:
   out=ROOT/f"{e['id']}_S14_{suffix}";out.write_bytes(r.content);e[cha+'_file']=str(out);e[cha+'_bytes']=len(r.content)
  else:
   e[cha+'_status']=r.status_code;e[cha+'_response']=r.text[:250]
 params={'net':'XA','sta':'S14','loc':'*','cha':'SHZ','starttime':str(t-3600),'endtime':str(t+7200),'level':'response','format':'xml'}
 r=S.get(meta,params=params,timeout=60)
 if r.status_code==200 and r.content:
  out=ROOT/f"{e['id']}_S14_station.xml";out.write_bytes(r.content);e['response_file']=str(out);e['response_bytes']=len(r.content)
 else:e['response_status']=r.status_code;e['response_error']=r.text[:250]
 log.append(e);print(json.dumps(e,indent=1))
(ROOT/'download_manifest.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
