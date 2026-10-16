"""Inventory preserved M0 sources and create timestamped inspection media."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, subprocess
from pathlib import Path
import cv2
import numpy as np

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path.cwd()); args=ap.parse_args()
    data=args.root/'data/personal/m0'; results=args.root/'results/personal-m0-20261009'
    results.mkdir(parents=True,exist_ok=True)
    local=json.loads((data/'source_inventory_local.json').read_text())
    rows=[]
    for source in local:
        rid=source['recording_id']; path=data/'raw'/source['filename']
        digest=sha256(path)
        assert digest==source['sha256'],f'Transfer checksum mismatch: {rid}'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)]))
        video=next(s for s in probe['streams'] if s['codec_type']=='video')
        a,b=map(int,video['avg_frame_rate'].split('/'))
        tags={**probe['format'].get('tags',{}),**video.get('tags',{})}
        row={'recording_id':rid,'filename':path.name,'relative_path':str(path.relative_to(args.root)),
             'file_size_bytes':path.stat().st_size,'sha256':digest,'duration_s':float(probe['format']['duration']),
             'width':video['width'],'height':video['height'],'average_frame_rate':a/b,
             'frame_rate_rational':video['avg_frame_rate'],'audio_present':any(s['codec_type']=='audio' for s in probe['streams']),
             'creation_timestamp':tags.get('creation_time',''),'camera_identity':'unknown','camera_confidence':'unreviewed'}
        rows.append(row)
        folder=results/'inspection'/rid; folder.mkdir(parents=True,exist_ok=True)
        (folder/'ffprobe.json').write_text(json.dumps(probe,indent=2))
        cap=cv2.VideoCapture(str(path)); next_sample=0.; frame_idx=0; samples=[]; tiles=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if t+1e-6>=next_sample:
                name=f'frame_{len(samples):06d}.jpg'
                cv2.imwrite(str(folder/name),frame,[cv2.IMWRITE_JPEG_QUALITY,94])
                samples.append({'frame':name,'source_time_s':t,'decoded_frame_index':frame_idx})
                if len(samples)%8==1:
                    thumb=cv2.resize(frame,(180,323))
                    cv2.rectangle(thumb,(0,0),(180,24),(0,0,0),-1)
                    cv2.putText(thumb,f'{rid[-4:]} t={t:.2f}',(3,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                    tiles.append(thumb)
                next_sample+=.5
            frame_idx+=1
        cap.release()
        assert samples,f'No decoded frames: {rid}'
        (folder/'sample_timestamps.json').write_text(json.dumps(samples,indent=2))
        cols=5; overview=np.full((math.ceil(len(tiles)/cols)*323,cols*180,3),235,np.uint8)
        for i,tile in enumerate(tiles):overview[(i//cols)*323:(i//cols+1)*323,(i%cols)*180:(i%cols+1)*180]=tile
        cv2.imwrite(str(folder/'overview.jpg'),overview,[cv2.IMWRITE_JPEG_QUALITY,94])
        if row['audio_present']:
            subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(path),'-vn','-ac','1','-ar','16000',str(folder/'audio.wav')],check=True)
        print(rid,round(row['duration_s'],3),'seconds',len(samples),'inspection frames',flush=True)
    with (data/'raw_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (data/'source_checksums.sha256').write_text(''.join(f"{r['sha256']}  raw/{r['filename']}\n" for r in rows))
    (data/'session_metadata.json').write_text(json.dumps({'schema_version':1,'status':'checkpoint_A_inventory_complete_pairing_pending',
        'source_count':len(rows),'sources_preserved_and_transfer_verified':True,'original_input_directory':'input/humanoid',
        'sampling_hz':2,'sampling_timestamps':'decoded presentation timestamps, no time interpolation',
        'canonical_orientation':'blue number 2 upright, studs towards top of primary image','records':rows},indent=2))
    (results/'checkpoint_A_inventory.md').write_text('# M0 source inventory\n\n'+f'{len(rows)} original recordings copied to simulation host and verified byte-for-byte against local SHA-256 hashes. '+
        'Camera identity, session pairing and trial counts remain pending visual inspection. WhatsApp exports are the supplied originals; original phone acquisition timestamps are not assumed from export filenames.\n')
    (results/'commands.json').write_text(json.dumps({'checkpoint_A':['.venv-sim/bin/python scripts/m0_prepare.py'],
        'libraries':{'opencv':cv2.__version__,'numpy':np.__version__},'source_manifest':'data/personal/m0/raw_manifest.csv'},indent=2))

if __name__=='__main__':main()
