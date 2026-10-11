import { useEffect, useState, type ReactNode } from 'react';

// A camera's annotated frames when it shares them (the laptop webcam client); otherwise the
// skeletons it reports (the Jetson sends keypoints, never video).
export default function LiveCameraView({base,source,room,skeleton,hasPeople,overlay}: {base:string; source:string; room:string; skeleton:ReactNode; hasPeople:boolean; overlay?:(width:number,height:number)=>ReactNode}) {
  const [image,setImage] = useState('');
  const [size,setSize] = useState<[number,number] | null>(null);
  useEffect(() => {
    let disposed = false, current = '';
    let timer: ReturnType<typeof setTimeout>;
    const abort = new AbortController();
    setImage('');
    async function next() {
      try {
        const response = await fetch(new URL(`/camera-preview/${encodeURIComponent(room)}/${encodeURIComponent(source)}`,base),{cache:'no-store',signal:AbortSignal.any([abort.signal,AbortSignal.timeout(3000)])});
        if (!response.ok) throw new Error('no frames');
        const blob = await response.blob();
        if (disposed) return;
        if (current) URL.revokeObjectURL(current);
        current = URL.createObjectURL(blob); setImage(current);
      } catch {
        if (!disposed) { if(current) URL.revokeObjectURL(current); current=''; setImage(''); }
      } finally { if (!disposed) timer=setTimeout(next,current ? 350 : 2000); }
    }
    void next();
    return () => {disposed=true; abort.abort(); clearTimeout(timer); if(current) URL.revokeObjectURL(current);};
  },[base,source,room]);
  return <div className="live-camera">
    {image ? <><img src={image} alt="Live camera" onLoad={e => { const i = e.currentTarget; if (!size || size[0] !== i.naturalWidth || size[1] !== i.naturalHeight) setSize([i.naturalWidth, i.naturalHeight]); }}/>{overlay && size && overlay(size[0], size[1])}</> : <>
      {skeleton}
      {!hasPeople && <div className="simulation-message" role="status"><p>{source ? `Nobody in view of ${source} right now.` : 'No live camera is reporting. Start Hearth with scripts/start.ps1 (the Jetson starts with it).'}</p></div>}
    </>}
    <span className="simulation-badge">LIVE CAMERA · {source || 'none'}{image ? '' : ' · skeleton'}</span>
  </div>;
}
