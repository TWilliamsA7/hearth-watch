import { useEffect, useState } from 'react';

export default function LiveCameraView({base,source,room}: {base:string; source:string; room:string}) {
  const [image,setImage] = useState('');
  const [error,setError] = useState('Waiting for the annotated camera feed…');
  useEffect(() => {
    let disposed = false, current = '';
    let timer: ReturnType<typeof setTimeout>;
    const abort = new AbortController();
    setImage(''); setError('Waiting for the annotated camera feed…');
    async function next() {
      try {
        const response = await fetch(new URL(`/camera-preview/${encodeURIComponent(room)}/${encodeURIComponent(source)}`,base),{cache:'no-store',signal:AbortSignal.any([abort.signal,AbortSignal.timeout(3000)])});
        if (!response.ok) throw new Error('No current camera feed. Start Hearth with scripts/start.ps1 -Camera 0 to share the annotated webcam here.');
        const blob = await response.blob();
        if (disposed) return;
        if (current) URL.revokeObjectURL(current);
        current = URL.createObjectURL(blob); setImage(current); setError('');
      } catch (e) {
        if (!disposed) { if(current) URL.revokeObjectURL(current); current=''; setImage(''); setError((e as Error).message); }
      } finally { if (!disposed) timer=setTimeout(next,350); }
    }
    void next();
    return () => {disposed=true; abort.abort(); clearTimeout(timer); if(current) URL.revokeObjectURL(current);};
  },[base,source,room]);
  return <div className="live-camera">
    {image ? <img src={image} alt="Live camera with posture and hand detection bounding boxes"/> : <div className="simulation-message" role="status"><p>{error}</p></div>}
    <span className="simulation-badge">LIVE CAMERA · {source}</span>
  </div>;
}
