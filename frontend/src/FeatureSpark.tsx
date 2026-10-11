import { useEffect, useRef, useState } from 'react';

export default function FeatureSpark({ value, sampleId, max, preview }: { value: number | null; sampleId: string; max: number; preview: boolean }) {
  const previous = useRef(sampleId);
  const [samples, setSamples] = useState<number[]>(value == null ? [] : [value]);
  useEffect(() => {
    if (previous.current === sampleId) return;
    previous.current = sampleId;
    if (value != null) setSamples(old => [...old, value].slice(-60));
  }, [sampleId, value]);
  const points = value == null ? '' : preview ? `0,${34 - value / max * 32} 100,${34 - value / max * 32}`
    : samples.map((sample, index) => `${index * 100 / 59},${34 - sample / max * 32}`).join(' ');
  return <svg className="spark" viewBox="0 0 100 36" preserveAspectRatio="none" aria-label={preview ? 'Static example feature' : 'Recent reported feature samples'} role="img"><polyline points={points} fill="none" stroke="var(--amber)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" /></svg>;
}
