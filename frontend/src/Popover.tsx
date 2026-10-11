import { useEffect, useRef, useState, type ReactNode } from 'react';

// A button that opens a panel. The panel closes on a click elsewhere, on Escape, on the
// button again, or when its content calls close() (e.g. after Generate).
export default function Popover({ label, title, className = '', align = 'left', children }: {
  label: ReactNode; title?: string; className?: string; align?: 'left' | 'right' | 'up';
  children: (close: () => void) => ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const close = () => { setOpen(false); trigger.current?.focus(); };
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') close(); };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape); };
  }, [open]);
  return <div className={`popover ${className}`} ref={root}>
    <button ref={trigger} type="button" aria-expanded={open} title={title} className={open ? 'selected' : ''} onClick={() => setOpen(value => !value)}>{label}</button>
    {open && <div className={`popover-panel ${align}`} role="dialog" aria-label={title}>{children(close)}</div>}
  </div>;
}
