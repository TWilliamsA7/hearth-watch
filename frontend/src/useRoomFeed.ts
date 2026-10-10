import { useEffect, useState } from 'react';
import { emptyFeed, reduceFeed, type SourceMode } from './feedState.ts';
import { parseMessage } from './protocol.ts';
import { websocketUrl } from './api.ts';

export function useRoomFeed(base: string | null, mode: SourceMode) {
  const [state, setState] = useState(emptyFeed);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  const [now, setNow] = useState(Date.now());
  const [session, setSession] = useState('');
  useEffect(() => {
    setState(emptyFeed()); setConnected(false); setError(''); setSession(`${base}:${mode}`);
    if (!base) return;
    let disposed = false;
    let socket: WebSocket | undefined;
    let retry: ReturnType<typeof setTimeout>;
    let backoff = 500;
    const tick = setInterval(() => setNow(Date.now()), 250);
    function connect() {
      if (disposed) return;
      // Clear authoritative state before each reconnect; the server resends its snapshot.
      setState(emptyFeed());
      socket = new WebSocket(websocketUrl(base!));
      socket.onopen = () => { if (!disposed) { setConnected(true); setError(''); backoff = 500; } };
      socket.onmessage = event => {
        if (disposed) return;
        try {
          const message = parseMessage(JSON.parse(event.data));
          if (!message) { setError('An unsupported message was ignored.'); return; }
          setState(previous => reduceFeed(previous, message, Date.now(), mode));
        } catch { setError('An unreadable message was ignored.'); }
      };
      socket.onerror = () => { if (!disposed) setError('Cannot reach the backend. Check its address and that it is running.'); };
      socket.onclose = () => {
        if (disposed) return;
        setConnected(false);
        retry = setTimeout(connect, backoff);
        backoff = Math.min(backoff * 2, 5000);
      };
    }
    connect();
    return () => { disposed = true; clearInterval(tick); clearTimeout(retry); socket?.close(); };
  }, [base, mode]);
  const current = session === `${base}:${mode}`;
  return { state: current ? state : emptyFeed(), connected: current && connected, error: current ? error : '', now };
}
