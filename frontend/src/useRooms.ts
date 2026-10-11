import { useEffect, useState } from 'react';
import { DEFAULT_ROOM, request } from './api';

export type RoomConfig = { room_ids: string[]; gesture_hold_seconds: number | null };

// The monitor's configured rooms and the help-gesture hold time residents are told about.
export function useRooms(base: string) {
  const [config, setConfig] = useState<RoomConfig>({ room_ids: [DEFAULT_ROOM], gesture_hold_seconds: null });
  useEffect(() => {
    let active = true;
    request<RoomConfig>(base, '/rooms').then(data => { if (active && data.room_ids?.length) setConfig(data); }).catch(() => {});
    return () => { active = false; };
  }, [base]);
  return config;
}
