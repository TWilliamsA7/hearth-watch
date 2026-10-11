import type { ActorCommand, LiveSlot, LiveWorld } from "./generated/sim";

/** What a click in the 3D view landed on, besides a person: a slot on a piece of furniture, or open floor. */
export type Pick = { kind: "slot"; slotId: string; x: number; z: number } | { kind: "floor"; x: number; z: number };

/** The slot on piece `piece` (an index into room.objects) nearest a point on it, if it has any. */
export function nearestSlot(slots: readonly LiveSlot[], piece: number, x: number, z: number): LiveSlot | null {
  let best: LiveSlot | null = null;
  let bestD = Infinity;
  for (const s of slots) {
    if (s.piece !== piece) continue;
    const d = Math.hypot(s.at[0] - x, s.at[1] - z);
    if (d < bestD) {
      bestD = d;
      best = s;
    }
  }
  return best;
}

/** What someone can be told to do at a pick, as (button label, command). */
export function actionsFor(pick: Pick, world: LiveWorld): [string, ActorCommand][] {
  if (pick.kind === "floor") {
    const at: [number, number] = [Math.round(pick.x * 1000) / 1000, Math.round(pick.z * 1000) / 1000];
    return [
      ["Go here", { action: "go", at, slot_id: null }],
      ["Kneel here", { action: "kneel", at, slot_id: null }],
      ["Pick something up here", { action: "pick_up", at, slot_id: null }],
      ["Fall here", { action: "fall", at, slot_id: null }],
    ];
  }
  const slot = world.slots.find((s) => s.slot_id === pick.slotId);
  if (!slot) return [];
  const lie = world.slots.some((s) => s.group === slot.group && s.kind === "lie");
  const out: [string, ActorCommand][] = [];
  if (slot.kind === "sit") out.push(["Sit here", { action: "sit", slot_id: slot.slot_id, at: null }]);
  if (lie) out.push(["Lie down here", { action: "lie", slot_id: slot.slot_id, at: null }]);
  return out;
}

/** A short name for what was picked, for the menu's title. */
export function describePick(pick: Pick, world: LiveWorld): string {
  if (pick.kind === "floor") return `floor (${pick.x.toFixed(2)}, ${pick.z.toFixed(2)})`;
  const slot = world.slots.find((s) => s.slot_id === pick.slotId);
  const kind = slot ? world.room.objects[slot.piece]?.variant ?? world.room.objects[slot.piece]?.kind : null;
  return kind ? `${kind} · ${pick.slotId}` : pick.slotId;
}
