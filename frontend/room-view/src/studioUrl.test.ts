import { describe, expect, it } from "vitest";

import type { EngineSpec } from "./generated/sim";
import { DEFAULT_OVERLAYS } from "./LiveView3D";
import { readSettings, roomDiffers, writeSettings, type StudioSettings } from "./studioUrl";

const DEFAULTS: StudioSettings = {
  seed: 7, actors: 6, width: 7, depth: 6, density: 0.6, preset: "corner_left", camHeight: 2.4, camTilt: 12, camFov: 62,
  lighting: "day", activity: 1, social: 1, overlays: DEFAULT_OVERLAYS, cameraMode: "orbit",
};

describe("studio settings in the URL", () => {
  it("writes nothing for the defaults", () => {
    expect(writeSettings(DEFAULTS, DEFAULTS)).toBe("");
    expect(readSettings("", DEFAULTS)).toEqual(DEFAULTS);
  });

  it("round-trips what was changed", () => {
    const s: StudioSettings = {
      ...DEFAULTS, seed: 42, actors: 3, density: 0.35, preset: "jetson_corner", camFov: 43, lighting: "dim",
      cameraMode: "top", overlays: { ...DEFAULT_OVERLAYS, slots: true, paths: false },
    };
    const search = writeSettings(s, DEFAULTS);
    const q = new URLSearchParams(search);
    expect(q.get("seed")).toBe("42");
    expect(q.has("w")).toBe(false); // unchanged: not written
    expect(readSettings(search, DEFAULTS)).toEqual(s);
  });

  it("ignores what it can't use", () => {
    const s = readSettings("?seed=abc&cam=nowhere&light=blinding&view=sideways&people=", DEFAULTS);
    expect(s).toEqual(DEFAULTS);
  });

  it("sees when the running room isn't the linked one", () => {
    const spec = { actors: 6, room: { width: 7, depth: 6, furniture_density: 0.6 } } as EngineSpec;
    expect(roomDiffers(DEFAULTS, { seed: 7, spec })).toBe(false);
    expect(roomDiffers(DEFAULTS, { seed: 8, spec })).toBe(true);
    expect(roomDiffers({ ...DEFAULTS, actors: 2 }, { seed: 7, spec })).toBe(true);
    expect(roomDiffers({ ...DEFAULTS, lighting: "dim" }, { seed: 7, spec })).toBe(false); // the view, not the room
  });
});
