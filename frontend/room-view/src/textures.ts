import * as THREE from "three";

/**
 * Procedural textures drawn on canvases, so the room needs no image assets: wood-plank
 * floors, painted walls, a patterned rug. Each is made once per look and shared.
 */

/** A small seeded PRNG, so textures look the same every time. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const cache = new Map<string, THREE.Texture>();

function canvasTexture(key: string, size: number, draw: (ctx: CanvasRenderingContext2D, size: number) => void): THREE.Texture {
  const hit = cache.get(key);
  if (hit) return hit;
  if (typeof document === "undefined") return new THREE.Texture(); // no DOM (tests): untextured
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  draw(ctx, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  cache.set(key, tex);
  return tex;
}

/** Oak planks, about 2 m of floor per tile. */
export function woodFloor(): THREE.Texture {
  return canvasTexture("wood", 1024, (ctx, n) => {
    const rnd = mulberry32(7);
    const rows = 8;
    const h = n / rows;
    for (let r = 0; r < rows; r++) {
      let x = -rnd() * n * 0.5;
      while (x < n) {
        const len = n * (0.35 + rnd() * 0.45);
        const tone = 158 + rnd() * 36;
        ctx.fillStyle = `rgb(${tone + 18}, ${tone * 0.74}, ${tone * 0.48})`;
        ctx.fillRect(x, r * h, len, h);
        // Warm oak with visible grain, even when viewed from across the room.
        for (let g = 0; g < 24; g++) {
          ctx.strokeStyle = `rgba(70, 40, 20, ${0.08 + rnd() * 0.10})`;
          ctx.lineWidth = 1 + rnd() * 2;
          const y = r * h + rnd() * h;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.bezierCurveTo(x + len * 0.3, y + (rnd() - 0.5) * 6, x + len * 0.7, y + (rnd() - 0.5) * 6, x + len, y);
          ctx.stroke();
        }
        ctx.fillStyle = "rgba(40, 24, 12, 0.55)"; // the joint between planks
        ctx.fillRect(x + len - 2, r * h, 2, h);
        x += len;
      }
      ctx.fillStyle = "rgba(40, 24, 12, 0.6)";
      ctx.fillRect(0, r * h, n, 2);
    }
  });
}

/** Matte paint with a faint roller texture. */
export function paintedWall(): THREE.Texture {
  return canvasTexture("wall", 512, (ctx, n) => {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, n, n);
    const rnd = mulberry32(3);
    for (let i = 0; i < 9000; i++) {
      const v = 235 + rnd() * 20;
      ctx.fillStyle = `rgba(${v}, ${v}, ${v}, 0.35)`;
      ctx.fillRect(rnd() * n, rnd() * n, 2 + rnd() * 3, 2 + rnd() * 3);
    }
  });
}

/** A bordered rug in one of a few palettes. */
export function rugTexture(variant: number): THREE.Texture {
  const palettes = [
    ["#7b3b3b", "#d9c2a0", "#3d4f66"],
    ["#2f5d62", "#e8dcc2", "#a65e3a"],
    ["#5a4a78", "#e6d7b8", "#7e8f55"],
  ];
  const [base, line, accent] = palettes[variant % palettes.length];
  return canvasTexture(`rug${variant % palettes.length}`, 512, (ctx, n) => {
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, n, n);
    ctx.strokeStyle = line;
    ctx.lineWidth = 14;
    ctx.strokeRect(28, 28, n - 56, n - 56);
    ctx.lineWidth = 4;
    ctx.strokeRect(54, 54, n - 108, n - 108);
    ctx.fillStyle = accent;
    for (let i = 0; i < 5; i++) {
      const c = n / 2;
      const r = 40 + i * 28;
      ctx.beginPath();
      ctx.moveTo(c, c - r);
      ctx.lineTo(c + r, c);
      ctx.lineTo(c, c + r);
      ctx.lineTo(c - r, c);
      ctx.closePath();
      ctx.globalAlpha = 0.18 + (i % 2) * 0.25;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  });
}

/** Fabric weave for upholstery (used as a bump-ish colour variation). */
export function fabric(): THREE.Texture {
  return canvasTexture("fabric", 256, (ctx, n) => {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, n, n);
    const rnd = mulberry32(11);
    for (let y = 0; y < n; y += 2) {
      ctx.fillStyle = `rgba(0, 0, 0, ${0.03 + rnd() * 0.05})`;
      ctx.fillRect(0, y, n, 1);
    }
    for (let x = 0; x < n; x += 2) {
      ctx.fillStyle = `rgba(0, 0, 0, ${0.02 + rnd() * 0.04})`;
      ctx.fillRect(x, 0, 1, n);
    }
  });
}
