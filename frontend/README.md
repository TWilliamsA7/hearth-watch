# Your dashboard starter

This is a React + TypeScript dashboard with clearly labeled local example data.
It does not yet connect to the backend, camera, or simulator.

## Start it

With Node.js 22 installed, open a terminal in `frontend` and run:

```sh
npm install
npm run generate:types
npm run dev
```

Open the local address printed by Vite (usually http://127.0.0.1:5173).
Use `npm run build` to check types and create a production build.
All assets are local; the dashboard does not need an external font service.

## What each file does

- `index.html`: the page containing the empty root where React mounts.
- `src/main.tsx`: starts React and loads the stylesheet.
- `src/App.tsx`: assembles dashboard panels and handles preview buttons and local acknowledgement.
- `src/RoomView.tsx`: draws normalized keypoints as an SVG skeleton.
- `src/demoData.ts`: creates example observations and events using the shared contract.
- `src/styles.css`: colors, spacing, cards, and mobile layout.
- `scripts/generate-types.mjs`: converts the shared JSON Schema into TypeScript types.
- `src/generated/contract.ts`: generated contract; do not edit manually.
- `package.json`: dependencies and commands.
- `tsconfig.json`: TypeScript checks.
- `DEMO.md`: a rehearsal checklist and short demo script.

## Next integration steps

Coordinate with role 2 for `WS /ws/room/room-01`. Consume `observation`,
`event`, and `source_status` envelopes. Use incoming observations for the skeleton
and backend events for alerts. On disconnect or stale data, clear the current
observation and show unknown/offline. Define reconnection behavior with role 2.

Ask role 2 for the alert acknowledgement endpoint and authoritative alert state.
Ask role 3 for scenario-control endpoints and evaluation response types.
These endpoints are not yet defined, so the starter does not invent them.
Keep simulation, live, and replay source labels visible after integration.
Face `null` means detection did not run; it is different from `false`.

The example alert is independent of the example skeleton. Preview buttons only
exercise UI states; they do not run fall detection or produce evaluation metrics.
