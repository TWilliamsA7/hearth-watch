# Designing Hearth and bringing Adobe artwork into the app

The working website is React plus CSS. Git tracks its files, and GitHub lets your
team share them. A design file is a reference; GitHub does not convert it into a
working dashboard. This project currently runs locally for the hackathon demo.

## What the dashboard design does

The dashboard keeps the supplied reference layout with a solid warm beige page,
baby blue cards, navy borders, dark gray text, muted purple secondary text,
lavender accents, red alerts, soft green status indicators, light beige rooms,
and a dark purple floor. The Caregiver/Patient switch and Simulate fall button
use lavender backgrounds.

General interface text uses the locally bundled Commissioner font. The teal
HearthWatch wordmark uses Windows Lucida Calligraphy, with a drawn heart flourish
connected to the final h. Georgia remains on the large measurements, patient
greeting, SOS label, and appointment times. The flame logo has been removed.

Palette values in `src/styles.css`: bg #F7F1E8, card #EDF5FA, line #253C63,
text #34383E, mute #776A8A, amber #B6A2D9, ember #C94343, ok #92BEA0,
room #F4ECDD, floor #514064, brand #247C7F.

Connection settings and simulator/evaluation controls are compact header menus.
Alerts can be reviewed and acknowledged from the event-log panel. The dashboard
fills the desktop viewport; small phones can scroll without losing controls.

The illustrated room appears only in preview. Connected views show observed
keypoints/model bounds in camera coordinates, not video or a measured floor map.
Preview posture samples do not run a fall detector. Alerts remain backend-owned.
Hip position is normalized image height, not centimeters. Incident age is elapsed
time since the server's alert timestamp for live camera incidents only, not proof
of time on the floor. Simulation/replay incident ages stay blank because those
timestamps can differ from wall time. No clinical threshold marker is invented.

Patient help posts to the real monitoring help-request API. Receipt is distinct
from caregiver response. The reference's Margaret and appointment details appear
only in preview and are labeled as examples. Connected views have no invented
appointments or check-in confirmations. Scheduling remains unconnected.

The preview timer and static feature plots are labeled as examples. Connected
feature plots show samples actually reported by the backend. There are no
centimeter measurements or claims that caregivers have been notified.

Source type remains visible. Unknown observations do not appear safe. Preview
examples and measured results remain clearly different. No decorative charts or
invented metrics are used. Buttons retain their backend actions.

## A simple Adobe-to-code workflow

1. **Plan the screen.** Sketch where the room, alerts, and controls belong. You can
   use an Adobe document or an image as a reference; no particular Adobe app is
   required by this codebase.
2. **Choose colors and type.** Use the CSS variables at the top of `src/styles.css`
   to update the palette. System fonts keep the demo independent of internet font
   services. For another font, include a licensed local font file and use `@font-face`.
3. **Export artwork only.** For an Illustrator logo, export SVG when appropriate.
   For a Photoshop illustration, export an appropriately sized PNG, JPEG, or WebP.
   Keep the editable Adobe source as your design reference.
4. **Add exported artwork to `frontend/public/assets/`.** A file named
   `hearth-logo.svg` there can be referenced as `/assets/hearth-logo.svg` in React.
   For example: `<img src="/assets/hearth-logo.svg" alt="Hearth" />`.
5. **Build the layout in React and CSS.** `App.tsx` determines the page structure,
   `styles.css` its appearance, and the existing components retain functionality.
   A screenshot of a button is only an image; a React button can respond to clicks.
6. **Review locally.** Run `npm.cmd run dev`, inspect desktop and narrow/mobile
   layouts, and try preview alerts/offline states. Then connect the real backend.
7. **Share through Git.** Review changed files, commit the design and exported assets
   on your dashboard branch, then push that branch or open a pull request. Avoid
   adding dependency folders or build output to the commit.

If you have a design screenshot, logo export, or reference, use it to describe the
specific colors, spacing, and layout you want changed. The current design can be
adjusted without rebuilding the monitoring integrations.

## Current palette

| Role | Color | CSS variable |
| --- | --- | --- |
| Page background | `#1A1310` | `--bg` |
| Cards | `#251A15` | `--card` |
| Borders | `#3B2B22` | `--line` |
| Text | `#F5E6D6` | `--text` |
| Secondary text | `#A58F7E` | `--mute` |
| Amber | `#F5A54A` | `--amber` |
| Ember | `#F0643F` | `--ember` |
| Green | `#7FC88A` | `--ok` |
| Room | `#2B1E17` | `--room` |
| Floor | `#3A2A20` | `--floor` |

Use color alongside text and icons, never as the only indication of an alert or
connection state. Focus outlines support keyboard navigation. Motion is disabled
when the user's system prefers reduced motion.

## Caregiving photography

A short introductory image can show an older adult receiving respectful support
from a caregiver. Keep the person an active participant in the scene and use a
photograph you have permission to use. Add it beside the introduction, rather
than a full-screen cover that pushes alerts and the room observation out of view.
Use descriptive alternative text when the photograph conveys useful information;
use empty alternative text if it is purely decorative. No stock photo or AI-made
person has been added to this dashboard. The assets folder is ready for your own
Adobe export; the room view and alerts remain the primary content.

Readability update: pale baby blue cards (#EDF5FA), larger 14–16px interface
labels, a 34px bold Lucida wordmark with a heavier heart flourish, a 16px
caption, and larger measurement values. The caption remains visible on narrow
screens.

Cards now use glacial blue (#E3F0F3); the brand caption is removed.
The original 60 degree red marker is shown only in UI preview, using a 0–90
degree bar. It is a reference example, not the backend detector threshold.
Hip height from floor is labeled unavailable because only image coordinates
are supplied. Preview Time down shows its example status; disconnected data
remains unknown. Metric captions stay visible at narrow widths.
