# Designing Hearth and bringing Adobe artwork into the app

The working website is React plus CSS. Git tracks its files, and GitHub lets your
team share them. A design file is a reference; GitHub does not convert it into a
working dashboard. This project currently runs locally for the hackathon demo.

## What the dashboard design does

The foundation is warm off-white and beige, with muted teal/sage for monitoring
and lavender for quiet accents. Text is dark slate, not pure black. Cards use
warm off-white, not pure white. Main reading text is 16px, secondary text is at
least 14px, and buttons/fields have a minimum height of 44px. Use the browser's
zoom controls for larger text; responsive layouts adapt to the narrower viewport.

The left navigation links to real sections on the same page: room overview,
incident alerts, scenarios, and evaluation. The overview stays focused on what is
observed and what requires review. The connection form is inside Connection
settings. The room is the largest panel, alerts sit beside it, and scenario controls
and measured results come below. Mobile screens stack the same sections.

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
| Page background | `#F2EEE6` | `--page` |
| Cards | `#FBFAF6` | `--surface` |
| Primary text | `#35474B` | `--ink` |
| Secondary text | `#5B696B` | `--muted` |
| Main actions | `#476D70` | `--green` |
| Soft sage | `#E6EDE7` | `--green-soft` |
| Lavender accents | `#71627E` | `--lavender` |
| Soft lavender | `#EEEBF2` | `--lavender-soft` |
| Incident accents | `#8B542D` | `--amber` |
| Incident background | `#F9EEDF` | `--amber-soft` |

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
