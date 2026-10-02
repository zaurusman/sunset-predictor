# Motion & sky refresh — design

**Date:** 2026-10-02 · **Status:** approved in brainstorming (mockups: Option 2 v3.2, 7-day v1)

## Goal

Make Afterglow feel alive and current without taking focus from the data: a soft
sky band that reflects the evening's score, one authored "sunset" moment on
Tonight, and purposeful motion everywhere else. Every existing element and
behaviour stays.

## Delivery: two PRs

1. **`chore/next-16`** — upgrade Next 15.1.11 → 16.3.x (React 19.2). No visual change.
   - `next lint` is removed in 16: switch the `lint` script to the ESLint CLI
     (codemod `next-lint-to-eslint-cli`).
   - Browser floor becomes Safari/iOS 16.4+, Chrome/Firefox 111+ (accepted).
   - Docker `node:20-alpine` already satisfies Node ≥ 20.9.
   - Re-check the `experimental.optimizePackageImports: []` workaround still builds.
   - Verify: `type-check`, lint, `next build`, all three pages + sheets on a Vercel preview.
2. **`feature/motion-sky`** — everything below, built on PR 1.

## Visual system

### Sky band
- A fixed layer behind all content, owned by the root layout (so it persists and
  cross-fades across tab changes). Pages tell it which mood to show.
- **Pastel** palettes per category (top ~45% of the screen, fading into the page):
  Poor = grey, Decent = dusty sand, Good = peach, Great = coral-peach, Epic = lavender→pink→peach.
- **Dark mode** (default proposed here, not yet seen by the user): the same hue
  families at low luminance over `slate-950` — e.g. Epic `#2e1f4d → #3b1d3a → #3a2418`,
  Poor `#1e293b → #0f172a`. Text contrast must stay ≥ 4.5:1.
- What each page shows:
  - **Tonight** → tonight's category.
  - **7 days** → the selected day's category (follows chart/card selection).
  - **History** → a neutral warm dusk (it is about the past, not one evening).
- Mood changes cross-fade over ~900 ms.

### Chrome
- Location pill, theme toggle, tab bar, date button: **frosted glass**
  (`white/70` + backdrop blur) so the sky reads through them.
- Sticky header (logo row + tabs); on scroll it gains a frosted background.
- The orange wordmark/brand text uses the darker brand orange (`#9a3412`) where it sits on the sky.
- Cards stay solid white (dark: `slate-900`), unchanged layout.

## Motion

Easing for arrivals: `cubic-bezier(.16,1,.3,1)`. Exits faster than entrances.
Content is readable from the first frame; no animation gates information.

### Tonight
- **The sunset (first open of the day only):** day-blue sky → a large centred sun
  sinks (~1.95 s) behind the frosted chrome to the top edge of the score card →
  a line of light flares along that edge → the score starts filling → the sky
  **cross-fades** from day into tonight's palette (~2.2 s, slow-in, like dusk;
  no radial reveal). Epic nights then get a very slow drifting warm glow.
  "First open of the day" is stored per date in `localStorage`; later opens get
  the quick version (sky cross-fade only, score count-up).
- **Score ring:** gradient arc with a glowing tip that travels as the number
  counts up. The **label (Poor…Epic) and colours are set to the final verdict
  from the first frame**; only the number counts.
- **Headline:** words rise from under a mask, ~70 ms apart.
- **Cards:** rise in, ~70 ms apart (total stagger capped at 280 ms).
- **Changing evening** (date picker, location, fresh data replacing cached): sky
  cross-fades, ring and number glide to the new value, headline re-rises.
- **When to look:** curve reveals left→right, a dashed marker sweeps to the
  peak, the peak dot pops and pulses once, peak time emphasises.
- **Rating** (exactly as approved in v3.2), wired to the real component:
  - Tap: button bounces and fills with its rating's soft colour, dark ink
    (Nothing grey · Dull stone · Pleasant peach · Very good peach→pink ·
    Exceptional lavender→pink→peach); 12 rays burst; other options step back.
  - The tap animation plays immediately; the card turns into the confirmation
    when the server confirms. On error the fill reverts, options return, error shows.
  - Confirmation: check ring + mark draw in the rating's colour, chip pops,
    card height animates to fit. Content keeps today's logic: server message or
    "The model said N — off by a lot. Logged.", the coffee link after an
    "Exceptional" rating, and "Change my rating".
  - A rating restored from storage on load shows the confirmation without animation.
- Other Tonight pieces (date picker, evidence drawer, install prompt, error
  alert, photo button) get the shared overlay/drawer motion below.

### 7 days
- Arriving: bars grow up from the baseline (~60 ms apart), cards rise in, today's card opens.
- **Chart ↔ cards linked (new):** tapping a bar selects that day → its bar is
  emphasised (others at ~32% opacity), the sky shifts to its palette, its card
  opens (others close) and scrolls into view below the sticky header.
- Tapping a card toggles it and also selects it (chart + sky follow).
- Each card: small score ring (matches Tonight), date, label, sunset time, chevron.
- Opening a card: height animates (grid-rows), then "When to look" draws,
  reasons appear, breakdown/pathway bars fill. All existing content kept:
  When to look, Why, Breakdown (components + weights), Ways it could be
  beautiful, Holding it back.
- Heads-up note, algorithm version and support footer kept.

### History
- Heatmap cells fade in as a diagonal wave by week column (total ≤ 600 ms);
  month bars grow; year buttons and progress bar keep current behaviour.

### Shared
- **Tab changes:** React `<ViewTransition>` (Next 16) — content slides ~20 px in
  the direction of travel with a slight blur; the tab highlight is a shared
  element that stretches as it moves.
- **Sheets** (location, iOS install): slide up from the bottom with a fading
  backdrop; exit ~220 ms. **Modal** (photo submit): fade + scale from 0.96.
- **Drawers / dropdowns** (evidence drawer, date picker): height via grid-rows, content fades in.
- **Buttons:** 0.95 press-down on tap.
- **Loading / errors:** existing states kept; alerts slide down and fade in.

### Reduced motion
Follows the OS `prefers-reduced-motion`. Spatial movement, the sun, rays and
the drifting glow are removed; colour/opacity changes and state confirmations
remain as short fades. No in-app toggle (the demo toggle was for preview only).

## Architecture

- `lib/motion.ts` — easing/duration tokens, `useReducedMotion()`,
  `useCountUp()`, `shouldPlaySunset(date)` / `markSunsetPlayed(date)`.
- `lib/sky.ts` — category → light/dark palette.
- `components/sky/SkyProvider.tsx` + `SkyBackdrop.tsx` — context in the root
  layout: `setMood(category)`, `playSunset(horizonEl)`; renders sky layers, sun,
  horizon flare, glow.
- Updated: `AppNav` (frosted, sticky, shared tab highlight), `VerdictCard`
  (ring/tip/count-up/headline), `ViewingCurve` (reveal/scrubber/peak),
  `RateSunset` (new moment, same data flow), `ForecastChart` (grow, selection,
  tap→select), `SunsetCard` (controlled expansion from the page, mini ring,
  animated open), `ComponentBreakdown` (bars fill on reveal), `HeatmapGrid`,
  sheets/modal/drawers, `globals.css` (view-transition CSS, keyframes,
  reduced-motion overrides).
- No new runtime dependencies.

## Out of scope

The "full visual redesign" (new typography, glass cards, layout changes) —
possible follow-up. No changes to scores, API, or data.

## Verification

`type-check`, lint and `next build` pass; every page checked in the browser at
375 px and desktop, light and dark, with and without reduced motion; rating
tested against the local backend (success and failure); 7-day chart→card
linking; no console errors; Impeccable hook findings triaged.
