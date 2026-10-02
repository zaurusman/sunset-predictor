# Next 16 Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the frontend from Next 15.1.11 to Next 16.3.x (React 19.2) with no visible change, so the motion work can use React's `<ViewTransition>`.

**Architecture:** Dependency bump plus the two mandatory migrations that touch this app (`next lint` removal → ESLint CLI with a flat config; Turbopack becomes the default bundler). No application code is expected to change: the app uses only client-side `useSearchParams`, has no middleware, no `next/image`, no server `cookies()`/`headers()`.

**Tech Stack:** Next.js 16.3.x, React 19.2, TypeScript 5.9, ESLint 9 (flat config), Tailwind 3.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-02-motion-and-sky-design.md` → "Delivery: two PRs", item 1.
- Branch: `chore/next-16`, based on `origin/main` (e7445a2). Finish with a PR to `main`; never push to `main`.
- No visual or behavioural change in this PR.
- Browser floor becomes Safari/iOS 16.4+, Chrome/Firefox 111+ (accepted by the user).
- **The shell exports `NODE_ENV=production` globally.** Every `npm`/`next` command below runs under `env -i HOME="$HOME" PATH="$PATH" LANG=en_US.UTF-8` — otherwise `npm ci` skips devDependencies and `next dev` misbehaves.
- Stage files explicitly; never commit `.next/`, `node_modules/`, `.superpowers/`, `.impeccable/`.
- All commands run from the worktree: `/Users/yotamtsabari/sunset-predictor/.claude/worktrees/programming-design-skills-2cbe5c`.

Shorthand used below:

```bash
# zsh does not word-split a variable, so use a tiny wrapper script rather than CLEAN='env -i …'.
printf '#!/bin/sh\nexec env -i HOME="$HOME" PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" LANG=en_US.UTF-8 "$@"\n' > /tmp/clean && chmod +x /tmp/clean
CLEAN=/tmp/clean   # then: $CLEAN npm ci
```

---

### Task 1: Baseline on Next 15

Record what passes *before* the upgrade, so any later failure can be attributed correctly.

**Files:** none changed.

- [ ] **Step 1: Install dependencies**

Run: `cd frontend && $CLEAN npm ci`
Expected: completes; `node_modules/next/package.json` reports `15.1.11`.

- [ ] **Step 2: Type-check**

Run: `cd frontend && $CLEAN npm run type-check`
Expected: exit 0. If not, write the errors down as pre-existing.

- [ ] **Step 3: Production build**

Run: `cd frontend && $CLEAN npx next build`
Expected: "Compiled successfully", routes `/`, `/forecast`, `/heatmap`, `/manifest.webmanifest` listed. Note the First Load JS size of `/` for comparison.

- [ ] **Step 4: Lint (expected to be non-functional)**

Run: `cd frontend && $CLEAN npx next lint` (and press nothing).
Expected: it prompts to configure ESLint or errors, because the repo has **no** ESLint config. Note this — lint has never actually run.

---

### Task 2: Bump Next, React and types

**Files:**
- Modify: `frontend/package.json`, `frontend/package-lock.json`

- [ ] **Step 1: Install the new versions**

```bash
cd frontend && $CLEAN npm install next@16.3.8 react@19.2 react-dom@19.2 eslint-config-next@16.3.8 \
  && $CLEAN npm install -D @types/react@19.2 @types/react-dom@19.2
```

Expected: `package.json` shows `"next": "16.3.8"`, React `^19.2.x`; no peer-dependency errors.

- [ ] **Step 2: Type-check**

Run: `cd frontend && $CLEAN npm run type-check`
Expected: exit 0. Next 16 regenerates `next-env.d.ts`; if it changed, keep the regenerated file.

- [ ] **Step 3: Production build (Turbopack is now the default)**

Run: `cd frontend && $CLEAN npx next build`
Expected: "▲ Next.js 16.3.8 (Turbopack)", "Compiled successfully", same routes as Task 1. First Load JS for `/` within ~10% of the baseline.

If the build fails on `experimental.optimizePackageImports`, delete that `experimental` block from `frontend/next.config.mjs` (it was a webpack/SWC workaround for Node 23 on ARM64) and rebuild.

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/next-env.d.ts frontend/next.config.mjs
git commit -m "chore(frontend): upgrade to Next 16.3 and React 19.2"
```

(Add `next.config.mjs` only if Step 3 changed it.)

---

### Task 3: Replace `next lint` with the ESLint CLI

Next 16 removed `next lint`. The repo has never had an ESLint config, so this adds the first one.

**Files:**
- Create: `frontend/eslint.config.mjs`
- Modify: `frontend/package.json` (`scripts.lint`)

- [ ] **Step 1: Add a flat config**

`frontend/eslint.config.mjs`:

```js
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const config = [
  ...nextVitals,
  ...nextTs,
  { ignores: [".next/**", "node_modules/**", "next-env.d.ts"] },
];

export default config;
```

- [ ] **Step 2: Point the script at ESLint**

In `frontend/package.json`: `"lint": "eslint ."`

- [ ] **Step 3: Run it**

Run: `cd frontend && $CLEAN npm run lint`
Expected: ESLint runs and reports results (it no longer prompts). Findings in existing code are **pre-existing**, because lint never ran before. Fix only findings that are errors *caused by the upgrade*; list the rest in the PR description as a follow-up rather than changing app code in this PR.

- [ ] **Step 4: Commit**

```bash
git add frontend/eslint.config.mjs frontend/package.json
git commit -m "chore(frontend): run ESLint directly (next lint was removed in Next 16)"
```

---

### Task 4: Verify the running app locally

**Files:**
- Modify (local only, gitignored): `.claude/launch.json` — add a configuration that serves *this worktree's* frontend.

- [ ] **Step 1: Add a worktree preview config**

Add to `configurations` in `.claude/launch.json`:

```json
{
  "name": "frontend-next16",
  "runtimeExecutable": "/bin/zsh",
  "runtimeArgs": ["-c", "cd /Users/yotamtsabari/sunset-predictor/.claude/worktrees/programming-design-skills-2cbe5c/frontend && env -i HOME=$HOME PATH=$PATH LANG=en_US.UTF-8 npx next start -p 3016"],
  "port": 3016
}
```

(`next start` serves the Task 2 production build; the API defaults to `http://localhost:8000`.)

- [ ] **Step 2: Start the backend and the frontend**

Use the preview tool: `preview_start {name: "backend"}`, then `preview_start {name: "frontend-next16"}`.

- [ ] **Step 3: Check every page at 375 px, light and dark**

With a location saved (`localStorage["afterglow:location"] = {"latitude":32.0853,"longitude":34.7818,"name":"Tel Aviv"}`):
- `/` — verdict card, rating buttons, When to look, evidence drawer, date picker opens, location sheet opens and closes.
- `/forecast` — chart renders, day cards expand.
- `/heatmap` — grid renders, year buttons switch.
- Theme toggle switches light/dark on each page.
- Clear storage once → first-run location screen appears.

Expected: identical to production; `read_console_messages` shows no errors; `preview_logs` for the frontend shows no runtime errors.

- [ ] **Step 4: Stop the frontend preview**

`preview_stop` the `frontend-next16` server.

---

### Task 5: Pull request and Vercel preview

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin chore/next-16
gh pr create --base main --title "chore(frontend): upgrade to Next 16" --body "<summary, browser-floor note, lint follow-ups, verification done>"
```

- [ ] **Step 2: Check the Vercel preview**

Vercel builds a preview per push. Confirm the build succeeds and open the preview URL: `/`, `/forecast`, `/heatmap` load and fetch data from the Render backend.

- [ ] **Step 3: Hand back**

Report the PR link, preview result, and any lint follow-ups. Merging is the user's call.
