import { describe, expect, it } from "vitest";
import { markSunsetPlayed, shouldPlaySunset, staggerDelay, SUNSET_PLAYED_KEY } from "./motion";

function memory(): Storage {
  const m = new Map<string, string>();
  return {
    getItem: (k) => m.get(k) ?? null,
    setItem: (k, v) => void m.set(k, String(v)),
    removeItem: (k) => void m.delete(k),
    clear: () => m.clear(),
    key: (i) => [...m.keys()][i] ?? null,
    get length() {
      return m.size;
    },
  };
}

describe("staggerDelay", () => {
  it("steps per index and never exceeds the cap", () => {
    expect(staggerDelay(0, 70, 280)).toBe(0);
    expect(staggerDelay(2, 70, 280)).toBe(140);
    expect(staggerDelay(9, 70, 280)).toBe(280);
  });
});

describe("sunset played", () => {
  it("plays once per date", () => {
    const s = memory();
    expect(shouldPlaySunset("2026-10-03", s)).toBe(true);
    markSunsetPlayed("2026-10-03", s);
    expect(s.getItem(SUNSET_PLAYED_KEY)).toBe("2026-10-03");
    expect(shouldPlaySunset("2026-10-03", s)).toBe(false);
    expect(shouldPlaySunset("2026-10-04", s)).toBe(true);
  });

  it("plays when storage throws (private mode) but never crashes", () => {
    const broken = {
      getItem: () => {
        throw new Error("denied");
      },
    };
    expect(shouldPlaySunset("2026-10-03", broken)).toBe(true);
    expect(() =>
      markSunsetPlayed("2026-10-03", {
        setItem: () => {
          throw new Error("denied");
        },
      }),
    ).not.toThrow();
  });
});
