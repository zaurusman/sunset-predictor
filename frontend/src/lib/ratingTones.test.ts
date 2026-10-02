import { describe, expect, it } from "vitest";
import { ratingTone } from "./ratingTones";

describe("ratingTone", () => {
  it("gives every rating a readable ink and at least two ray colours", () => {
    for (const v of [1, 2, 3, 4, 5] as const) {
      for (const dark of [false, true]) {
        const t = ratingTone(v, dark);
        expect(t.ink).toMatch(/^#[0-9a-f]{6}$/i);
        expect(t.rays.length).toBeGreaterThanOrEqual(2);
      }
    }
  });

  it("matches the approved light tones", () => {
    expect(ratingTone(3, false).fill).toBe("linear-gradient(135deg,#fef3c7,#fed7aa)");
    expect(ratingTone(5, false).ink).toBe("#5b21b6");
  });
});
