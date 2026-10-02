import { describe, expect, it } from "vitest";
import { skyGradient } from "./sky";

describe("skyGradient", () => {
  it("returns a distinct gradient per mood and theme", () => {
    const moods = ["Poor", "Decent", "Good", "Great", "Epic", "Neutral", "Day"] as const;
    const light = moods.map((m) => skyGradient(m, false));
    const dark = moods.map((m) => skyGradient(m, true));
    expect(new Set(light).size).toBe(moods.length);
    expect(new Set(dark).size).toBe(moods.length);
    light.concat(dark).forEach((g) => expect(g.startsWith("linear-gradient(180deg")).toBe(true));
  });

  it("fades into the page colour", () => {
    expect(skyGradient("Epic", false)).toContain("#f8fafc");
    expect(skyGradient("Epic", true)).toContain("#020617");
  });
});
