import { afterEach, describe, expect, it, vi } from "vitest";
import { isToday, localToday, SCORE_BANDS, scoreCategory } from "./utils";

describe("scoreCategory", () => {
  // Must match SCORE_THRESHOLDS in backend/app/services/scoring_engine.py,
  // which labels the category pill. A drift here paints an 82 "Great" purple.
  it("bands at the backend's thresholds", () => {
    expect(SCORE_BANDS).toEqual([
      [82, "Epic"],
      [70, "Great"],
      [55, "Good"],
      [38, "Decent"],
      [0, "Poor"],
    ]);
  });

  it("puts each edge in the upper band", () => {
    expect(scoreCategory(81.9)).toBe("Great");
    expect(scoreCategory(82)).toBe("Epic");
    expect(scoreCategory(70)).toBe("Great");
    expect(scoreCategory(69.9)).toBe("Good");
    expect(scoreCategory(55)).toBe("Good");
    expect(scoreCategory(38)).toBe("Decent");
    expect(scoreCategory(37)).toBe("Poor");
  });
});

describe("localToday", () => {
  const tz = process.env.TZ;
  afterEach(() => {
    vi.useRealTimers();
    process.env.TZ = tz;
  });

  // 01:30 in Tel Aviv is still the previous day in UTC. "Today" must be the
  // user's date, which is also what the server calls tonight.
  it("uses the local date after midnight, not the UTC one", () => {
    process.env.TZ = "Asia/Jerusalem";
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-02T22:30:00Z"));
    expect(localToday()).toBe("2026-10-03");
    expect(isToday("2026-10-03")).toBe(true);
    expect(isToday("2026-10-02")).toBe(false);
  });
});
