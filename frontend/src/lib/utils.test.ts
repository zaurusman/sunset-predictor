import { describe, expect, it } from "vitest";
import { SCORE_BANDS, scoreCategory } from "./utils";

describe("scoreCategory", () => {
  // Must match SCORE_THRESHOLDS in backend/app/services/scoring_engine.py,
  // which labels the category pill. A drift here paints an 82 "Great" purple.
  it("bands at the backend's thresholds", () => {
    expect(SCORE_BANDS).toEqual([
      [85, "Epic"],
      [72, "Great"],
      [55, "Good"],
      [38, "Decent"],
      [0, "Poor"],
    ]);
  });

  it("puts each edge in the upper band", () => {
    expect(scoreCategory(84.9)).toBe("Great");
    expect(scoreCategory(85)).toBe("Epic");
    expect(scoreCategory(82)).toBe("Great");
    expect(scoreCategory(72)).toBe("Great");
    expect(scoreCategory(71)).toBe("Good");
    expect(scoreCategory(55)).toBe("Good");
    expect(scoreCategory(38)).toBe("Decent");
    expect(scoreCategory(37)).toBe("Poor");
  });
});
