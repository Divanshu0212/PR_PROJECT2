import { beforeEach, describe, expect, it } from "vitest";
import { useResumeStore } from "./resumeStore";

const base = () => ({
  name: "Jane", headline: null, summary: null, emails: [], phones: [], urls: [],
  work: [{ company: "Acme", title: "Eng", bullets: ["Built X"] }],
  education: [], skills: ["python"], certifications: [],
});

beforeEach(() => useResumeStore.getState().setResume(base() as any));

describe("resume store", () => {
  it("edits a top-level field", () => {
    useResumeStore.getState().setField("summary", "Senior engineer");
    expect(useResumeStore.getState().resume!.summary).toBe("Senior engineer");
  });

  it("adds and removes a bullet on a work entry", () => {
    useResumeStore.getState().addBullet(0);
    expect(useResumeStore.getState().resume!.work[0].bullets).toHaveLength(2);
    useResumeStore.getState().editBullet(0, 1, "Led Y");
    expect(useResumeStore.getState().resume!.work[0].bullets[1]).toBe("Led Y");
    useResumeStore.getState().removeBullet(0, 0);
    expect(useResumeStore.getState().resume!.work[0].bullets).toEqual(["Led Y"]);
  });

  it("adds and removes skills without duplicates", () => {
    useResumeStore.getState().addSkill("python"); // dup ignored
    useResumeStore.getState().addSkill("aws");
    expect(useResumeStore.getState().resume!.skills).toEqual(["python", "aws"]);
    useResumeStore.getState().removeSkill("python");
    expect(useResumeStore.getState().resume!.skills).toEqual(["aws"]);
  });

  it("updates style settings", () => {
    useResumeStore.getState().setStyle({ fontSize: 12 });
    expect(useResumeStore.getState().style.fontSize).toBe(12);
  });

  it("applyTailored swaps in tailored resume and records previous score", () => {
    useResumeStore.getState().applyTailored(
      { name: "Jane", work: [{ company: "Acme", title: "Eng", bullets: ["Built X in Python"] }], skills: ["python"], emails: [], phones: [], urls: [], education: [], certifications: [] } as any,
      88, 60,
    );
    const s = useResumeStore.getState();
    expect(s.resume!.work[0].bullets[0]).toBe("Built X in Python");
    expect(s.optimize?.score).toBe(88);
    expect(s.optimize?.previousScore).toBe(60);
  });
});
