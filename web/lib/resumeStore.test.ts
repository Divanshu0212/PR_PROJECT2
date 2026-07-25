import { beforeEach, describe, expect, it } from "vitest";
import { useResumeStore } from "./resumeStore";

const base = () => ({
  name: "Jane", headline: null, summary: null, emails: [], phones: [], urls: [],
  work: [{ company: "Acme", title: "Eng", bullets: ["Built X"] }],
  education: [],
  projects: [{ name: "CredVault", url: null, tech: ["Python"], bullets: ["Built auth"] }],
  skills: ["python"], certifications: [],
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

  it("adds, edits and removes a bullet on a project entry", () => {
    useResumeStore.getState().addProjectBullet(0);
    expect(useResumeStore.getState().resume!.projects[0].bullets).toHaveLength(2);
    useResumeStore.getState().editProjectBullet(0, 1, "Added caching");
    expect(useResumeStore.getState().resume!.projects[0].bullets[1]).toBe("Added caching");
    useResumeStore.getState().removeProjectBullet(0, 0);
    expect(useResumeStore.getState().resume!.projects[0].bullets).toEqual(["Added caching"]);
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

  it("applyOptimize swaps in tailored resume, sets scores and components", () => {
    useResumeStore.getState().applyOptimize({
      tailored: { name: "Jane", work: [{ company: "Acme", title: "Eng", bullets: ["Built X in Python"] }], skills: ["python"], emails: [], phones: [], urls: [], education: [], projects: [], certifications: [] } as any,
      displayScore: 88,
      baselineDisplayScore: 60,
      components: [{ label: "Keyword match", before: 0.5, after: 0.9 }],
      gaps: [],
      fabricationsBlocked: 2,
      previousScore: 55,
    });
    const s = useResumeStore.getState();
    expect(s.resume!.work[0].bullets[0]).toBe("Built X in Python");
    expect(s.optimize?.score).toBe(88);
    expect(s.optimize?.baselineScore).toBe(60);
    expect(s.optimize?.previousScore).toBe(55);
    expect(s.optimize?.components[0].after).toBe(0.9);
    expect(s.optimize?.fabricationsBlocked).toBe(2);
  });
});
