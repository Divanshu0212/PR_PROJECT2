import { create } from "zustand";
import type { StructuredResume } from "./types";

export interface StyleSettings {
  fontSize: number; margin: number; lineSpacing: number; accent: string;
  sectionOrder: string[];
}
export interface OptimizeView {
  score: number; previousScore: number | null;
  baselineScore: number | null; // this run's ORIGINAL-résumé score (before tailoring)
  components: { label: string; before: number; after: number }[];
  gaps: { text: string; priority: string; status: string }[];
  fabricationsBlocked: number;
  originalResume: StructuredResume; // pre-optimize, for before/after
}

interface State {
  resume: StructuredResume | null;
  provenance: unknown;
  style: StyleSettings;
  optimize: OptimizeView | null;
  setResume: (r: StructuredResume) => void;
  setProvenance: (p: unknown) => void;
  setField: <K extends keyof StructuredResume>(k: K, v: StructuredResume[K]) => void;
  addBullet: (workIdx: number) => void;
  editBullet: (workIdx: number, bulletIdx: number, text: string) => void;
  removeBullet: (workIdx: number, bulletIdx: number) => void;
  addSkill: (s: string) => void;
  removeSkill: (s: string) => void;
  addProjectBullet: (projIdx: number) => void;
  editProjectBullet: (projIdx: number, bulletIdx: number, text: string) => void;
  removeProjectBullet: (projIdx: number, bulletIdx: number) => void;
  setStyle: (patch: Partial<StyleSettings>) => void;
  applyOptimize: (view: {
    tailored: StructuredResume;
    displayScore: number;
    baselineDisplayScore: number | null;
    components: OptimizeView["components"];
    gaps: OptimizeView["gaps"];
    fabricationsBlocked: number;
    previousScore: number | null;
  }) => void;
}

const DEFAULT_STYLE: StyleSettings = {
  fontSize: 14, margin: 48, lineSpacing: 1.4, accent: "#2563eb",
  sectionOrder: ["summary", "skills", "work", "projects", "education"],
};

function mutate(r: StructuredResume, fn: (draft: StructuredResume) => void): StructuredResume {
  const copy: StructuredResume = JSON.parse(JSON.stringify(r));
  fn(copy);
  return copy;
}

// Backfill array fields a résumé may be missing (e.g. one parsed before the
// projects field existed), so every component can read them without guarding.
function normalize(r: StructuredResume): StructuredResume {
  return {
    ...r,
    work: r.work ?? [],
    education: r.education ?? [],
    projects: r.projects ?? [],
    skills: r.skills ?? [],
    certifications: r.certifications ?? [],
    emails: r.emails ?? [],
    phones: r.phones ?? [],
    urls: r.urls ?? [],
  };
}

export const useResumeStore = create<State>((set, get) => ({
  resume: null, provenance: null, style: DEFAULT_STYLE, optimize: null,
  setResume: (r) => set({ resume: normalize(r), optimize: null }),
  setProvenance: (p) => set({ provenance: p }),
  setField: (k, v) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { (d as any)[k] = v; }) })),
  addBullet: (wi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.work[wi].bullets.push(""); }) })),
  editBullet: (wi, bi, text) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.work[wi].bullets[bi] = text; }) })),
  removeBullet: (wi, bi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.work[wi].bullets.splice(bi, 1); }) })),
  addSkill: (skill) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { if (!d.skills.map((x) => x.toLowerCase()).includes(skill.toLowerCase())) d.skills.push(skill); }) })),
  removeSkill: (skill) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.skills = d.skills.filter((x) => x.toLowerCase() !== skill.toLowerCase()); }) })),
  addProjectBullet: (pi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.projects[pi].bullets.push(""); }) })),
  editProjectBullet: (pi, bi, text) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.projects[pi].bullets[bi] = text; }) })),
  removeProjectBullet: (pi, bi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.projects[pi].bullets.splice(bi, 1); }) })),
  setStyle: (patch) => set((s) => ({ style: { ...s.style, ...patch } })),
  applyOptimize: (v) => set((s) => ({
    optimize: {
      score: v.displayScore,
      previousScore: v.previousScore,
      baselineScore: v.baselineDisplayScore,
      components: v.components,
      gaps: v.gaps,
      fabricationsBlocked: v.fabricationsBlocked,
      originalResume: s.resume!, // the résumé that went in, for before/after
    },
    resume: normalize(v.tailored),
  })),
}));
