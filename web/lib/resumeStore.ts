import { create } from "zustand";
import type { StructuredResume } from "./types";

export interface StyleSettings {
  fontSize: number; margin: number; lineSpacing: number; accent: string;
  sectionOrder: string[];
}
export interface OptimizeView {
  score: number; previousScore: number | null;
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
  applyTailored: (tailored: StructuredResume, score: number, previousScore: number | null) => void;
  setGaps: (gaps: OptimizeView["gaps"], fabricationsBlocked: number) => void;
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

export const useResumeStore = create<State>((set, get) => ({
  resume: null, provenance: null, style: DEFAULT_STYLE, optimize: null,
  setResume: (r) => set({ resume: r, optimize: null }),
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
  applyTailored: (tailored, score, previousScore) => set((s) => ({
    optimize: {
      score, previousScore,
      gaps: s.optimize?.gaps ?? [], fabricationsBlocked: s.optimize?.fabricationsBlocked ?? 0,
      originalResume: s.resume!,
    },
    resume: tailored,
  })),
  setGaps: (gaps, fabricationsBlocked) => set((s) => ({
    optimize: s.optimize ? { ...s.optimize, gaps, fabricationsBlocked } : {
      score: 0, previousScore: null, gaps, fabricationsBlocked, originalResume: s.resume!,
    },
  })),
}));
