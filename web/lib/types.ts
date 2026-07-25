// Mirrors the rho backend Pydantic models (partial: only fields the UI uses).
export interface WorkExperience {
  company: string; title: string;
  start_date?: string | null; end_date?: string | null;
  bullets: string[]; bullet_prov?: string[][];
  company_prov?: string[]; title_prov?: string[];
}
export interface Education {
  institution: string; degree?: string | null; field?: string | null; end_year?: string | null;
}
export interface StructuredResume {
  name: string; headline?: string | null; summary?: string | null;
  emails: string[]; phones: string[]; urls: string[];
  work: WorkExperience[]; education: Education[];
  skills: string[]; certifications: string[];
  skills_prov?: string[][];
}
export interface ParseResponse { structured_resume: StructuredResume; provenance_map: unknown; }
export interface Gap { requirement: { text: string; priority: string }; status: string; }
export interface MatchResult { predicted_score: number; gaps: Gap[]; }
export interface FabricationReport { total_edits: number; verified_edits: number; fabrication_rate?: number; rejected_edits: { added_text: string; reason: string }[]; }
export interface TailoredResume { resume: StructuredResume; fabrication_report: FabricationReport; }
export interface OptimizeResult { match_result: MatchResult; tailored_resume: TailoredResume; final_score: number; previous_score?: number | null; }
export type JobState = "queued" | "running" | "done" | "error";
export interface JobStatus { id: string; state: JobState; stage?: string | null; result?: OptimizeResult | null; error?: string | null; }
