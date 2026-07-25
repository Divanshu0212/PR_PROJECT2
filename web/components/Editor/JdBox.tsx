"use client";
import { useState } from "react";
import { startOptimize, pollOptimize, BackendUnreachable } from "@/lib/api";
import { useResumeStore } from "@/lib/resumeStore";

export function JdBox() {
  const resume = useResumeStore((s) => s.resume);
  const optimize = useResumeStore((s) => s.optimize);
  const { applyTailored, setGaps } = useResumeStore.getState();
  const [jd, setJd] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!resume || !jd.trim()) return;
    setBusy(true); setError(null); setStage("starting");
    const prevScore = optimize?.score ?? null;
    try {
      const started = await startOptimize(resume, jd);
      const done = await pollOptimize(started.id, {
        intervalMs: 1500, timeoutMs: 180000,
      });
      const r = done.result!;
      applyTailored(r.tailored_resume.resume, r.final_score, prevScore);
      setGaps(
        r.match_result.gaps.map((g) => ({ text: g.requirement.text, priority: g.requirement.priority, status: g.status })),
        r.tailored_resume.fabrication_report.rejected_edits.length,
      );
    } catch (e) {
      setError(e instanceof BackendUnreachable
        ? "Backend unreachable. Start it: uvicorn rho.api.app:app --reload"
        : (e as Error).message);
    } finally { setBusy(false); setStage(null); }
  }

  return (
    <div className="space-y-3 border-t border-hairline pt-4">
      <h2 className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">Target role</h2>
      <textarea
        className="h-32 w-full resize-y rounded-sm border border-hairline bg-white/60 p-2 text-sm text-ink outline-none transition-colors placeholder:text-ink-muted/50 focus:border-studio"
        value={jd} onChange={(e) => setJd(e.target.value)} placeholder="Paste the target job description…" />
      <button disabled={!resume || !jd.trim() || busy}
        className="w-full rounded-sm bg-ink py-2.5 font-label text-[11px] uppercase tracking-[0.14em] text-paper transition-colors hover:bg-studio disabled:cursor-not-allowed disabled:bg-ink-muted/30 disabled:text-ink-muted"
        onClick={run}>
        {busy ? `Optimising… ${stage ?? ""}` : "Optimise score →"}
      </button>
      {error && (
        <p className="border-l-2 border-studio pl-2 text-sm text-studio">{error}</p>
      )}
      {optimize && !busy && (
        <div className="space-y-2 rounded-sm border border-hairline bg-white/60 p-3">
          <div className="flex items-baseline gap-2">
            <span className="font-label text-[11px] uppercase tracking-[0.1em] text-ink-muted">Score</span>
            <span className="text-2xl font-semibold text-ink">{optimize.score.toFixed(0)}</span>
            <span className="text-sm text-ink-muted">/100</span>
            {optimize.previousScore !== null && (
              <span className="ml-1 rounded-sm bg-studio/10 px-1.5 py-0.5 font-label text-[11px] text-studio">
                ▲ from {optimize.previousScore.toFixed(0)}
              </span>
            )}
          </div>
          <div className="text-sm text-ink-muted">
            Unsourced edits blocked: <span className="text-ink">{optimize.fabricationsBlocked}</span>
            <span className="ml-1 text-xs">(fabrication gate)</span>
          </div>
          {optimize.gaps.length > 0 && (
            <div className="border-t border-hairline pt-2 text-sm text-ink-muted">
              Gaps: <span className="text-ink">{optimize.gaps.filter((g) => g.status !== "present").map((g) => g.text).join(", ")}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
