"use client";
import type { StructuredResume } from "@/lib/types";
import type { OptimizeView, StyleSettings } from "@/lib/resumeStore";

function bulletBefore(original: StructuredResume | undefined, wi: number, bi: number): string | null {
  const b = original?.work?.[wi]?.bullets?.[bi];
  return b !== undefined ? b : null;
}

function projectBulletBefore(original: StructuredResume | undefined, pi: number, bi: number): string | null {
  const b = original?.projects?.[pi]?.bullets?.[bi];
  return b !== undefined ? b : null;
}

export function ResumePreview({ resume, style, optimize }: {
  resume: StructuredResume; style: StyleSettings; optimize: OptimizeView | null;
}) {
  const sheet: React.CSSProperties = {
    fontSize: style.fontSize, padding: style.margin, lineHeight: style.lineSpacing,
    ["--accent" as any]: style.accent,
  };
  return (
    <article style={sheet} className="print-sheet mx-auto max-w-3xl bg-white text-neutral-900 shadow">
      <header>
        <h1 className="text-2xl font-bold">{resume.name}</h1>
        {resume.headline && <p className="text-[color:var(--accent)]">{resume.headline}</p>}
      </header>
      {style.sectionOrder.map((section) => {
        if (section === "summary" && resume.summary)
          return <section key="summary"><h2 className="mt-4 font-semibold uppercase text-sm text-[color:var(--accent)]">Summary</h2><p>{resume.summary}</p></section>;
        if (section === "skills" && resume.skills.length)
          return <section key="skills"><h2 className="mt-4 font-semibold uppercase text-sm text-[color:var(--accent)]">Skills</h2>
            <ul className="flex flex-wrap gap-2">{resume.skills.map((s) => <li key={s} className="rounded bg-neutral-100 px-2">{s}</li>)}</ul></section>;
        if (section === "work" && resume.work.length)
          return <section key="work"><h2 className="mt-4 font-semibold uppercase text-sm text-[color:var(--accent)]">Experience</h2>
            {resume.work.map((w, wi) => (
              <div key={wi} className="mt-2">
                <div className="flex justify-between"><strong>{w.title}</strong><span>{w.start_date}{w.end_date ? `–${w.end_date}` : ""}</span></div>
                <div className="italic">{w.company}</div>
                <ul className="list-disc pl-5">
                  {w.bullets.map((b, bi) => {
                    const before = optimize ? bulletBefore(optimize.originalResume, wi, bi) : null;
                    const changed = before !== null && before !== b;
                    return (
                      <li key={bi}>
                        {changed && <span className="print-hide mr-1 text-neutral-400 line-through">{before}</span>}
                        <span>{b}</span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}</section>;
        if (section === "education" && resume.education.length)
          return <section key="education"><h2 className="mt-4 font-semibold uppercase text-sm text-[color:var(--accent)]">Education</h2>
            {resume.education.map((e, ei) => <div key={ei}>{e.institution}{e.degree ? ` — ${e.degree}` : ""}{e.field ? `, ${e.field}` : ""}{e.end_year ? ` (${e.end_year})` : ""}</div>)}</section>;
        if (section === "projects" && resume.projects?.length)
          return <section key="projects"><h2 className="mt-4 font-semibold uppercase text-sm text-[color:var(--accent)]">Projects</h2>
            {resume.projects.map((p, pi) => (
              <div key={pi} className="mt-2">
                <div className="flex justify-between gap-2">
                  <strong>{p.name}</strong>
                  {p.url && <a href={p.url} className="text-[color:var(--accent)] underline" target="_blank" rel="noreferrer">link</a>}
                </div>
                {p.tech.length > 0 && <div className="text-sm italic text-neutral-600">{p.tech.join(", ")}</div>}
                <ul className="list-disc pl-5">
                  {p.bullets.map((b, bi) => {
                    const before = optimize ? projectBulletBefore(optimize.originalResume, pi, bi) : null;
                    const changed = before !== null && before !== b;
                    return (
                      <li key={bi}>
                        {changed && <span className="print-hide mr-1 text-neutral-400 line-through">{before}</span>}
                        <span>{b}</span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}</section>;
        return null;
      })}
    </article>
  );
}
