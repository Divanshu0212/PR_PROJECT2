"use client";
import { UploadDropzone } from "@/components/UploadDropzone";
import { FieldEditors } from "@/components/Editor/FieldEditors";
import { WorkEditor } from "@/components/Editor/WorkEditor";
import { ProjectsEditor } from "@/components/Editor/ProjectsEditor";
import { SkillsEditor } from "@/components/Editor/SkillsEditor";
import { StyleControls } from "@/components/Editor/StyleControls";
import { JdBox } from "@/components/Editor/JdBox";
import { ResumePreview } from "@/components/Preview/ResumePreview";
import { useResumeStore } from "@/lib/resumeStore";

export default function Page() {
  const resume = useResumeStore((s) => s.resume);
  const style = useResumeStore((s) => s.style);
  const optimize = useResumeStore((s) => s.optimize);
  return (
    <main className="grid min-h-screen grid-cols-1 lg:grid-cols-[minmax(0,26rem)_1fr]">
      <section className="space-y-6 overflow-y-auto border-hairline bg-desk/40 px-6 py-6 lg:border-r">
        <header>
          <p className="font-label text-[11px] uppercase tracking-[0.24em] text-studio">rho</p>
          <h1 className="text-xl font-semibold text-ink">Résumé editor</h1>
        </header>
        <UploadDropzone />
        {resume && (
          <div className="space-y-6">
            <FieldEditors />
            <SkillsEditor />
            <WorkEditor />
            <ProjectsEditor />
            <details className="group border-t border-hairline pt-3">
              <summary className="cursor-pointer list-none font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted transition-colors hover:text-studio">
                Styling <span className="ml-1 inline-block transition-transform group-open:rotate-90">›</span>
              </summary>
              <StyleControls />
            </details>
            <JdBox />
          </div>
        )}
      </section>
      <section className="flex justify-center overflow-y-auto bg-paper px-6 py-10">
        {resume ? (
          <div className="w-full max-w-3xl shadow-[0_1px_2px_rgba(28,27,25,0.06),0_16px_40px_-16px_rgba(28,27,25,0.25)]">
            <ResumePreview resume={resume} style={style} optimize={optimize} />
          </div>
        ) : (
          <div className="flex h-full max-w-sm flex-col items-center justify-center text-center">
            <p className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">Preview</p>
            <p className="mt-2 text-sm text-ink-muted">
              Upload a résumé on the left to see it rendered here as a formatted sheet.
            </p>
          </div>
        )}
      </section>
    </main>
  );
}
