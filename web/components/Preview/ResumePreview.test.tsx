import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResumePreview } from "./ResumePreview";

const style = { fontSize: 14, margin: 48, lineSpacing: 1.4, accent: "#000", sectionOrder: ["summary", "skills", "work", "education"] };
const resume = {
  name: "Jane Doe", headline: "Engineer", summary: "Builds things",
  emails: [], phones: [], urls: [], education: [], certifications: [],
  skills: ["python", "aws"],
  work: [{ company: "Acme", title: "Engineer", bullets: ["Built X in Python"] }],
};

describe("ResumePreview", () => {
  it("renders name, skills and bullets", () => {
    render(<ResumePreview resume={resume as any} style={style as any} optimize={null} />);
    expect(screen.getByText("Jane Doe")).toBeDefined();
    expect(screen.getByText("python")).toBeDefined();
    expect(screen.getByText(/Built X in Python/)).toBeDefined();
  });

  it("shows before/after when a bullet was tailored", () => {
    const original = { ...resume, work: [{ company: "Acme", title: "Engineer", bullets: ["Built X"] }] };
    render(<ResumePreview resume={resume as any} style={style as any}
      optimize={{ score: 80, previousScore: 60, gaps: [], fabricationsBlocked: 0, originalResume: original } as any} />);
    // original struck-through text present alongside the tailored version
    expect(screen.getByText("Built X")).toBeDefined();
    expect(screen.getByText(/Built X in Python/)).toBeDefined();
  });
});
