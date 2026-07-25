import type { Config } from "tailwindcss";
export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "var(--ink)",
        "ink-muted": "var(--ink-muted)",
        paper: "var(--paper)",
        desk: "var(--desk)",
        hairline: "var(--hairline)",
        studio: "var(--studio)",
      },
      fontFamily: {
        tool: ["var(--font-tool)"],
        label: ["var(--font-label)"],
        sheet: ["var(--font-sheet)"],
      },
    },
  },
  plugins: [],
} satisfies Config;
