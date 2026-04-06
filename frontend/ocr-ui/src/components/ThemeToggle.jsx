import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(() => {
    if (typeof window === "undefined") return true;
    const stored = localStorage.getItem("atcrs-theme");
    if (stored) return stored === "dark";
    return true; // default dark
  });

  useEffect(() => {
    const root = document.documentElement;
    root.classList.add("theme-transition");

    if (isDark) {
      root.classList.remove("light");
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
      root.classList.add("light");
    }

    localStorage.setItem("atcrs-theme", isDark ? "dark" : "light");

    const timeout = setTimeout(() => {
      root.classList.remove("theme-transition");
    }, 400);
    return () => clearTimeout(timeout);
  }, [isDark]);

  return (
    <button
      id="theme-toggle"
      onClick={() => setIsDark((prev) => !prev)}
      className="relative flex items-center justify-center w-10 h-10 border border-outline-variant bg-surface-container hover:bg-surface-container-high transition-all group"
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <span
        className="material-symbols-outlined text-on-surface-variant group-hover:text-primary transition-colors text-[20px]"
      >
        {isDark ? "light_mode" : "dark_mode"}
      </span>
    </button>
  );
}
