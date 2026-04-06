import { ThemeToggle } from "./ThemeToggle";
import { ConfidenceChart } from "./ConfidenceChart";

export function OutputView({
  text,
  stats,
  previewUrl,
  annotatedUrl,
  boxes,
  classification,
  fileName,
  selectedMode,
  onCopy,
  onDownloadTxt,
  onDownloadPdf,
  onProcessAnother,
}) {
  // Generate line numbers for the terminal
  const lines = text ? text.split("\n") : [];
  const lineNumbers = lines.map((_, i) => String(i + 1).padStart(3, "0")).join("\n");

  return (
    <div className="flex-1 flex flex-col min-h-screen">
      {/* Top bar */}
      <header className="w-full h-16 flex items-center justify-between px-8 bg-surface-container-low shadow-2xl z-50">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-emerald-500 font-medium">
            <span className="material-symbols-outlined text-[20px]">check_circle</span>
            <span className="font-label tracking-tight text-sm">Analysis Complete</span>
          </div>
          <div className="flex items-center gap-3">
            {stats.engine && (
              <div className="bg-primary/10 border border-primary px-3 py-0.5 flex items-center gap-2">
                <span className="text-[10px] font-label text-primary uppercase tracking-widest">Engine</span>
                <span className="text-[11px] font-mono text-primary font-bold">{stats.engine}</span>
              </div>
            )}
            {stats.word_count > 0 && (
              <div className="bg-secondary-container/20 border border-secondary px-3 py-0.5 flex items-center gap-2">
                <span className="text-[10px] font-label text-secondary uppercase tracking-widest">Words</span>
                <span className="text-[11px] font-mono text-secondary font-bold">{stats.word_count}</span>
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <ThemeToggle />
          <button
            id="process-another-btn"
            onClick={onProcessAnother}
            className="flex items-center gap-2 px-6 py-2 border border-outline/30 hover:border-primary transition-colors text-sm font-label uppercase tracking-widest group text-on-surface"
          >
            Process Another
            <span className="material-symbols-outlined text-[18px] group-hover:translate-x-1 transition-transform">arrow_right_alt</span>
          </button>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 w-full max-w-[1600px] mx-auto p-8 grid grid-cols-1 md:grid-cols-[38%_62%] gap-12">
        {/* Left Column: Source Document */}
        <section className="flex flex-col gap-6">
          <div className="flex flex-col gap-4">
            <h3 className="text-[11px] font-label font-bold text-primary tracking-[0.2em] uppercase">Source Document</h3>
            <div className="bg-surface-container-lowest p-1 border border-outline-variant/30">
              {(annotatedUrl || previewUrl) ? (
                <img
                  alt="Source document"
                  className="w-full aspect-[4/3] object-cover grayscale opacity-80 contrast-125"
                  src={annotatedUrl || previewUrl}
                />
              ) : (
                <div className="w-full aspect-[4/3] bg-surface-container-high flex items-center justify-center">
                  <span className="material-symbols-outlined text-on-surface-variant text-6xl opacity-20">image</span>
                </div>
              )}
            </div>
          </div>

          {/* File metadata */}
          <div className="flex flex-col gap-3 p-4 bg-surface-container-low border-l-2 border-primary">
            <div className="grid grid-cols-2 gap-y-2 text-[12px] font-mono">
              <span className="text-on-surface-variant">FILE_ID:</span>
              <span className="text-tertiary-fixed">{fileName}</span>
              <span className="text-on-surface-variant">DOMAIN_MODE:</span>
              <span className="text-tertiary-fixed">{selectedMode === "palmleaf" ? "Palm Leaf" : "Standard"}</span>
              <span className="text-on-surface-variant">ENGINE:</span>
              <span className="text-tertiary-fixed">{stats.engine || "—"}</span>
            </div>
          </div>

          {/* Classification chart */}
          {classification && Object.keys(classification).length > 0 && !classification.error && (
            <ConfidenceChart classification={classification} />
          )}

          {/* Decorative quote */}
          <div className="mt-auto p-6 bg-surface-container-lowest border-t border-outline-variant/20 flex flex-col gap-4">
            <p className="text-xs text-on-surface-variant leading-relaxed font-body italic opacity-60">
              "Cognitive recognition algorithms have isolated the glyph sequences by mapping spatial density against known epigraphic variants."
            </p>
          </div>
        </section>

        {/* Right Column: Extracted Text */}
        <section className="flex flex-col gap-4 h-full">
          <div className="flex items-end justify-between">
            <h3 className="text-[11px] font-label font-bold text-primary tracking-[0.2em] uppercase">Extracted Text</h3>
            <div className="flex items-center gap-2">
              <button
                id="copy-text-btn"
                onClick={onCopy}
                disabled={!text}
                className="p-2 bg-surface-container-high hover:text-primary transition-colors flex items-center justify-center disabled:opacity-40 text-on-surface"
              >
                <span className="material-symbols-outlined text-[18px]">content_copy</span>
              </button>
              <button
                id="export-txt-btn"
                onClick={onDownloadTxt}
                disabled={!text}
                className="px-4 py-1.5 bg-surface-container-high hover:text-primary transition-colors text-[10px] font-label font-bold uppercase tracking-widest border border-outline-variant/30 disabled:opacity-40 text-on-surface"
              >
                Export TXT
              </button>
              <button
                id="export-pdf-btn"
                onClick={onDownloadPdf}
                disabled={!text}
                className="px-4 py-1.5 bg-error text-on-error hover:brightness-110 transition-colors text-[10px] font-label font-bold uppercase tracking-widest flex items-center gap-2 disabled:opacity-40"
              >
                <span className="material-symbols-outlined text-[16px]">picture_as_pdf</span>
                Export PDF
              </button>
            </div>
          </div>

          {/* Terminal Box */}
          <div className="flex-1 flex flex-col bg-surface-container-lowest border border-primary overflow-hidden">
            <div className="flex-1 terminal-scroll overflow-y-auto font-mono text-sm leading-relaxed p-6 flex">
              {text ? (
                <>
                  {/* Line Numbers */}
                  <div className="pr-6 text-outline-variant text-right select-none border-r border-outline-variant/10 shrink-0 whitespace-pre">
                    {lineNumbers}
                  </div>
                  {/* Content */}
                  <div className="pl-6 text-on-surface w-full whitespace-pre-wrap break-words font-tamil">
                    {text}
                  </div>
                </>
              ) : (
                <div className="flex-1 flex items-center justify-center text-on-surface-variant opacity-40">
                  <div className="text-center">
                    <span className="material-symbols-outlined text-5xl mb-4 block opacity-30">terminal</span>
                    <p className="text-sm font-label">No text extracted</p>
                  </div>
                </div>
              )}
            </div>

            {/* Terminal Footer */}
            <div className="px-6 py-3 bg-surface-container-lowest border-t border-outline-variant/30 flex items-center gap-6">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-label text-on-surface-variant uppercase tracking-tighter">Word Count:</span>
                <span className="text-[11px] font-mono text-on-surface">{stats.word_count} words</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-label text-on-surface-variant uppercase tracking-tighter">Character Count:</span>
                <span className="text-[11px] font-mono text-on-surface">{stats.char_count} chars</span>
              </div>
              <div className="ml-auto flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 animate-pulse" />
                <span className="text-[10px] font-mono text-on-surface-variant uppercase">Stream Stable</span>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Watermark */}
      <div className="fixed bottom-0 right-0 p-8 pointer-events-none opacity-[0.04] overflow-hidden">
        <h1 className="font-headline font-black text-[12vw] leading-none text-on-surface translate-y-1/2 tracking-tighter">ATCRS</h1>
      </div>
    </div>
  );
}
