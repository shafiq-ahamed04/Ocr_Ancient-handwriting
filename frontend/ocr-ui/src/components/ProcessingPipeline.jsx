import { ThemeToggle } from "./ThemeToggle";

const STAGES = [
  {
    icon: "tune",
    title: "Image Preprocessing",
    desc: "Noise reduction and contrast normalization complete.",
  },
  {
    icon: "layers",
    title: "Feature Extraction",
    desc: "Identifying distinct character glyphs across text regions.",
  },
  {
    icon: "memory",
    title: "CRNN Decoding",
    desc: "Mapping visual features to cognitive linguistic weights...",
  },
  {
    icon: "verified",
    title: "Output Ready",
    desc: "Generating archival report and translation.",
  },
];

export function ProcessingPipeline({ currentStep, fileName, previewUrl, selectedMode }) {
  // Map currentStep (0-4) to pipeline stages (0-3)
  // currentStep 0 = waiting, 1 = stage 0 active, 2 = stage 1 active, 3 = stage 2 active, 4 = all done
  const getStageState = (stageIdx) => {
    if (currentStep >= stageIdx + 2) return "complete";
    if (currentStep === stageIdx + 1) return "active";
    if (currentStep === 4) return "complete";
    return "pending";
  };

  // Calculate active line height percentage
  const progressPercent = Math.min(((currentStep) / 4) * 100, 100);

  return (
    <div className="flex-1 flex flex-col min-h-screen relative overflow-hidden">
      {/* Dot grid overlay */}
      <div className="fixed inset-0 pointer-events-none opacity-10" style={{
        backgroundImage: 'radial-gradient(circle, var(--outline-variant) 1px, transparent 1px)',
        backgroundSize: '24px 24px'
      }} />

      {/* Header Navigation */}
      <header className="relative z-10 flex items-center justify-between px-8 py-10 max-w-7xl mx-auto w-full">
        <div className="flex-1" />
        <div className="flex-1 text-center">
          <h1 className="font-headline font-bold text-xl tracking-tight text-on-surface">Analyzing Document</h1>
        </div>
        <div className="flex-1 flex justify-end">
          <ThemeToggle />
        </div>
      </header>

      <main className="relative z-10 max-w-2xl mx-auto px-6 pb-20 flex flex-col items-center">
        {/* Document Thumbnail */}
        <section className="flex flex-col items-center mb-16">
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-outline-variant opacity-20" />
            {previewUrl ? (
              <img
                alt="Document being processed"
                className="w-[180px] h-auto object-cover border border-outline-variant relative z-10 grayscale hover:grayscale-0 transition-all duration-500"
                src={previewUrl}
              />
            ) : (
              <div className="w-[180px] h-[120px] bg-surface-container-high border border-outline-variant relative z-10 flex items-center justify-center">
                <span className="material-symbols-outlined text-on-surface-variant text-4xl opacity-30">image</span>
              </div>
            )}
          </div>
          <p className="mt-4 font-mono text-[10px] uppercase tracking-widest text-on-surface-variant">
            {fileName}
          </p>
        </section>

        {/* Pipeline Stepper */}
        <section className="w-full relative pl-8 sm:pl-12">
          {/* Vertical line background */}
          <div
            className="absolute left-10 sm:left-14 top-0 w-[1px] bg-surface-container-high"
            style={{ height: 'calc(100% - 48px)' }}
          />
          {/* Active progress line */}
          <div
            className="absolute left-10 sm:left-14 top-0 w-[1px] bg-primary transition-all duration-1000"
            style={{ height: `${progressPercent}%` }}
          />

          <div className="space-y-16">
            {STAGES.map((stage, idx) => {
              const state = getStageState(idx);
              return (
                <div key={idx} className="relative flex items-start gap-8">
                  {/* Step indicator */}
                  <div className={`relative z-20 flex items-center justify-center w-5 h-5 mt-1 shrink-0 ${
                    state === "complete"
                      ? "bg-primary"
                      : state === "active"
                      ? "bg-surface border border-primary pulse-ring"
                      : "bg-surface-container-high"
                  }`}>
                    {state === "complete" ? (
                      <span className="material-symbols-outlined text-on-primary text-[14px] font-bold">check</span>
                    ) : state === "active" ? (
                      <div className="w-2 h-2 bg-primary" />
                    ) : (
                      <div className="w-1.5 h-1.5 bg-surface-variant" />
                    )}
                  </div>

                  {/* Step content */}
                  <div className={`flex flex-col ${state === "pending" ? "opacity-40" : ""}`}>
                    <div className="flex items-center gap-3">
                      <span className={`material-symbols-outlined text-lg ${
                        state === "active" || state === "complete" ? "text-primary" : "text-on-surface-variant"
                      }`}>{stage.icon}</span>
                      <h3 className={`font-headline font-bold tracking-wide ${
                        state === "active" ? "text-primary" : state === "complete" ? "text-on-surface" : "text-on-surface-variant"
                      }`}>{stage.title}</h3>
                    </div>
                    <p className={`text-on-surface-variant text-sm mt-1 max-w-xs ${state === "active" ? "italic" : ""}`}>
                      {stage.desc}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Bottom Status */}
        <footer className="mt-24 text-center">
          <div className="flex items-center justify-center gap-3 mb-2">
            <div className="w-1 h-1 bg-primary animate-pulse" />
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-on-surface-variant italic">
              Processing… do not close this tab
            </p>
            <div className="w-1 h-1 bg-primary animate-pulse" />
          </div>
        </footer>
      </main>
    </div>
  );
}
