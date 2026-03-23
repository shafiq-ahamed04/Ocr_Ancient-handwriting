import clsx from "clsx";

export function ProgressSteps({ steps, currentStep, loading, className }) {
  return (
    <div className={clsx("flex flex-wrap items-center justify-center gap-3", className)}>
      {steps.map((step, idx) => {
        const isPast = idx < currentStep;
        const isCurrent = idx === currentStep;
        
        return (
          <div key={step} className="flex items-center gap-3">
            <div
              className={clsx(
                "flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-300",
                isPast
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                  : isCurrent
                  ? "bg-indigo-500/10 text-indigo-300 border border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.2)]"
                  : "bg-zinc-900/50 text-zinc-500 border border-zinc-800"
              )}
            >
              {isPast ? (
                <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
              ) : isCurrent && loading ? (
                <div className="size-3.5 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin" />
              ) : (
                <span className="flex size-3.5 items-center justify-center text-[10px] opacity-70">
                  {idx + 1}
                </span>
              )}
              {step}
            </div>
            
            {idx < steps.length - 1 && (
              <div
                className={clsx(
                  "h-px w-6 transition-all duration-300",
                  isPast ? "bg-emerald-500/30" : "bg-zinc-800"
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
