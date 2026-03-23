import clsx from "clsx";

export function ConfidenceChart({ classification, className }) {
  if (!classification?.candidates || classification.candidates.length === 0) {
    return null;
  }

  const { predicted_label, candidates } = classification;

  return (
    <div className={clsx("rounded-xl border border-zinc-800/50 bg-zinc-900/20 p-4", className)}>
      <h4 className="mb-4 text-sm font-medium text-zinc-300">Single Character Classification</h4>
      
      <div className="flex items-start gap-6">
        {/* Top prediction highlight */}
        <div className="flex flex-col items-center justify-center rounded-lg border border-indigo-500/20 bg-indigo-500/5 w-24 p-3 shadow-[0_0_15px_rgba(99,102,241,0.1)]">
          <span className="text-xs text-zinc-500 mb-1">Prediction</span>
          <span className="text-4xl font-bold text-indigo-400 leading-none">{predicted_label}</span>
          <span className="mt-2 text-[10px] font-medium text-indigo-300/80 uppercase tracking-wider">
            {candidates[0]?.probability.toFixed(1)}% Conf
          </span>
        </div>

        {/* Bar charts for top 3-5 candidates */}
        <div className="flex-1 space-y-3">
          {candidates.map((c, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="w-8 text-right text-lg font-bold text-zinc-200">
                {c.label}
              </div>
              
              <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-zinc-800/60">
                <div 
                  className={clsx(
                    "absolute inset-y-0 left-0 rounded-full transition-all duration-700",
                    i === 0 ? "bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]" : "bg-zinc-600"
                  )}
                  style={{ width: `${c.probability}%`, opacity: 0.3 + (c.probability / 100) * 0.7 }}
                />
              </div>
              
              <div className="w-12 text-right text-xs font-medium tabular-nums text-zinc-400">
                {c.probability.toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
