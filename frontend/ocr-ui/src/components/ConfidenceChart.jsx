export function ConfidenceChart({ classification, className }) {
  if (!classification?.candidates || classification.candidates.length === 0) {
    return null;
  }

  const { predicted_label, candidates } = classification;

  return (
    <div className={`border border-outline-variant/30 bg-surface-container p-4 ${className || ""}`}>
      <h4 className="mb-4 text-[11px] font-label font-bold text-primary tracking-[0.2em] uppercase">
        Single Character Classification
      </h4>

      <div className="flex items-start gap-6">
        {/* Top prediction highlight */}
        <div className="flex flex-col items-center justify-center border border-primary/30 bg-primary/5 w-24 p-3">
          <span className="text-[10px] text-on-surface-variant mb-1 font-label uppercase tracking-wider">Prediction</span>
          <span className="text-4xl font-bold text-primary leading-none font-tamil">{predicted_label}</span>
          <span className="mt-2 text-[10px] font-bold text-primary/80 uppercase tracking-wider font-mono">
            {candidates[0]?.probability.toFixed(1)}%
          </span>
        </div>

        {/* Bar charts */}
        <div className="flex-1 space-y-3">
          {candidates.map((c, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="w-8 text-right text-lg font-bold text-on-surface font-tamil">
                {c.label}
              </div>
              <div className="relative h-2 flex-1 overflow-hidden bg-surface-container-high">
                <div
                  className={`absolute inset-y-0 left-0 transition-all duration-700 ${
                    i === 0 ? "bg-primary shadow-[0_0_10px_rgba(44,57,201,0.5)]" : "bg-on-surface-variant/30"
                  }`}
                  style={{ width: `${c.probability}%`, opacity: 0.3 + (c.probability / 100) * 0.7 }}
                />
              </div>
              <div className="w-12 text-right text-xs font-medium tabular-nums text-on-surface-variant font-mono">
                {c.probability.toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
