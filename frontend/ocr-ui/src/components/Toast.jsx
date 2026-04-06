export function Toast({ tone = "info", title, description, onClose }) {
  const tones = {
    info: "border-primary/30 bg-primary/10 text-on-surface",
    success: "border-emerald-500/30 bg-emerald-500/10 text-on-surface",
    error: "border-error/30 bg-error/10 text-on-surface",
    warning: "border-amber-500/30 bg-amber-500/10 text-on-surface",
  };

  return (
    <div
      className={`pointer-events-auto w-full max-w-lg border p-4 shadow-lg backdrop-blur ${tones[tone]}`}
      role="status"
      aria-live="polite"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-bold font-label">{title}</div>
          {description ? (
            <div className="mt-1 text-sm opacity-80 font-body">{description}</div>
          ) : null}
        </div>
        <button
          onClick={onClose}
          className="px-2 py-1 text-sm opacity-80 hover:opacity-100 transition text-on-surface-variant"
          aria-label="Close notification"
          type="button"
        >
          ✕
        </button>
      </div>
    </div>
  );
}
