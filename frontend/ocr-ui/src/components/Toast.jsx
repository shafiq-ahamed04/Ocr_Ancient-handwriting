import clsx from "clsx";

export function Toast({ tone = "info", title, description, onClose }) {
  const tones = {
    info: "border-indigo-500/30 bg-indigo-950/40 text-indigo-100",
    success: "border-emerald-500/30 bg-emerald-950/40 text-emerald-100",
    error: "border-rose-500/30 bg-rose-950/40 text-rose-100",
    warning: "border-amber-500/30 bg-amber-950/40 text-amber-100",
  };

  return (
    <div
      className={clsx(
        "pointer-events-auto w-full max-w-lg rounded-2xl border p-4 shadow-lg backdrop-blur",
        tones[tone]
      )}
      role="status"
      aria-live="polite"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold">{title}</div>
          {description ? (
            <div className="mt-1 text-sm opacity-90">{description}</div>
          ) : null}
        </div>
        <button
          onClick={onClose}
          className="rounded-lg px-2 py-1 text-sm opacity-80 hover:opacity-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
          aria-label="Close notification"
          type="button"
        >
          ✕
        </button>
      </div>
    </div>
  );
}

