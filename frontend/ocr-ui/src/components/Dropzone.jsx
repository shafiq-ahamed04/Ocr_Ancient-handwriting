import { useCallback, useId, useRef } from "react";
import clsx from "clsx";

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export function Dropzone({
  file,
  previewUrl,
  disabled,
  onPickFile,
  onClear,
  className,
}) {
  const inputId = useId();
  const inputRef = useRef(null);

  const openPicker = useCallback(() => {
    if (disabled) return;
    inputRef.current?.click();
  }, [disabled]);

  const onInputChange = useCallback(
    (e) => {
      const next = e.target.files?.[0] ?? null;
      onPickFile(next);
      // allow picking the same file again
      e.target.value = "";
    },
    [onPickFile]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      if (disabled) return;
      const next = e.dataTransfer.files?.[0] ?? null;
      onPickFile(next);
    },
    [disabled, onPickFile]
  );

  const onDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  return (
    <div className={className}>
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={onInputChange}
        disabled={disabled}
      />

      <div
        className={clsx(
          "group relative overflow-hidden rounded-2xl border border-zinc-800/70 bg-zinc-950/30 p-4 transition hover:border-zinc-700",
          disabled && "opacity-60"
        )}
        onDrop={onDrop}
        onDragOver={onDragOver}
      >
        <div className="flex items-start gap-4">
          <div className="flex size-16 items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-950">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected upload preview"
                className="size-16 rounded-2xl object-cover"
              />
            ) : (
              <span className="text-2xl text-zinc-400" aria-hidden="true">
                ⤒
              </span>
            )}
          </div>

          <div className="min-w-0 flex-1">
            <div className="text-sm font-semibold text-zinc-100">
              {file ? "Image selected" : "Drop an image here"}
            </div>
            <div className="mt-1 text-sm text-zinc-400">
              {file ? (
                <span className="break-all">
                  {file.name} · {formatBytes(file.size)}
                </span>
              ) : (
                <span>
                  Or{" "}
                  <button
                    type="button"
                    className="text-indigo-300 underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-400"
                    onClick={openPicker}
                    disabled={disabled}
                  >
                    choose a file
                  </button>
                </span>
              )}
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={openPicker}
                disabled={disabled}
                className="rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-100 hover:bg-zinc-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-600"
              >
                {file ? "Replace" : "Browse"}
              </button>
              {file ? (
                <button
                  type="button"
                  onClick={onClear}
                  disabled={disabled}
                  className="rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-100 hover:bg-zinc-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-600"
                >
                  Clear
                </button>
              ) : null}
            </div>
          </div>
        </div>

        <div className="pointer-events-none absolute inset-0 opacity-0 transition group-hover:opacity-100">
          <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-indigo-900/10 to-transparent" />
        </div>
      </div>
    </div>
  );
}

