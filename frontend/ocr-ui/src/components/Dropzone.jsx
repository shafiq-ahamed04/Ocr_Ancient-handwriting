import { useCallback, useId, useRef } from "react";

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

export function Dropzone({ file, previewUrl, disabled, onPickFile, onClear, className }) {
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

  if (file && previewUrl) {
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
        <div className="w-full bg-surface-container-lowest flex flex-col items-center p-8 border border-outline-variant transition-all">
          <img
            src={previewUrl}
            alt="Selected document preview"
            className="max-h-48 w-auto object-contain mb-4 border border-outline-variant/30"
          />
          <p className="font-mono text-[11px] text-on-surface-variant mb-1">{file.name}</p>
          <p className="font-mono text-[10px] text-on-surface-variant opacity-60 mb-4">{formatBytes(file.size)}</p>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={openPicker}
              disabled={disabled}
              className="px-6 py-2 border border-outline-variant text-primary font-bold uppercase tracking-widest text-xs hover:bg-primary/10 transition-colors"
            >
              Replace
            </button>
            <button
              type="button"
              onClick={onClear}
              disabled={disabled}
              className="px-6 py-2 border border-outline-variant text-on-surface-variant font-bold uppercase tracking-widest text-xs hover:bg-surface-container-high transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
      </div>
    );
  }

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
        className={`w-full h-80 custom-dashed bg-surface-container-lowest flex flex-col items-center justify-center space-y-4 p-8 transition-all duration-300 hover:bg-surface-container-low cursor-pointer ${
          disabled ? "opacity-60 pointer-events-none" : ""
        }`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onClick={openPicker}
      >
        <span className="material-symbols-outlined text-primary text-6xl mb-2">cloud_upload</span>
        <div className="text-center">
          <h2 className="text-2xl font-bold text-on-surface mb-1 font-headline">Drop your Tamil document here</h2>
          <p className="text-on-surface-variant font-mono text-xs uppercase opacity-60">JPG, PNG, TIFF supported</p>
        </div>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            openPicker();
          }}
          disabled={disabled}
          className="mt-6 px-10 py-3 border border-outline-variant text-primary font-bold uppercase tracking-widest text-xs hover:bg-primary/10 transition-colors"
        >
          Browse File
        </button>
      </div>
    </div>
  );
}
