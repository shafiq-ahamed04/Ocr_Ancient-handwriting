import clsx from "clsx";

export function Spinner({ className }) {
  return (
    <span
      className={clsx(
        "inline-block size-4 animate-spin rounded-full border-2 border-zinc-500 border-t-transparent",
        className
      )}
      aria-label="Loading"
      role="status"
    />
  );
}

