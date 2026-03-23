import clsx from "clsx";

const VARIANTS = {
  primary:
    "bg-indigo-600 text-white hover:bg-indigo-500 active:bg-indigo-700 focus-visible:outline-indigo-400",
  secondary:
    "bg-zinc-800 text-zinc-100 hover:bg-zinc-700 active:bg-zinc-800 focus-visible:outline-zinc-500",
  ghost:
    "bg-transparent text-zinc-100 hover:bg-zinc-900/60 active:bg-zinc-900 focus-visible:outline-zinc-600",
  danger:
    "bg-rose-600 text-white hover:bg-rose-500 active:bg-rose-700 focus-visible:outline-rose-400",
};

export function Button({
  as: Comp = "button",
  variant = "secondary",
  className,
  disabled,
  ...props
}) {
  return (
    <Comp
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        VARIANTS[variant],
        className
      )}
      disabled={disabled}
      {...props}
    />
  );
}

