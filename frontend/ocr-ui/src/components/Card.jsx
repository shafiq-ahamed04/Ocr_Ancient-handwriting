import clsx from "clsx";

export function Card({ className, ...props }) {
  return (
    <div
      className={clsx(
        "rounded-2xl border border-zinc-800/70 bg-zinc-950/40 shadow-[0_0_0_1px_rgba(255,255,255,0.03)] backdrop-blur",
        className
      )}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }) {
  return (
    <div
      className={clsx(
        "flex items-start justify-between gap-4 border-b border-zinc-800/70 p-5",
        className
      )}
      {...props}
    />
  );
}

export function CardTitle({ className, ...props }) {
  return (
    <div className={clsx("text-base font-semibold text-zinc-100", className)} {...props} />
  );
}

export function CardDescription({ className, ...props }) {
  return (
    <div className={clsx("mt-1 text-sm text-zinc-400", className)} {...props} />
  );
}

export function CardBody({ className, ...props }) {
  return <div className={clsx("p-5", className)} {...props} />;
}

