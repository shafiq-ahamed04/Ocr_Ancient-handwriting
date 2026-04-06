export function Spinner({ className }) {
  return (
    <span
      className={`inline-block size-4 animate-spin border-2 border-primary border-t-transparent ${className || ""}`}
      aria-label="Loading"
      role="status"
    />
  );
}
