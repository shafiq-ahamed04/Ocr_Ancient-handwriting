import { useState } from "react";

export function BoundingBoxViewer({ imageUrl, annotatedUrl, boxes, className }) {
  const [showBoxes, setShowBoxes] = useState(true);

  if (!imageUrl && !annotatedUrl) return null;

  return (
    <div className={`relative w-full overflow-hidden border border-outline-variant bg-surface-container-lowest ${className || ""}`}>
      <div className="absolute top-2 right-2 z-10 flex gap-2">
        {annotatedUrl && (
          <button
            onClick={() => setShowBoxes(!showBoxes)}
            className="bg-surface-container/80 px-2.5 py-1 text-xs font-bold font-label text-on-surface-variant backdrop-blur hover:text-primary transition uppercase tracking-wider"
          >
            {showBoxes ? "Hide Boxes" : "Show Boxes"}
          </button>
        )}
      </div>

      <div className="relative aspect-[4/3] w-full bg-surface-container flex items-center justify-center p-2">
        {annotatedUrl && showBoxes ? (
          <img
            src={annotatedUrl}
            alt="Annotated document"
            className="w-full h-full object-contain"
          />
        ) : (
          <img
            src={imageUrl}
            alt="Original document"
            className="w-full h-full object-contain"
          />
        )}
      </div>

      {boxes && boxes.length > 0 && (
        <div className="border-t border-outline-variant/30 bg-surface-container-lowest px-4 py-2 text-xs flex justify-between">
          <span className="text-on-surface-variant font-mono">Detected {boxes.length} text regions</span>
          <span className="text-emerald-500 font-mono">
            Avg confidence: {Math.round(boxes.reduce((a, b) => a + b.confidence, 0) / boxes.length)}%
          </span>
        </div>
      )}
    </div>
  );
}
