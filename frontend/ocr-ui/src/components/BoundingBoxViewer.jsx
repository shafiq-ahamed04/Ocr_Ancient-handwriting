import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

export function BoundingBoxViewer({ imageUrl, annotatedUrl, boxes, className }) {
  const [showBoxes, setShowBoxes] = useState(true);
  const [hoveredBox, setHoveredBox] = useState(null);
  
  if (!imageUrl && !annotatedUrl) return null;

  return (
    <div className={clsx("relative w-full overflow-hidden rounded-xl border border-zinc-800/70 bg-zinc-950/50", className)}>
      <div className="absolute top-2 right-2 z-10 flex gap-2">
        {annotatedUrl && (
          <button
            onClick={() => setShowBoxes(!showBoxes)}
            className="rounded-lg bg-zinc-900/80 px-2.5 py-1 text-xs font-medium text-zinc-300 backdrop-blur hover:bg-zinc-800 hover:text-white transition"
          >
            {showBoxes ? "Hide Boxes" : "Show Boxes"}
          </button>
        )}
      </div>

      <div className="relative aspect-[4/3] w-full bg-zinc-900/30 flex items-center justify-center p-2">
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
      
      {/* Box stats strip */}
      {boxes && boxes.length > 0 && (
        <div className="border-t border-zinc-800/50 bg-zinc-950/80 px-4 py-2 text-xs text-zinc-400 flex justify-between backdrop-blur">
          <span>Detected {boxes.length} text regions</span>
          <span className="text-emerald-400/80">Average confidence: {Math.round(boxes.reduce((a, b) => a + b.confidence, 0) / boxes.length)}%</span>
        </div>
      )}
    </div>
  );
}
