import { useEffect, useMemo, useState } from "react";
import { ThemeToggle } from "./components/ThemeToggle";
import { Dropzone } from "./components/Dropzone";
import { Toast } from "./components/Toast";
import { ProcessingPipeline } from "./components/ProcessingPipeline";
import { OutputView } from "./components/OutputView";
import { BoundingBoxViewer } from "./components/BoundingBoxViewer";
import { ConfidenceChart } from "./components/ConfidenceChart";

const VIEWS = { UPLOAD: "upload", PROCESSING: "processing", OUTPUT: "output" };

function App() {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const [view, setView] = useState(VIEWS.UPLOAD);
  const [selectedMode, setSelectedMode] = useState("palmleaf");

  // Backend response state
  const [currentStep, setCurrentStep] = useState(0);
  const [annotatedUrl, setAnnotatedUrl] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [classification, setClassification] = useState(null);
  const [stats, setStats] = useState({ word_count: 0, char_count: 0, engine: "" });
  const [lineResults, setLineResults] = useState([]);

  const previewUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const apiBase = import.meta.env.VITE_API_BASE ?? "";

  const runOCR = async () => {
    if (!file) {
      setToast({
        tone: "warning",
        title: "No image selected",
        description: "Choose or drop an image first, then run OCR.",
      });
      return;
    }

    const endpoint = selectedMode === "palmleaf" ? "/ocr/palmleaf" : "/ocr";
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setCurrentStep(0);
    setView(VIEWS.PROCESSING);

    try {
      // Animate pipeline steps
      const stepTimers = [
        setTimeout(() => setCurrentStep(1), 600),
        setTimeout(() => setCurrentStep(2), 1800),
      ];

      const res = await fetch(`${apiBase}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      stepTimers.forEach(clearTimeout);
      setCurrentStep(3);

      let data = null;
      try {
        data = await res.json();
      } catch {
        // ignore parse error
      }

      if (!res.ok) {
        const message = (data && (data.detail || data.message)) || `Backend returned ${res.status}`;
        throw new Error(message);
      }

      const nextText = (data && typeof data.text === "string" ? data.text : "").trim();
      setText(nextText);
      setAnnotatedUrl(data?.annotated_image || null);
      setBoxes(data?.boxes || []);
      setClassification(data?.classification || null);
      setLineResults(data?.lines || []);
      setStats({
        word_count: data?.word_count || 0,
        char_count: data?.char_count || 0,
        engine: data?.engine || (selectedMode === "palmleaf" ? "CRNN" : "Tesseract"),
      });

      // Show Done briefly, then move to output
      setTimeout(() => {
        setCurrentStep(4);
        setTimeout(() => setView(VIEWS.OUTPUT), 800);
      }, 500);

    } catch (err) {
      console.error(err);
      setToast({
        tone: "error",
        title: "OCR Failed",
        description: err.message || "Could not connect to the backend server.",
      });
      setView(VIEWS.UPLOAD);
      setCurrentStep(0);
    } finally {
      setLoading(false);
    }
  };

  const onClear = () => {
    setFile(null);
    setText("");
    setAnnotatedUrl(null);
    setBoxes([]);
    setClassification(null);
    setCurrentStep(0);
    setStats({ word_count: 0, char_count: 0, engine: "" });
    setLineResults([]);
    setView(VIEWS.UPLOAD);
  };

  const shouldRejectFile = (maybeFile) => {
    if (!maybeFile) return false;
    const maxBytes = 10 * 1024 * 1024;
    if (maybeFile.size > maxBytes) {
      setToast({
        tone: "warning",
        title: "File too large",
        description: "Please upload an image under 10 MB.",
      });
      return true;
    }
    return false;
  };

  const onCopy = async () => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setToast({ tone: "success", title: "Copied", description: "Text copied to clipboard." });
    } catch {
      setToast({
        tone: "error",
        title: "Copy failed",
        description: "Your browser blocked clipboard access.",
      });
    }
  };

  const onDownloadTxt = () => {
    if (!text) return;
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "extracted-tamil-text.txt";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const onDownloadPdf = async () => {
    if (!text) return;
    try {
      const formData = new FormData();
      formData.append("text", text);
      const res = await fetch(`${apiBase}/export/pdf`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Failed to generate PDF");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "extracted-tamil-text.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setToast({ tone: "error", title: "PDF Export Failed", description: err.message });
    }
  };

  return (
    <div className="relative min-h-screen flex flex-col">
      {/* Decorative corner lines */}
      <div className="fixed top-0 left-0 w-32 h-px bg-gradient-to-r from-primary/20 to-transparent pointer-events-none z-50" />
      <div className="fixed top-0 left-0 w-px h-32 bg-gradient-to-b from-primary/20 to-transparent pointer-events-none z-50" />
      <div className="fixed bottom-0 right-0 w-32 h-px bg-gradient-to-l from-primary/20 to-transparent pointer-events-none z-50" />
      <div className="fixed bottom-0 right-0 w-px h-32 bg-gradient-to-t from-primary/20 to-transparent pointer-events-none z-50" />

      {view === VIEWS.UPLOAD && (
        <UploadView
          file={file}
          previewUrl={previewUrl}
          loading={loading}
          selectedMode={selectedMode}
          annotatedUrl={annotatedUrl}
          boxes={boxes}
          onSelectMode={setSelectedMode}
          onPickFile={(f) => {
            if (shouldRejectFile(f)) return;
            setFile(f);
            setText("");
            setAnnotatedUrl(null);
            setBoxes([]);
            setClassification(null);
            setCurrentStep(0);
            setStats({ word_count: 0, char_count: 0, engine: "" });
          }}
          onClearFile={() => {
            setFile(null);
            setAnnotatedUrl(null);
          }}
          onRunOCR={runOCR}
        />
      )}

      {view === VIEWS.PROCESSING && (
        <ProcessingPipeline
          currentStep={currentStep}
          fileName={file?.name || "document.jpg"}
          previewUrl={previewUrl}
          selectedMode={selectedMode}
        />
      )}

      {view === VIEWS.OUTPUT && (
        <OutputView
          text={text}
          stats={stats}
          previewUrl={previewUrl}
          annotatedUrl={annotatedUrl}
          boxes={boxes}
          classification={classification}
          fileName={file?.name || "document.jpg"}
          selectedMode={selectedMode}
          onCopy={onCopy}
          onDownloadTxt={onDownloadTxt}
          onDownloadPdf={onDownloadPdf}
          onProcessAnother={onClear}
        />
      )}

      {/* Toast */}
      {toast && (
        <div className="pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center px-4">
          <Toast {...toast} onClose={() => setToast(null)} />
        </div>
      )}
    </div>
  );
}

/** Upload & Mode Selection View */
function UploadView({
  file,
  previewUrl,
  loading,
  selectedMode,
  annotatedUrl,
  boxes,
  onSelectMode,
  onPickFile,
  onClearFile,
  onRunOCR,
}) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12 overflow-x-hidden">
      {/* Side Decoration */}
      <aside className="fixed left-8 top-1/2 -translate-y-1/2 hidden xl:flex flex-col space-y-12 opacity-20 pointer-events-none">
        <div className="font-mono text-[10px] [writing-mode:vertical-lr] tracking-[1em] text-primary uppercase">
          System.Core.Initializing...
        </div>
        <div className="w-px h-32 bg-outline-variant mx-auto" />
        <div className="font-mono text-[10px] [writing-mode:vertical-lr] tracking-[1em] text-primary uppercase">
          Buffer_Stream_001
        </div>
      </aside>
      <aside className="fixed right-8 top-1/2 -translate-y-1/2 hidden xl:flex flex-col space-y-12 opacity-20 pointer-events-none">
        <div className="font-mono text-[10px] [writing-mode:vertical-lr] tracking-[1em] text-primary uppercase">
          Tamil_Glyph_Lib_v2
        </div>
        <div className="w-px h-32 bg-outline-variant mx-auto" />
        <div className="font-mono text-[10px] [writing-mode:vertical-lr] tracking-[1em] text-primary uppercase">
          Cognitive_Active
        </div>
      </aside>

      {/* Header */}
      <header className="flex flex-col items-center mb-16">
        <div className="w-20 h-20 bg-primary-container flex items-center justify-center mb-6 relative group overflow-hidden">
          <span className="text-on-primary-container text-4xl font-bold relative z-10 font-tamil">த</span>
          <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
        <div className="flex items-center gap-4">
          <h1 className="text-5xl font-black tracking-tighter text-primary uppercase font-headline">ATCRS</h1>
          <ThemeToggle />
        </div>
        <p className="text-on-surface-variant text-sm font-mono tracking-[0.2em] uppercase opacity-70 mt-2">
          Digitizing Tamil Heritage with AI
        </p>
      </header>

      {/* Main Content */}
      <main className="w-full max-w-4xl space-y-12">
        {/* Upload Zone */}
        <section className="relative group">
          {annotatedUrl ? (
            <BoundingBoxViewer
              imageUrl={previewUrl}
              annotatedUrl={annotatedUrl}
              boxes={boxes}
            />
          ) : (
            <Dropzone
              file={file}
              previewUrl={previewUrl}
              disabled={loading}
              onPickFile={onPickFile}
              onClear={onClearFile}
            />
          )}
        </section>

        {/* Mode Selection */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Standard OCR */}
          <article
            id="mode-standard"
            onClick={() => onSelectMode("standard")}
            className={`p-8 border cursor-pointer group flex flex-col justify-between min-h-[180px] transition-all ${
              selectedMode === "standard"
                ? "bg-surface-container-high border-2 border-primary"
                : "bg-surface-container border-outline-variant hover:border-outline"
            }`}
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className={`text-xl font-bold mb-3 flex items-center gap-2 font-headline ${
                  selectedMode === "standard" ? "text-primary" : "text-on-surface"
                }`}>
                  <span className="material-symbols-outlined text-on-surface-variant">description</span>
                  Standard OCR
                </h3>
                <p className="text-on-surface-variant text-sm leading-relaxed max-w-[280px]">
                  Best for printed Tamil text and typed documents. Uses Tesseract 5.0 engine.
                </p>
              </div>
              {selectedMode === "standard" && (
                <div className="w-4 h-4 bg-primary flex items-center justify-center">
                  <span className="material-symbols-outlined text-on-primary text-[12px] font-bold">check</span>
                </div>
              )}
            </div>
            <div className={`mt-4 flex items-center gap-2 transition-opacity ${
              selectedMode === "standard" ? "opacity-100" : "opacity-30 group-hover:opacity-100"
            }`}>
              {selectedMode === "standard" ? (
                <>
                  <span className="inline-block w-2 h-2 bg-primary animate-pulse" />
                  <span className="text-[10px] font-mono uppercase text-primary tracking-tighter">Engine Active</span>
                </>
              ) : (
                <span className="text-[10px] font-mono uppercase text-on-surface-variant tracking-tighter">Ready for stream</span>
              )}
            </div>
          </article>

          {/* Palm Leaf Mode */}
          <article
            id="mode-palmleaf"
            onClick={() => onSelectMode("palmleaf")}
            className={`relative p-8 cursor-pointer group flex flex-col justify-between min-h-[180px] transition-all ${
              selectedMode === "palmleaf"
                ? "bg-surface-container-high border-2 border-primary"
                : "bg-surface-container border border-outline-variant hover:border-outline"
            }`}
          >
            {selectedMode === "palmleaf" && (
              <div className="absolute inset-0 pointer-events-none" style={{
                background: "radial-gradient(circle at center, rgba(44, 57, 201, 0.1) 0%, transparent 70%)"
              }} />
            )}
            <div className="relative flex items-start justify-between">
              <div>
                <h3 className={`text-xl font-bold mb-3 flex items-center gap-2 font-headline ${
                  selectedMode === "palmleaf" ? "text-primary" : "text-on-surface"
                }`}>
                  <span className="material-symbols-outlined" style={selectedMode === "palmleaf" ? {fontVariationSettings: "'FILL' 1"} : {}}>potted_plant</span>
                  Palm Leaf Mode
                </h3>
                <p className={`text-sm leading-relaxed max-w-[280px] ${
                  selectedMode === "palmleaf" ? "text-on-surface" : "text-on-surface-variant"
                }`}>
                  Deep learning CRNN for ancient manuscripts and palm leaf inscriptions.
                </p>
              </div>
              {selectedMode === "palmleaf" && (
                <div className="w-4 h-4 bg-primary flex items-center justify-center">
                  <span className="material-symbols-outlined text-on-primary text-[12px] font-bold">check</span>
                </div>
              )}
            </div>
            <div className={`relative mt-4 flex items-center gap-2 transition-opacity ${
              selectedMode === "palmleaf" ? "opacity-100" : "opacity-30 group-hover:opacity-100"
            }`}>
              {selectedMode === "palmleaf" ? (
                <>
                  <span className="inline-block w-2 h-2 bg-primary animate-pulse" />
                  <span className="text-[10px] font-mono uppercase text-primary tracking-tighter">AI Core Engaged</span>
                </>
              ) : (
                <span className="text-[10px] font-mono uppercase text-on-surface-variant tracking-tighter">Ready for stream</span>
              )}
            </div>
          </article>
        </section>

        {/* Action Button */}
        <footer className="flex justify-center pt-8">
          <button
            id="begin-analysis-btn"
            onClick={onRunOCR}
            disabled={loading || !file}
            className="w-full max-w-[480px] bg-primary-container text-on-primary-container py-6 px-12 text-xl font-black uppercase tracking-[0.2em] flex items-center justify-center gap-4 transition-all group hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed font-headline"
          >
            {loading ? "Processing..." : "Begin Analysis"}
            <span className="material-symbols-outlined group-hover:translate-x-2 transition-transform">arrow_forward</span>
          </button>
        </footer>
      </main>
    </div>
  );
}

export default App;
