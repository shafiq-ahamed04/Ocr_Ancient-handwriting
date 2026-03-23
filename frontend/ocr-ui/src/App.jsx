import { useEffect, useMemo, useState } from "react";
import { Button } from "./components/Button";
import { Card, CardBody, CardDescription, CardHeader, CardTitle } from "./components/Card";
import { Dropzone } from "./components/Dropzone";
import { Spinner } from "./components/Spinner";
import { Toast } from "./components/Toast";
import { ProgressSteps } from "./components/ProgressSteps";
import { BoundingBoxViewer } from "./components/BoundingBoxViewer";
import { ConfidenceChart } from "./components/ConfidenceChart";

const PIPELINE_STEPS = ["Upload", "Analyze", "Engine Check", "Extract", "Done"];

function App() {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const [lastRunAt, setLastRunAt] = useState(null);
  
  // State from the backend
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

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setCurrentStep(1); // Preprocess

    try {
      // Simulate pipeline progress for UX
      const progressInterval = setInterval(() => {
        setCurrentStep((prev) => (prev < 3 ? prev + 1 : prev));
      }, 800);

      const res = await fetch(`${apiBase}/ocr`, {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);
      setCurrentStep(4); // Done

      let data = null;
      try {
        data = await res.json();
      } catch {
        // ignore
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
        engine: data?.engine || "Standard Tesseract",
      });
      setLastRunAt(new Date());

      if (!nextText && (!data?.classification || Object.keys(data.classification).length === 0)) {
        setToast({
          tone: "info",
          title: "Analysis Results",
          description: "No text extracted. Try a clearer image.",
        });
      }

    } catch (err) {
      console.error(err);
      setToast({
        tone: "error",
        title: "OCR Failed",
        description: err.message || "Could not connect to the backend server.",
      });
      setCurrentStep(0);
    } finally {
      setLoading(false);
      setTimeout(() => setCurrentStep(4), 500); // ensure it reaches 'Done' visually before stopping spinner
    }
  };

  const onClear = () => {
    setFile(null);
    setText("");
    setLastRunAt(null);
    setAnnotatedUrl(null);
    setBoxes([]);
    setClassification(null);
    setCurrentStep(0);
    setStats({ word_count: 0, char_count: 0, engine: "" });
    setLineResults([]);
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
    <div className="relative min-h-screen bg-zinc-950 selection:bg-indigo-500/30">
      {/* Background gradients */}
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute top-0 w-full h-[500px] bg-[radial-gradient(ellipse_at_top,rgba(99,102,241,0.15),transparent_50%)]" />
        <div className="absolute top-1/4 left-0 w-[500px] h-[500px] bg-[radial-gradient(ellipse_at_center,rgba(16,185,129,0.05),transparent_50%)]" />
        <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-[radial-gradient(ellipse_at_bottom_right,rgba(244,63,94,0.05),transparent_50%)]" />
      </div>

      <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
        
        {/* Header */}
        <header className="flex flex-col gap-4 border-b border-zinc-800/80 pb-6 mb-2">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
            <div>
              <div className="flex items-center gap-3">
                <div className="flex size-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/20">
                  <span className="text-xl font-bold text-white leading-none tracking-tighter">அ</span>
                </div>
                <h1 className="text-3xl font-bold tracking-tight text-white">
                  Ancient Tamil OCR
                </h1>
              </div>
              <p className="mt-2 text-[15px] text-zinc-400 max-w-2xl">
                Upload any Tamil document—printed or handwritten. Our <b>Hybrid AI Pipeline</b> 
                automatically detects the document type and selects either Tesseract or 
                a custom Deep Learning CRNN for the best accuracy.
              </p>
            </div>
            
            <div className="flex items-center gap-3">
              <Button variant="ghost" onClick={onClear} disabled={loading && !file && !text && !annotatedUrl}>
                Reset All
              </Button>
              <Button 
                variant="primary" 
                onClick={runOCR} 
                disabled={loading || !file}
                className="shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_25px_rgba(99,102,241,0.5)] bg-gradient-to-r from-indigo-600 to-indigo-500 border border-indigo-500"
              >
                {loading ? (
                  <>
                    <Spinner className="border-indigo-200 border-t-transparent size-4" />
                    Processing Pipeline…
                  </>
                ) : (
                  <>
                    <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Run OCR Engine
                  </>
                )}
              </Button>
            </div>
          </div>

          {(loading || currentStep > 0) && (
            <div className="mt-4 py-3 px-4 rounded-xl bg-zinc-900/40 border border-zinc-800/60 backdrop-blur-sm">
              <ProgressSteps 
                steps={PIPELINE_STEPS} 
                currentStep={currentStep} 
                loading={loading}
              />
            </div>
          )}
        </header>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 min-h-[600px]">
          
          {/* Left Column - Input */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <Card className="flex-1 flex flex-col shadow-xl shadow-black/20">
              <CardHeader className="bg-zinc-900/40 pb-4">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <svg className="size-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Input Document
                  </CardTitle>
                  <CardDescription>Upload an image for text extraction.</CardDescription>
                </div>
              </CardHeader>
              <CardBody className="flex-1 flex flex-col pt-4">
                
                {annotatedUrl ? (
                   <BoundingBoxViewer 
                     imageUrl={previewUrl} 
                     annotatedUrl={annotatedUrl}
                     boxes={boxes}
                     className="mb-4"
                   />
                ) : (
                  <Dropzone
                    className="flex-1 flex flex-col mb-4"
                    file={file}
                    previewUrl={previewUrl}
                    disabled={loading}
                    onPickFile={(f) => {
                      if (shouldRejectFile(f)) return;
                      setFile(f);
                      // Auto-clear old results when new file is picked
                      setText("");
                      setAnnotatedUrl(null);
                      setBoxes([]);
                      setClassification(null);
                      setCurrentStep(0);
                      setStats({ word_count: 0, char_count: 0, engine: "" });
                    }}
                    onClear={() => {
                      setFile(null);
                      setAnnotatedUrl(null);
                    }}
                  />
                )}

                <div className="mt-auto pt-4 flex flex-wrap gap-2 border-t border-zinc-800/50 justify-between items-center text-xs text-zinc-500">
                  <span>Backend: {apiBase || "proxy:8000"}</span>
                  {lastRunAt && <span>Ran at: {lastRunAt.toLocaleTimeString()}</span>}
                </div>
              </CardBody>
            </Card>

            {/* Classification Card (Shows if a single character is analyzed) */}
            {classification && Object.keys(classification).length > 0 && !classification.error && (
              <ConfidenceChart classification={classification} />
            )}
            
          </div>

          {/* Right Column - Extracted Text */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            <Card className="flex-1 flex flex-col shadow-xl shadow-black/20">
              <CardHeader className="bg-zinc-900/40 pb-4">
                <div className="flex-1">
                  <div className="flex flex-wrap items-center gap-4">
                    <CardTitle className="flex items-center gap-2">
                      <svg className="size-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Digitized Content
                    </CardTitle>
                    {stats.engine && (
                      <div className="flex items-center gap-1.5 rounded-full bg-indigo-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-indigo-400 border border-indigo-500/20 shadow-sm">
                        <div className="size-1 rounded-full bg-indigo-400 animate-pulse" />
                        AI Engine: {stats.engine}
                      </div>
                    )}
                  </div>
                  <CardDescription>
                    {stats.word_count > 0 
                      ? `${stats.word_count} words · ${stats.char_count} characters detected`
                      : "Awaiting execution..."}
                  </CardDescription>
                </div>
                
                <div className="flex flex-wrap items-center gap-2">
                  <Button variant="ghost" onClick={onCopy} disabled={!text} className="text-xs px-3">
                    Copy Text
                  </Button>
                  <div className="h-4 w-px bg-zinc-700" />
                  <Button variant="secondary" onClick={onDownloadTxt} disabled={!text} className="text-xs px-3 bg-zinc-800 hover:bg-zinc-700">
                    TXT
                  </Button>
                  <Button variant="primary" onClick={onDownloadPdf} disabled={!text} className="text-xs px-3 bg-rose-600 hover:bg-rose-500 focus-visible:outline-rose-400 border-rose-500">
                    PDF
                  </Button>
                </div>
              </CardHeader>
              
              <CardBody className="flex-1 flex flex-col p-0">
                <div className="relative flex-1 bg-[#1a1c23] m-4 rounded-xl border border-zinc-800/80 overflow-hidden shadow-inner">
                  {text ? (
                    <textarea 
                      readOnly
                      value={text}
                      className="absolute inset-0 w-full h-full resize-none bg-transparent p-5 text-[15px] leading-relaxed text-zinc-100 font-sans focus:outline-none selection:bg-indigo-500/40"
                    />
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-600 gap-3">
                      <svg className="size-12 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                      <p className="text-sm font-medium">No digitized text available</p>
                    </div>
                  )}
                </div>
              </CardBody>
            </Card>
          </div>
          
        </div>

      </div>

      {toast ? (
        <div className="pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center px-4">
          <Toast {...toast} onClose={() => setToast(null)} />
        </div>
      ) : null}
    </div>
  );
}

export default App;

