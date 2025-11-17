import { useEffect, useMemo, useState } from "react";

/*
  MNIST Carousel Client
  ---------------------
  - Pulls MNIST samples from the public sprite & labels (Google Cloud hosted)
  - Shows a minimal carousel of digits
  - Sends the selected 28x28 grayscale image to your Azure ML endpoint
  - Displays predicted LaTeX, confidence, and correctness vs ground truth

  How to use
  ----------
  1) Put your endpoint URL and key (if any) in the controls at the top.
  2) Click any digit card to select it.
  3) Press Predict.

  Notes
  -----
  - For demo safety we only load the first N_SAMPLES images.
  - The sprite is 784 x 65000; each row (1 x 784) is a single 28x28 image flattened.
  - We reconstruct the 28x28 image on the fly and render to a per-item canvas.
*/

const MNIST_IMAGES_SPRITE =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

// Tweakable: how many examples to load into the carousel
const N_SAMPLES = 200; // keep small for fast page load

// Basic card component
function Card({ children, className = "" }) {
  return (
    <div className={`rounded-2xl shadow-sm border border-gray-200 bg-white ${className}`}>
      {children}
    </div>
  );
}

export default function App() {
  const [endpoint, setEndpoint] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Dataset state
  const [labels, setLabels] = useState<number[]>([]); // length N_SAMPLES
  const [images, setImages] = useState<string[]>([]); // dataURL for each 28x28 canvas

  // UI state
  const [selected, setSelected] = useState<number | null>(null);
  const [result, setResult] = useState<{
    latex: string;
    probs: number[];
    pred: number;
    confidence: number;
    correct: boolean;
  } | null>(null);

  // Load labels and sprite, then reconstruct first N_SAMPLES images
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        setError("");

        // 1) Fetch labels (uint8 one-hot; length = 65000 * 10)
        const labRes = await fetch(MNIST_LABELS);
        const labBuf = await labRes.arrayBuffer();
        const labArr = new Uint8Array(labBuf);
        const lbls: number[] = [];
        const NUM_CLASSES = 10;
        for (let i = 0; i < N_SAMPLES; i++) {
          const base = i * NUM_CLASSES;
          let argmax = 0;
          let best = -1;
          for (let c = 0; c < NUM_CLASSES; c++) {
            const v = labArr[base + c];
            if (v > best) {
              best = v;
              argmax = c;
            }
          }
          lbls.push(argmax);
        }

        // 2) Load sprite image into an offscreen canvas
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const im = new Image();
          im.crossOrigin = "anonymous";
          im.onload = () => resolve(im);
          im.onerror = reject;
          im.src = MNIST_IMAGES_SPRITE;
        });

        // Sprite is 784 x 65000 (width x height). Each row = 1 flattened 28x28 image.
        const SPRITE_W = 784; // 28*28

        const spriteCanvas = document.createElement("canvas");
        spriteCanvas.width = SPRITE_W;
        spriteCanvas.height = img.height; // full height
        const spriteCtx = spriteCanvas.getContext("2d");
        if (!spriteCtx) throw new Error("No 2D context");
        spriteCtx.drawImage(img, 0, 0);

        const perImageCanvases: string[] = [];
        for (let i = 0; i < N_SAMPLES; i++) {
          // Extract the i-th row (y=i) of width 784, height 1
          const rowData = spriteCtx.getImageData(0, i, SPRITE_W, 1);
          // Rebuild a 28x28 ImageData
          const outCanvas = document.createElement("canvas");
          outCanvas.width = 28;
          outCanvas.height = 28;
          const outCtx = outCanvas.getContext("2d");
          if (!outCtx) throw new Error("No out 2D context");
          const out = outCtx.createImageData(28, 28);

          for (let p = 0; p < 28 * 28; p++) {
            const g = rowData.data[p * 4]; // grayscale in R channel
            // map flat index p -> (x, y)
            const y = Math.floor(p / 28);
            const x = p % 28;
            const idx = (y * 28 + x) * 4;
            out.data[idx + 0] = g;
            out.data[idx + 1] = g;
            out.data[idx + 2] = g;
            out.data[idx + 3] = 255;
          }
          outCtx.putImageData(out, 0, 0);
          perImageCanvases.push(outCanvas.toDataURL("image/png"));
        }

        if (!cancelled) {
          setLabels(lbls);
          setImages(perImageCanvases);
          setSelected(0);
        }
      } catch (e: any) {
        if (!cancelled) setError(e?.message || String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedLabel = useMemo(() => (selected != null ? labels[selected] : null), [labels, selected]);

  async function handlePredict() {
    if (selected == null) return;
    if (!endpoint) {
      setError("Please provide your endpoint URL.");
      return;
    }
    setError("");
    setResult(null);

    try {
      setLoading(true);
      // images[selected] is a dataURL; we need its base64 payload
      const dataUrl = images[selected];
      const base64 = dataUrl.split(",")[1];

      const resp = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify({ image_base64: base64 }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }

      const json = await resp.json();

      // Robustly handle APIs that return a JSON string (e.g. '"{\"latex\": ... }"')
      let payload: any = json;
      if (typeof payload === "string") {
        try {
          payload = JSON.parse(payload);
        } catch (e) {
          // leave as-is if it can't be parsed
        }
      }

      // Normalize probs from various possible response shapes
      let probs: number[] = [];

      if (Array.isArray(payload?.probs)) {
        probs = payload.probs.map((v: any) => Number(v));
      } else if (Array.isArray(payload?.predictions)) {
        const firstPred = payload.predictions[0];
        if (firstPred) {
          if (Array.isArray(firstPred.probs)) probs = firstPred.probs.map((v: any) => Number(v));
          else if (Array.isArray(firstPred)) probs = firstPred.map((v: any) => Number(v));
          else if (Array.isArray(firstPred.probabilities)) probs = firstPred.probabilities.map((v: any) => Number(v));
        }
      } else if (Array.isArray(payload)) {
        // handle batch array responses
        const first = payload[0];
        if (first) {
          if (Array.isArray(first.probs)) probs = first.probs.map((v: any) => Number(v));
          else if (Array.isArray(first)) probs = first.map((v: any) => Number(v));
        }
      }

      // Coerce to finite numbers (fallback to 0)
      probs = probs.map((v: any) => {
        const n = Number(v);
        return Number.isFinite(n) ? n : 0;
      });

      // If we didn't find probs, try a few alternate keys
      if (!probs.length) {
        if (Array.isArray(payload?.scores)) probs = payload.scores.map((v: any) => Number(v));
        else if (Array.isArray(payload?.output)) probs = payload.output.map((v: any) => Number(v));
      }

      // If still empty, leave as-is (UI will show no bars)

      // Ensure probs are normalized to a probability distribution in [0,1]
      if (probs.length) {
        const sum = probs.reduce((a, b) => a + b, 0);
        const hasOutOfRange = probs.some((v) => v < 0 || v > 1);
        const notSummingToOne = Math.abs(sum - 1) > 1e-3;

        if (hasOutOfRange || notSummingToOne) {
          // Often responses are logits or un-normalized scores. Apply stable softmax.
          const max = Math.max(...probs);
          const exps = probs.map((v) => Math.exp(v - max));
          const expSum = exps.reduce((a, b) => a + b, 0) || 1;
          probs = exps.map((e) => e / expSum);
        } else if (Math.abs(sum - 1) > 1e-12) {
          // Minor numerical drift: normalize by sum
          const s = sum || 1;
          probs = probs.map((v) => v / s);
        }
      }

      // Compute prediction index (argmax) and confidence safely from normalized probs
      let pred: number = NaN;
      let confidence: number = NaN;
      if (probs.length) {
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[maxIdx]) maxIdx = i;
        }
        pred = maxIdx;
        confidence = probs[maxIdx];
      }

      // Extract and normalize LaTeX string
      const rawLatexRaw = payload?.latex ?? (payload?.predictions ? payload.predictions[0]?.latex : undefined) ?? "";
      let latex = "";
      if (typeof rawLatexRaw === "string") {
        // Normalize double-escaped backslashes (e.g. "\\text{1}")
        latex = rawLatexRaw.replace(/\\\\/g, "\\");
        // If the latex itself is a quoted JSON string, try to unquote it
        if (latex.startsWith('"') && latex.endsWith('"')) {
          try {
            latex = JSON.parse(latex);
          } catch (e) {
            // ignore
          }
        }
      }

      const correct = selectedLabel != null ? pred === selectedLabel : false;

      setResult({ latex, probs, pred, confidence, correct });
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="px-6 py-5 border-b bg-white sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">MNIST → LaTeX Client</h1>
          <div className="flex gap-2">
            <input
              className="px-3 py-2 rounded-xl border w-[28rem]"
              placeholder="Endpoint URL (e.g., https://<region>.inference.ml.azure.com/score)"
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
            />
            <input
              className="px-3 py-2 rounded-xl border w-56"
              placeholder="API Key (optional)"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <button
              className="px-4 py-2 rounded-xl bg-black text-white disabled:opacity-60"
              onClick={handlePredict}
              disabled={selected == null || loading}
            >
              {loading ? "Predicting…" : "Predict"}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-6">
        {/* Status / Error */}
        {error && (
          <Card className="p-4 mb-4 border-red-200 bg-red-50 text-red-800">
            <div className="text-sm font-medium">{error}</div>
          </Card>
        )}

        <section className="mb-6">
          <div className="flex items-end justify-between mb-2">
            <h2 className="text-lg font-semibold">Samples</h2>
            <div className="text-sm text-gray-500">loaded: {images.length}</div>
          </div>

          {/* Carousel */}
          <div className="relative">
            <div className="flex gap-3 overflow-x-auto pb-2 snap-x carousel">
              {images.map((src, i) => (
                <button
                  key={i}
                  onClick={() => setSelected(i)}
                  className={`snap-start shrink-0 ${selected === i ? "ring-2 ring-blue-500" : "ring-1 ring-transparent"} rounded-2xl bg-white border border-gray-200 shadow-sm p-3 flex flex-col items-center w-24 focus:outline-none`}
                  title={`Index ${i} · Label ${labels[i]}`}
                >
                  <img src={src} alt={`mnist-${i}`} className="w-16 h-16 image-render-pixel"/>
                  {/* <div className="text-xs mt-2 text-gray-500">Label: {labels[i]}</div> */}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Selected preview & result */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-5">
            <div className="text-sm text-gray-500 mb-2">Selected</div>
            {selected != null ? (
              <div className="flex items-center gap-5">
                <img src={images[selected]} alt="selected" className="w-28 h-28 image-render-pixel"/>
                <div>
                  <div className="text-sm text-gray-500">Ground truth</div>
                  <div className="text-2xl font-semibold">{labels[selected]}</div>
                  <div className="mt-2 text-xs text-gray-500">Index: {selected}</div>
                </div>
              </div>
            ) : (
              <div className="text-sm text-gray-500">Pick a sample above.</div>
            )}
          </Card>

          <Card className="p-5">
            <div className="text-sm text-gray-500 mb-2">Prediction</div>
            {result ? (
              <div className="space-y-1">
                <div className="text-xl font-semibold font-mono">{result.latex || `\\text{${String(result.pred)}}`}</div>
                <div className="text-sm">Predicted digit: <span className="font-medium">{result.pred}</span></div>
                <div className="text-sm">Confidence: <span className="font-medium">{Number.isFinite(result.confidence) ? `${(result.confidence * 100).toFixed(6)}% (${result.confidence.toFixed(6)})` : "–"}</span></div>
                <div className={`inline-block mt-2 px-2 py-1 rounded-full text-xs ${result.correct ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                  {result.correct ? "Correct" : "Incorrect"}
                </div>
              </div>
            ) : (
              <div className="text-sm text-gray-500">Run a prediction to see results.</div>
            )}
          </Card>
        </section>

        {/* Probabilities bar if present */}
        {result?.probs?.length === 10 && (
          <Card className="p-5 mt-6">
            <div className="text-sm text-gray-500 mb-3">Per-class probabilities</div>
            <div className="grid grid-cols-10 gap-2 items-end">
              {(() => {
                const probs = result.probs || [];
                const maxP = Math.max(...probs);
                const MAX_PX = 120; // full-height in pixels for the largest bar
                const MIN_PX = 6; // minimum visible height
                return probs.map((p, idx) => {
                  // If maxP is tiny/zero, fall back to absolute probability scaling
                  const heightPx = maxP > 0 ? Math.max(MIN_PX, (p / maxP) * MAX_PX) : Math.max(MIN_PX, p * MAX_PX);
                  return (
                    <div key={idx} className="flex flex-col items-center" title={`${(p * 100).toFixed(6)}%`}>
                      <div className="w-6 rounded-t bg-gray-900" style={{ height: `${heightPx}px` }} />
                      <div className="text-[10px] text-gray-500 mt-1">{idx}</div>
                      <div className="text-[10px] text-gray-400">{(p * 100).toFixed(2)}%</div>
                    </div>
                  );
                });
              })()}
            </div>
          </Card>
        )}

      </main>

      <style>{`
        .image-render-pixel { image-rendering: pixelated; }
      `}</style>
    </div>
  );
}
