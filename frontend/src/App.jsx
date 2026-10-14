import { useEffect, useRef, useState } from "react";
import {
  Activity, ArrowRight, Check, ChevronDown, CircleAlert, Database,
  FileImage, FlaskConical, Image as ImageIcon, Info, Layers3, LoaderCircle,
  Moon, Search, ShieldCheck, SlidersHorizontal, Sun, Upload, X
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const METHODS = [
  { id: "intensite", label: "Intensity histogram", detail: "32-bin grayscale distribution" },
  { id: "texture", label: "GLCM texture", detail: "Contrast, energy, homogeneity and correlation" },
  { id: "forme", label: "Hu shape moments", detail: "Seven compact shape descriptors" }
];
const CLASSES = {
  benign: { label: "Benign folder", cls: "benign" },
  malignant: { label: "Malignant folder", cls: "malignant" },
  normal: { label: "Normal folder", cls: "normal" },
  unknown: { label: "Unlabelled", cls: "unknown" }
};
const count = (value) => Number(value || 0).toLocaleString("en-US");
const distance = (value) => Number(value).toFixed(4);

export default function App() {
  const inputRef = useRef(null);
  const [theme, setTheme] = useState(() => localStorage.getItem("cbir-theme") || "dark");
  const [api, setApi] = useState("checking");
  const [dataset, setDataset] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [methods, setMethods] = useState(["intensite", "texture", "forme"]);
  const [topK, setTopK] = useState(5);
  const [metric, setMetric] = useState("euclidienne");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState(null);
  const [selected, setSelected] = useState(null);
  const [page, setPage] = useState("search");

  useEffect(() => {
    localStorage.setItem("cbir-theme", theme);
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    let active = true;
    fetch(API_BASE + "/api/status")
      .then((r) => { if (!r.ok) throw new Error("API unavailable"); return r.json(); })
      .then((data) => { if (active) { setDataset(data); setApi("online"); } })
      .catch(() => { if (active) setApi("offline"); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!file) { setPreview(""); return undefined; }
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  function chooseFile(nextFile) {
    setError("");
    setResults(null);
    setSelected(null);
    if (!nextFile) return;
    if (!["image/png", "image/jpeg", "image/webp", "image/bmp"].includes(nextFile.type)) {
      setFile(null);
      setError("Choose a PNG, JPG, WEBP or BMP image.");
      return;
    }
    if (nextFile.size > 15 * 1024 * 1024) {
      setFile(null);
      setError("The selected image exceeds the 15 MB upload limit.");
      return;
    }
    setFile(nextFile);
    setPage("search");
  }

  function toggleMethod(id) {
    setMethods((current) => current.includes(id) ? current.filter((x) => x !== id) : [...current, id]);
  }

  async function runSearch(event) {
    if (event) event.preventDefault();
    if (!file) { setError("Choose a query image first."); return; }
    if (!methods.length) { setError("Select at least one descriptor."); return; }
    setLoading(true);
    setError("");
    setResults(null);
    setSelected(null);
    const body = new FormData();
    body.append("file", file);
    body.append("top_k", String(topK));
    body.append("distance_metric", metric);
    methods.forEach((method) => body.append("methods", method));
    try {
      const response = await fetch(API_BASE + "/api/search", { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Retrieval failed with HTTP " + response.status + ".");
      setResults(data);
    } catch (err) {
      setError(err.message || "Could not complete the retrieval request.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setResults(null);
    setSelected(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  const saved = dataset && dataset.saved_evaluation;
  const classEntries = dataset ? Object.entries(dataset.class_counts || {}) : [];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#top" onClick={() => setPage("search")}>
          <span className="brand-mark"><Layers3 size={21} /></span>
          <span className="brand-copy"><strong>RESEARCH</strong><span>ARCHIVE / CBIR</span></span>
        </a>
        <div className="sidebar-label">WORKSPACE</div>
        <nav className="main-nav">
          <button className={"nav-item " + (page === "search" ? "active" : "")} onClick={() => setPage("search")}><Search size={17}/><span>Image retrieval</span><span className="nav-count">01</span></button>
          <button className={"nav-item " + (page === "dataset" ? "active" : "")} onClick={() => setPage("dataset")}><Database size={17}/><span>Dataset overview</span></button>
          <button className={"nav-item " + (page === "methods" ? "active" : "")} onClick={() => setPage("methods")}><FlaskConical size={17}/><span>Methods & metrics</span></button>
        </nav>
        <div className="sidebar-spacer"/>
        <div className="dataset-mini">
          <div className="mini-heading"><span className={"status-dot " + (api === "online" ? "online" : "")}/>{ "LOCAL SERVICE" }</div>
          <strong>{api === "online" ? "API connected" : api === "checking" ? "Connecting…" : "API unavailable"}</strong>
          <span>{dataset && dataset.dataset_available ? count(dataset.total_images) + " images available for retrieval" : "BUSI dataset required locally"}</span>
        </div>
        <div className="sidebar-footer"><ShieldCheck size={14}/> Research use only · Not diagnostic</div>
      </aside>

      <main className="main-content" id="top">
        <header className="topbar">
          <div className="breadcrumbs"><span>Projects</span><span className="crumb-slash">/</span><strong>Medical CBIR System</strong></div>
          <div className="topbar-actions"><span className="project-tag"><span className="tag-dot"/> ACADEMIC / PRACTICE</span><button className="icon-button" aria-label="Toggle color theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun size={17}/> : <Moon size={17}/>}</button></div>
        </header>

        <section className="page-heading">
          <div><div className="eyebrow"><span>MEDICAL IMAGING</span><span className="eyebrow-divider">/</span><span>CONTENT-BASED RETRIEVAL</span></div><h1>Find visual <em>similarity.</em></h1><p className="heading-copy">Explore breast ultrasound images through handcrafted visual descriptors, distance-based ranking, and transparent retrieval metrics.</p></div>
          <div className="heading-stamp"><span>BUSI</span><span>DATASET</span><span className="stamp-line"/><span>v1.0</span></div>
        </section>

        {api === "offline" && <div className="notice notice-error"><CircleAlert size={18}/><div><strong>Local API is not responding.</strong><span>Start the FastAPI service in the project root, then reload this page. Expected address: {API_BASE}</span></div></div>}
        {api === "online" && dataset && !dataset.dataset_available && <div className="notice notice-warn"><Info size={18}/><div><strong>BUSI dataset not found.</strong><span>Place the dataset under data/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/ with benign, malignant and normal folders. The image preview works, but retrieval needs the local dataset.</span><a href="https://github.com/ghada-59/Medical_CBIR_System#dataset" target="_blank" rel="noreferrer">Dataset setup instructions <ArrowRight size={13}/></a></div></div>}
        {error && <div className="notice notice-error" role="alert"><CircleAlert size={18}/><div><strong>Could not complete retrieval</strong><span>{error}</span></div><button className="notice-close" onClick={() => setError("")} aria-label="Dismiss error"><X size={16}/></button></div>}

        {page === "search" && <>
          <div className="section-title-row"><div><span className="section-index">01</span><h2>Retrieval workspace</h2></div><span className="section-note">QUERY → DESCRIPTORS → RANKED RESULTS</span></div>
          <div className="workspace-grid">
            <section className="panel query-panel">
              <div className="panel-head"><div className="panel-title"><FileImage size={17}/><h3>Query image</h3></div><span className="step-pill">STEP 01</span></div>
              <input ref={inputRef} className="visually-hidden" type="file" accept=".png,.jpg,.jpeg,.webp,.bmp,image/png,image/jpeg,image/webp,image/bmp" onChange={(e) => chooseFile(e.target.files && e.target.files[0])}/>
              {preview ? <div className="query-preview">
                <div className="image-frame"><img src={preview} alt="Selected query image preview"/></div>
                <div className="preview-meta"><div><strong>{file.name}</strong><span>{(file.size / 1024).toFixed(0)} KB · preview ready</span></div><button className="small-icon-button" onClick={reset} aria-label="Remove selected image"><X size={15}/></button></div>
                <div className="preview-status"><Check size={14}/> Image loaded locally · preprocessing to 128 × 128</div>
              </div> : <button className="dropzone" onClick={() => inputRef.current && inputRef.current.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => {e.preventDefault(); chooseFile(e.dataTransfer.files && e.dataTransfer.files[0]);}}>
                <span className="upload-symbol"><Upload size={21}/></span><strong>Choose an ultrasound image</strong><span>Drag and drop here, or browse files</span><span className="file-types">PNG · JPG · WEBP · BMP <i>MAX 15 MB</i></span>
              </button>}
              {preview && <button className="replace-link" onClick={() => inputRef.current && inputRef.current.click()}><Upload size={14}/> Replace query image</button>}
              <div className="query-footnote"><Info size={14}/><span>Image preview appears immediately. Search is a separate action and requires the local BUSI dataset.</span></div>
            </section>

            <section className="panel settings-panel">
              <div className="panel-head"><div className="panel-title"><SlidersHorizontal size={17}/><h3>Search configuration</h3></div><span className="step-pill">STEP 02</span></div>
              <div className="setting-block"><div className="setting-label"><span>01 / DESCRIPTORS</span><span className="muted">{methods.length} selected</span></div>
                <div className="descriptor-list">{METHODS.map((m, i) => <label className={"descriptor-option " + (methods.includes(m.id) ? "selected" : "")} key={m.id}><input type="checkbox" checked={methods.includes(m.id)} onChange={() => toggleMethod(m.id)}/><span className="descriptor-check">{methods.includes(m.id) && <Check size={12}/>}</span><span className="descriptor-number">0{i + 1}</span><span className="descriptor-copy"><strong>{m.label}</strong><small>{m.detail}</small></span></label>)}</div>
              </div>
              <div className="setting-row"><label className="field-label">Retrieved images <span>TOP-K</span></label><div className="segmented-control">{[3,5,8,10].map((k) => <button key={k} className={topK === k ? "selected" : ""} onClick={() => setTopK(k)}>{k}</button>)}</div></div>
              <div className="setting-row"><label className="field-label" htmlFor="distance">Distance metric</label><div className="select-wrap"><select id="distance" value={metric} onChange={(e) => setMetric(e.target.value)}><option value="euclidienne">Euclidean distance</option><option value="cosinus">Cosine distance</option></select><ChevronDown size={15}/></div><p className="field-hint">Lower distance indicates a closer descriptor match.</p></div>
              <button className="primary-button" disabled={!file || !methods.length || !dataset || !dataset.dataset_available || api !== "online" || loading} onClick={runSearch}>{loading ? <><LoaderCircle className="spin" size={17}/> Indexing & searching…</> : <><Search size={17}/> Run similarity search <ArrowRight size={16}/></>}</button>
              {(!dataset || !dataset.dataset_available) && <p className="button-help">Add the BUSI dataset to enable retrieval.</p>}
              {dataset && dataset.dataset_available && !file && <p className="button-help">Select a query image to begin.</p>}
            </section>
          </div>

          <div className="section-title-row results-title"><div><span className="section-index">02</span><h2>Retrieved collection</h2></div><span className="section-note">{results ? results.results.length + " MATCHES" : "RESULTS APPEAR AFTER SEARCH"}</span></div>
          {loading ? <div className="empty-state"><LoaderCircle className="spin" size={28}/><strong>Building descriptor index</strong><span>Extracting image features and ranking nearest neighbours. The first search may take longer.</span></div> :
            results ? <section className="results-area">
              <div className="result-summary"><div><span className="summary-label">QUERY</span><strong>{results.query.filename}</strong></div><div><span className="summary-label">DATABASE</span><strong>{count(results.dataset_size)} images</strong></div><div><span className="summary-label">DESCRIPTORS</span><strong>{results.methods.map((m) => m.label).join(" + ")}</strong></div><div><span className="summary-label">DISTANCE</span><strong>{results.distance_label}</strong></div></div>
              <div className="results-grid">{results.results.map((item) => <button className={"result-card " + (selected && selected.rank === item.rank ? "chosen" : "")} key={item.relative_path + item.rank} onClick={() => setSelected(item)}>
                <div className="result-image"><img src={item.preview} alt={"Retrieved ultrasound result " + item.rank} loading="lazy"/><span className="rank-badge">#{String(item.rank).padStart(2,"0")}</span></div>
                <div className="result-card-body"><div className="result-card-top"><span className="result-file" title={item.filename}>{item.filename}</span><span className={"class-pill " + (CLASSES[item.class] || CLASSES.unknown).cls}>{(CLASSES[item.class] || CLASSES.unknown).label}</span></div><div className="distance-row"><span>Distance score</span><strong>{distance(item.distance)}</strong></div><div className="distance-track"><span style={{width: (Math.max(3, 100 * (1 - item.distance / (Math.max.apply(null, results.results.map((r) => r.distance)) * 1.15)))) + "%"}}/></div></div>
              </button>)}</div>
              <div className="retrieval-disclaimer"><ShieldCheck size={16}/><span>Similarity retrieval only. Folder labels describe the dataset, not a diagnosis of the uploaded image.</span></div>
            </section> : <div className="empty-state"><div className="empty-art"><ImageIcon size={27}/><Search size={19}/></div><strong>Your retrieved images will appear here</strong><span>Choose a query image, select descriptors and run a similarity search to rank the nearest images in the local BUSI collection.</span></div>}
          {selected && <div className="detail-drawer"><div className="drawer-head"><div><span className="section-index">SELECTED MATCH</span><h3>Rank #{selected.rank} · {selected.filename}</h3></div><button className="icon-button" onClick={() => setSelected(null)} aria-label="Close selected result"><X size={17}/></button></div><div className="drawer-content"><img src={selected.preview} alt="Selected retrieved image"/><div><span className="summary-label">DATASET FOLDER</span><strong className={"class-pill " + (CLASSES[selected.class] || CLASSES.unknown).cls}>{(CLASSES[selected.class] || CLASSES.unknown).label}</strong><span className="summary-label">DISTANCE SCORE</span><strong className="drawer-score">{distance(selected.distance)}</strong><p>Lower distance means the descriptors are closer under the selected metric. This score is not a probability.</p></div></div></div>}
        </>}

        {page === "dataset" && <section className="subpage">
          <div className="section-title-row"><div><span className="section-index">02</span><h2>Dataset overview</h2></div><span className="section-note">LOCAL DATA / BUSI</span></div>
          <div className="metric-cards"><div className="metric-card"><Database size={18}/><span>USABLE IMAGES</span><strong>{count(dataset && dataset.total_images)}</strong><small>Mask files excluded</small></div><div className="metric-card"><Layers3 size={18}/><span>DATASET CLASSES</span><strong>03</strong><small>Benign · Malignant · Normal</small></div><div className="metric-card"><ImageIcon size={18}/><span>PREPROCESSING</span><strong>128 × 128</strong><small>Grayscale, anti-aliased resize</small></div></div>
          <section className="panel class-panel"><div className="panel-head"><div className="panel-title"><Database size={17}/><h3>Images by dataset folder</h3></div><span className="section-note">LIVE FILE COUNT</span></div>{classEntries.map(([name, n]) => <div className="class-count" key={name}><div className="class-count-label"><span className={"class-dot " + ((CLASSES[name] || CLASSES.unknown).cls)}/><strong>{(CLASSES[name] || CLASSES.unknown).label}</strong><span>{count(n)} images</span></div><div className="class-bar"><span className={(CLASSES[name] || CLASSES.unknown).cls} style={{width: (dataset && dataset.total_images ? (n / dataset.total_images) * 100 : 0) + "%"}}/></div></div>)}<p className="field-hint">Counts are read from local dataset folders. Segmentation masks are excluded. Dataset download is not automatic.</p></section>
          <div className="notice notice-info"><Info size={18}/><div><strong>Dataset location</strong><code>data/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/</code><span>Expected folders: benign, malignant and normal. Dataset use remains subject to the original source terms.</span></div></div>
        </section>}

        {page === "methods" && <section className="subpage">
          <div className="section-title-row"><div><span className="section-index">03</span><h2>Methods & metrics</h2></div><span className="section-note">CLASSICAL FEATURE ENGINEERING</span></div>
          <div className="method-cards">{METHODS.map((m, i) => <article className="method-card" key={m.id}><span className="method-number">0{i + 1}</span><div className="method-icon">{i === 0 ? <Activity/> : i === 1 ? <FlaskConical/> : <Layers3/>}</div><h3>{m.label}</h3><p>{m.detail}</p><span className="method-foot">{i === 0 ? "Distribution" : i === 1 ? "Local texture" : "Shape descriptor"}</span></article>)}</div>
          <div className="section-title-row subsection-heading"><div><span className="section-index">04</span><h2>Saved evaluation protocol</h2></div></div>
          <div className="metric-cards"><div className="metric-card"><span>PRECISION@1</span><strong>{saved && saved.precision_at_1 != null ? (saved.precision_at_1 * 100).toFixed(2) + "%" : "—"}</strong><small>{saved ? "Read from saved report" : "No saved metric found"}</small></div><div className="metric-card"><span>PRECISION@5</span><strong>{saved && saved.precision_at_5 != null ? (saved.precision_at_5 * 100).toFixed(2) + "%" : "—"}</strong><small>{saved ? "Read from saved report" : "No saved metric found"}</small></div><div className="metric-card"><span>RELEVANCE RULE</span><strong>Same class</strong><small>Dataset folder label matches query</small></div></div>
          <section className="panel protocol-panel"><h3>How to interpret these values</h3><p><strong>Precision@1</strong> measures whether the nearest retrieved image shares the query's BUSI folder class. <strong>Precision@5</strong> is the fraction of the five nearest images with the same folder class. These are retrieval-consistency metrics, not diagnostic accuracy or model confidence.</p><p>Live distance values are descriptor distances, not probabilities. Saved evaluation values are shown only when a report exists.</p>{saved && <details><summary>View saved evaluation summary</summary><pre>{saved.text}</pre></details>}</section>
        </section>}

        <footer className="page-footer"><div><span className="footer-mark">RA</span><span>RESEARCH ARCHIVE <i>/</i> MEDICAL CBIR</span></div><span>Classical image retrieval · Reproducible research</span><span className="footer-version">LOCAL BUILD / 1.0</span></footer>
      </main>
    </div>
  );
}
