import { useRef, useState } from "react";
import { Download, FileUp, UploadCloud } from "lucide-react";
import { api } from "../api.js";
import { batchLabel } from "../format.js";

const MAX_BYTES = 5_000_000;

export default function UploadPanel({ row, notify, onChanged }) {
  const input = useRef(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [message, setMessage] = useState(null); // { tone, text, warnings }

  const upload = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setMessage({ tone: "bad", text: "Please choose a .csv file. In Excel: File → Save As → CSV (comma delimited)." });
      return;
    }
    if (file.size > MAX_BYTES) {
      setMessage({ tone: "bad", text: "The file is larger than 5 MB. Split it into smaller batches." });
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const meta = await api.uploadBatch(row.disease, file.name, await file.text());
      const kind = meta.labelled ? "with outcomes" : "without outcomes (drift check only)";
      setMessage({ tone: "ok", text: `${file.name} was added as ${batchLabel(meta.batch_name)}: ${meta.rows} patients ${kind}.`, warnings: meta.warnings });
      notify(`${row.display_name}: ${batchLabel(meta.batch_name)} uploaded. Press "Check" on the card to monitor it.`, "ok");
      onChanged();
    } catch (e) {
      setMessage({ tone: "bad", text: e.message });
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    upload(e.dataTransfer.files?.[0]);
  };

  return (
    <section className="panel">
      <h3>Upload a batch file</h3>
      <p className="panel-sub">
        A CSV with the {row.display_name.toLowerCase()} columns. Include an <code>Outcome</code> column (1 = disease, 0 = no disease)
        to measure accuracy; without it the batch gets a drift check only.
      </p>
      <div className={`dropzone ${dragging ? "is-dragging" : ""}`}
           onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
           onDragLeave={() => setDragging(false)} onDrop={onDrop}>
        <UploadCloud size={28} aria-hidden="true" />
        <p>Drag a CSV file here, or</p>
        <button type="button" className="btn btn-primary" onClick={() => input.current?.click()} disabled={busy}>
          <FileUp size={16} /> {busy ? "Uploading…" : "Choose file"}
        </button>
        <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => upload(e.target.files?.[0])} />
      </div>
      {message && (
        <div className={`upload-msg upload-${message.tone}`} role="status">
          <p>{message.text}</p>
          {message.warnings?.length > 0 && <ul>{message.warnings.map((w) => <li key={w}>{w}</li>)}</ul>}
        </div>
      )}
      <div className="row-gap upload-links">
        <a className="btn btn-ghost" href={`/api/diseases/${row.disease}/template`} download>
          <Download size={16} /> Template
        </a>
        <a className="btn btn-ghost" href={`/api/samples/${row.disease}_clean.csv`} download>Sample: clean</a>
        <a className="btn btn-ghost" href={`/api/samples/${row.disease}_drifted.csv`} download>Sample: drifted</a>
        <a className="btn btn-ghost" href={`/api/samples/${row.disease}_unlabelled.csv`} download>Sample: no outcomes</a>
      </div>
    </section>
  );
}
