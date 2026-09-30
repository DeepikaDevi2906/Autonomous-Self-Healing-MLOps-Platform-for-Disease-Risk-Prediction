import { useState } from "react";
import { Plus } from "lucide-react";
import UploadPanel from "./UploadPanel.jsx";
import { api } from "../api.js";
import { batchLabel, fmtTime } from "../format.js";
import useLoad from "../useLoad.js";

const CONTROLS = [
  { key: "drift", label: "Data drift", min: 0, max: 3, step: 0.25, show: (v) => (v ? `${v} SD` : "None"),
    help: "Shifts key measurements, like a new patient population." },
  { key: "concept_shift", label: "Concept shift", min: 0, max: 1, step: 0.1, show: (v) => (v ? `${Math.round(v * 100)}%` : "None"),
    help: "Changes how measurements relate to the outcome. Retraining can learn this." },
  { key: "label_noise", label: "Label noise", min: 0, max: 0.5, step: 0.05, show: (v) => (v ? `${Math.round(v * 100)}%` : "None"),
    help: "Random wrong outcomes. No model can learn noise, so nothing should be promoted." },
];

const MAX_BYTES = 5_000_000;

function describe(b) {
  if (b.source === "upload") return `uploaded: ${b.filename}${b.labelled === false ? " (no outcomes)" : ""}`;
  return [b.simulated_drift && `drift ${b.simulated_drift} SD`, b.simulated_concept_shift && `concept ${Math.round(b.simulated_concept_shift * 100)}%`,
          b.simulated_label_noise && `noise ${Math.round(b.simulated_label_noise * 100)}%`].filter(Boolean).join(", ") || "clean";
}

export default function SimulateTab({ row, notify, onChanged, refreshKey }) {
  const [settings, setSettings] = useState({ drift: 0, concept_shift: 0, label_noise: 0 });
  const [creating, setCreating] = useState(false);
  const batches = useLoad(() => api.batches(row.disease), [row.disease, refreshKey]);

  const create = async () => {
    setCreating(true);
    try {
      const meta = await api.createBatch(row.disease, settings);
      notify(`${batchLabel(meta.batch_name)} created with ${meta.rows} patients. Press "Check" on the ${row.display_name} card.`, "ok");
      onChanged();
    } catch (e) {
      notify(e.message, "bad");
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="stack">
      <div className="grid-2">
        <section className="panel">
          <h3>Simulate a batch</h3>
          <p className="panel-sub">Patients come from a held-out pool the models have never seen.</p>
          {CONTROLS.map((c) => (
            <label key={c.key} className="slider">
              <span className="slider-head"><span className="field-name">{c.label}</span><output>{c.show(settings[c.key])}</output></span>
              <input type="range" min={c.min} max={c.max} step={c.step} value={settings[c.key]}
                     onChange={(e) => setSettings((s) => ({ ...s, [c.key]: Number(e.target.value) }))} />
              <span className="field-hint">{c.help}</span>
            </label>
          ))}
          <button className="btn btn-primary" onClick={create} disabled={creating}><Plus size={16} />{creating ? "Creating…" : "Create batch"}</button>
        </section>

        <UploadPanel row={row} notify={notify} onChanged={onChanged} />
      </div>

      <section className="panel">
        <h3>Batches</h3>
        {!batches.data?.length ? <p className="empty">No batches yet.</p> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Batch</th><th className="num">Patients</th><th>Source</th><th>Status</th><th>Created</th></tr></thead>
              <tbody>
                {batches.data.slice(0, 10).map((b) => (
                  <tr key={b.batch_name}>
                    <td>{batchLabel(b.batch_name)}</td>
                    <td className="num">{b.rows ?? "?"}</td>
                    <td>{describe(b)}</td>
                    <td><span className={`badge ${b.pending ? "badge-warn" : "badge-neutral"}`}>{b.pending ? "Not checked" : "Checked"}</span></td>
                    <td className="muted">{fmtTime(b.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
