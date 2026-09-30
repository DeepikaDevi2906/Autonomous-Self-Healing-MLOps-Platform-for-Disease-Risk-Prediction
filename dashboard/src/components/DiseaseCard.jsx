import { Droplet, HeartPulse, Microscope, ScanSearch, ChevronRight } from "lucide-react";
import AucChart from "./AucChart.jsx";
import { DISEASE_META, STATUS, fmtAuc, statusDetail } from "../format.js";

const ICONS = { diabetes: Droplet, heart_disease: HeartPulse, breast_cancer: Microscope };

export default function DiseaseCard({ row, selected, onSelect, onCheck, checking }) {
  const meta = DISEASE_META[row.disease];
  const status = STATUS[row.status] || STATUS.healthy;
  const Icon = ICONS[row.disease] || Droplet;
  const c = row.champion;
  const pending = row.pending_batches?.length || 0;
  return (
    <article className={`card disease ${selected ? "is-selected" : ""}`} style={{ "--accent": meta.color, "--accent-soft": meta.soft }}>
      <div className="disease-head">
        <span className="disease-icon" aria-hidden="true"><Icon size={20} /></span>
        <div className="disease-title">
          <h2>{row.display_name}</h2>
          <p>{c ? `${c.algorithm} · version ${c.version}` : "No model trained"}</p>
        </div>
        <span className={`badge badge-${status.tone}`}>{status.word}</span>
      </div>

      <div className="disease-metrics">
        <div>
          <p className="metric-label">Test AUC</p>
          <p className="metric-value">{fmtAuc(c?.test_auc)}</p>
        </div>
        <div>
          <p className="metric-label">Latest batch AUC</p>
          <p className="metric-value metric-sm">{fmtAuc(row.latest_check?.batch_auc)}</p>
        </div>
        <div>
          <p className="metric-label">Drift share</p>
          <p className="metric-value metric-sm">{row.latest_check ? `${Math.round(row.latest_check.drift_share * 100)}%` : "–"}</p>
        </div>
      </div>

      <AucChart trend={row.trend} color={meta.color} id={row.disease} />
      <p className="status-line">{statusDetail(row)}</p>

      <div className="disease-actions">
        <button className="btn btn-primary" onClick={onCheck} disabled={checking || row.status === "retraining" || !c}>
          <ScanSearch size={16} />
          {checking ? "Checking…" : pending ? `Check ${pending} new batch${pending > 1 ? "es" : ""}` : "Re-check latest"}
        </button>
        <button className="btn btn-ghost" onClick={onSelect} aria-pressed={selected}>
          Details <ChevronRight size={16} />
        </button>
      </div>
    </article>
  );
}
