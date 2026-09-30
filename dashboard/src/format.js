export const fmtAuc = (v) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : Number(v).toFixed(3));

export const fmtPct = (v) => (v === null || v === undefined ? "–" : `${Math.round(Number(v) * 100)}%`);

export function fmtTime(value) {
  if (!value) return "–";
  const date = typeof value === "number" ? new Date(value) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function batchLabel(name) {
  if (!name) return "";
  const n = Number(String(name).replace("batch_", ""));
  return Number.isFinite(n) ? `batch ${n}` : name;
}

export const STATUS = {
  healthy: { word: "Healthy", tone: "ok" },
  drift: { word: "Data drift", tone: "warn" },
  degraded: { word: "Performance drop", tone: "bad" },
  retraining: { word: "Retraining", tone: "busy" },
  no_model: { word: "No model", tone: "idle" },
};

export function statusDetail(row) {
  const c = row.latest_check;
  if (row.status === "retraining") return "Training a challenger on the newest data";
  if (row.status === "no_model") return "Run initial setup to train a model";
  if (!c) return "Waiting for the first batch check";
  if (row.champion && c.model_version && c.model_version !== row.champion.version)
    return `Version ${row.champion.version} went live after ${batchLabel(c.batch_name)}; waiting for the next batch`;
  if (c.status !== "checked") return `${batchLabel(c.batch_name)} skipped (${c.status.replace("_", " ")})`;
  const pause = c.cooldown_remaining ? `, retraining paused for ${c.cooldown_remaining} more batch${c.cooldown_remaining > 1 ? "es" : ""}` : "";
  if (c.labelled === false && row.status !== "drift")
    return `${batchLabel(c.batch_name)} within limits (drift check only, no outcomes)`;
  if (row.status === "degraded")
    return `AUC ${fmtAuc(c.batch_auc)} on ${batchLabel(c.batch_name)}, baseline ${fmtAuc(c.baseline_auc)}${pause}`;
  if (row.status === "drift")
    return `${fmtPct(c.drift_share)} of features shifted in ${batchLabel(c.batch_name)}${pause}`;
  return `${batchLabel(c.batch_name)} within limits`;
}

const sentence = (t) => (t ? t.charAt(0).toUpperCase() + t.slice(1) + "." : "");

export function describeAlert(a) {
  const d = a.details || {};
  switch (a.reason) {
    case "DATA_DRIFT":
      return `Drift in ${batchLabel(d.batch_name)}: ${d.drifted_count} of ${d.total_columns} features shifted.`;
    case "PERFORMANCE_DROP":
      return `AUC fell to ${fmtAuc(d.batch_auc)} on ${batchLabel(d.batch_name)} against a baseline of ${fmtAuc(d.baseline_auc)}.`;
    case "MODEL_PROMOTED":
      return `Version ${d.version} (${d.algorithm}) went live, replacing version ${d.previous_version ?? "none"}. ${sentence(d.decision)}`;
    case "CHALLENGER_REJECTED":
      return `Challenger version ${d.challenger_version} (${d.algorithm}) was not promoted. ${sentence(d.decision)}`;
    case "ROLLBACK":
      return `Rolled back from version ${d.from_version} to version ${d.to_version}.`;
    default:
      return a.reason;
  }
}

export const ALERT_TITLE = {
  DATA_DRIFT: "Data drift",
  PERFORMANCE_DROP: "Performance drop",
  MODEL_PROMOTED: "New model live",
  CHALLENGER_REJECTED: "Challenger kept out",
  ROLLBACK: "Rollback",
};

export const DISEASE_META = {
  diabetes: { color: "#2E90FA", soft: "#EFF8FF" },
  heart_disease: { color: "#E31B54", soft: "#FFF1F3" },
  breast_cancer: { color: "#7A5AF8", soft: "#F4F3FF" },
};

export const shortTime = (v) =>
  v ? new Date(v).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "–";
