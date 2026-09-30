import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api.js";
import { batchLabel, fmtAuc } from "../format.js";
import useLoad from "../useLoad.js";

export default function OverviewTab({ row, thresholds, refreshKey }) {
  const { data } = useLoad(() => api.history(row.disease), [row.disease, refreshKey]);
  const checks = (data || []).filter((r) => r.status === "checked").reverse();
  const bars = checks.slice(-15).map((r) => ({ name: batchLabel(r.batch_name).replace("batch ", "#"), share: Math.round(r.drift_share * 100), drifted: r.drift_detected }));
  const latest = checks[checks.length - 1];
  const limit = Math.round((thresholds?.drift_share_threshold ?? 0.3) * 100);

  return (
    <div className="grid-2">
      <section className="panel">
        <h3>Feature drift per batch</h3>
        <p className="panel-sub">Share of features that shifted versus the live model's training data. Above {limit}% counts as drift.</p>
        {bars.length ? (
          <div style={{ height: 230 }}>
            <ResponsiveContainer>
              <BarChart data={bars} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="#EEF1F5" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#98A2B3" }} tickLine={false} axisLine={false} />
                <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11, fill: "#98A2B3" }} tickLine={false} axisLine={false} />
                <Tooltip formatter={(v) => [`${v}%`, "Features shifted"]} cursor={{ fill: "#F2F4F7" }} />
                <ReferenceLine y={limit} stroke="#F79009" strokeDasharray="5 4" label={{ value: "limit", position: "right", fontSize: 11, fill: "#B54708" }} />
                <Bar dataKey="share" radius={[6, 6, 0, 0]} maxBarSize={34}>
                  {bars.map((b, i) => <Cell key={i} fill={b.drifted ? "#F79009" : "#99D5CF"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="empty">No batches checked yet.</p>
        )}
      </section>

      <section className="panel">
        <h3>Latest check</h3>
        {latest ? (
          <>
            <dl className="facts">
              <div><dt>Batch</dt><dd>{batchLabel(latest.batch_name)} ({latest.rows} patients)</dd></div>
              <div><dt>Model checked</dt><dd>version {latest.model_version}</dd></div>
              <div><dt>Batch AUC</dt><dd>{latest.labelled === false ? <span className="muted">not measured (no outcomes)</span> : <>{fmtAuc(latest.batch_auc)} <span className="muted">baseline {fmtAuc(latest.baseline_auc)}</span></>}</dd></div>
              <div><dt>Result</dt><dd>{latest.reasons.length ? latest.reasons.map((r) => r.replace("_", " ").toLowerCase()).join(", ") : "within limits"}</dd></div>
            </dl>
            {latest.note && <p className="note">{latest.note}</p>}
            <h4 className="subhead">Shifted features</h4>
            <div className="chips">
              {latest.drifted_columns.length
                ? latest.drifted_columns.map((c) => <span key={c} className="chip chip-warn">{c}</span>)
                : <span className="muted">None</span>}
            </div>
          </>
        ) : (
          <p className="empty">Check a batch to see results here.</p>
        )}
      </section>
    </div>
  );
}
