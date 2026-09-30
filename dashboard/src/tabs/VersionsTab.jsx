import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { RotateCcw } from "lucide-react";
import { api } from "../api.js";
import { fmtAuc, fmtTime } from "../format.js";
import useLoad from "../useLoad.js";

function statusOf(v) {
  if (v.is_champion) return ["Live", "ok"];
  if (v.tags.status === "rejected") return ["Rejected", "neutral"];
  if (v.tags.rolled_back_at) return ["Rolled back", "warn"];
  if (v.tags.promoted_at) return ["Retired", "info"];
  return ["Candidate", "neutral"];
}

export default function VersionsTab({ row, refreshKey, notify, onChanged }) {
  const [confirming, setConfirming] = useState(false);
  const [working, setWorking] = useState(false);
  const { data, error } = useLoad(() => api.versions(row.disease), [row.disease, refreshKey]);
  if (error) return <p className="empty">Could not load versions: {error}</p>;
  if (!data) return <p className="empty">Loading…</p>;

  const history = data.champion_history || [];
  const previous = history.length > 1 ? history[history.length - 2] : null;
  const chart = [...data.versions].reverse().map((v) => ({ name: `v${v.version}`, auc: Number(v.tags.test_auc) || 0, live: v.is_champion, rejected: v.tags.status === "rejected" }));

  const rollback = async () => {
    setWorking(true);
    try {
      const r = await api.rollback(row.disease);
      notify(r.message, r.restored_version ? "ok" : "info");
      onChanged();
    } catch (e) {
      notify(e.message, "bad");
    } finally {
      setWorking(false);
      setConfirming(false);
    }
  };

  return (
    <div className="grid-2 grid-wide-left">
      <section className="panel">
        <h3>Registered versions</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Version</th><th>Algorithm</th><th>Trained by</th><th className="num">Test AUC</th><th className="num">Recent AUC</th><th>Status</th><th>Registered</th></tr>
            </thead>
            <tbody>
              {data.versions.map((v) => {
                const [label, tone] = statusOf(v);
                return (
                  <tr key={v.version} className={v.is_champion ? "is-live" : ""}>
                    <td>v{v.version}</td>
                    <td>{v.tags.algorithm}</td>
                    <td>{v.tags.training_mode === "initial" ? "Initial training" : "Retraining"}</td>
                    <td className="num">{fmtAuc(v.tags.test_auc)}</td>
                    <td className="num">{fmtAuc(v.tags.recent_auc)}</td>
                    <td><span className={`badge badge-${tone}`}>{label}</span></td>
                    <td className="muted">{fmtTime(v.created_at)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <h3>Test AUC by version</h3>
        <div style={{ height: 180 }}>
          <ResponsiveContainer>
            <BarChart data={chart} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid stroke="#EEF1F5" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#98A2B3" }} tickLine={false} axisLine={false} />
              <YAxis domain={[0.5, 1]} tick={{ fontSize: 11, fill: "#98A2B3" }} tickLine={false} axisLine={false} />
              <Tooltip formatter={(v) => [fmtAuc(v), "Test AUC"]} cursor={{ fill: "#F2F4F7" }} />
              <Bar dataKey="auc" radius={[6, 6, 0, 0]} maxBarSize={36}>
                {chart.map((c, i) => <Cell key={i} fill={c.live ? "#0F766E" : c.rejected ? "#D0D5DD" : "#99D5CF"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="rollback">
          {!previous && <p className="muted small">Rollback becomes available after a second version has gone live.</p>}
          {previous && !confirming && (
            <button className="btn btn-ghost" onClick={() => setConfirming(true)} disabled={row.status === "retraining"}>
              <RotateCcw size={16} /> Roll back to v{previous}
            </button>
          )}
          {previous && confirming && (
            <div className="confirm">
              <p>Version {previous} will serve predictions again, replacing version {row.champion?.version}.</p>
              <div className="row-gap">
                <button className="btn btn-danger" onClick={rollback} disabled={working}>{working ? "Rolling back…" : `Roll back to v${previous}`}</button>
                <button className="btn btn-ghost" onClick={() => setConfirming(false)}>Keep v{row.champion?.version}</button>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
