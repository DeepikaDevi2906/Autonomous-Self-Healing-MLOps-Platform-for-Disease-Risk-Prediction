import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import Header from "./components/Header.jsx";
import KpiRow from "./components/KpiRow.jsx";
import DiseaseCard from "./components/DiseaseCard.jsx";
import DetailPanel from "./components/DetailPanel.jsx";
import Toasts from "./components/Toasts.jsx";
import { ConnectionProblem, SetupNeeded } from "./components/StartupState.jsx";

export default function App() {
  const [overview, setOverview] = useState(null);
  const [retrains, setRetrains] = useState([]);
  const [error, setError] = useState(null);
  const [identified, setIdentified] = useState(false);
  const [selected, setSelected] = useState("diabetes");
  const [checking, setChecking] = useState({});
  const [toasts, setToasts] = useState([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [updatedAt, setUpdatedAt] = useState(null);
  const watched = useRef(new Set());

  const notify = useCallback((text, tone = "info") => {
    const id = Math.random().toString(36).slice(2);
    setToasts((t) => [...t, { id, text, tone }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 7000);
  }, []);

  const load = useCallback(async () => {
    try {
      if (!identified) {
        await api.identify();
        setIdentified(true);
      }
      const [ov, rh] = await Promise.all([api.overview(), api.retrainingHistory()]);
      setOverview(ov);
      setRetrains(rh);
      setError(null);
      setUpdatedAt(Date.now());
    } catch (e) {
      setError(e);
      if (e.kind) setIdentified(false);
    }
  }, [identified]);

  const refresh = useCallback(() => {
    load();
    setRefreshKey((k) => k + 1);
  }, [load]);

  useEffect(() => {
    load();
    const every = error ? 4000 : overview?.busy ? 2500 : 10000;
    const timer = setInterval(load, every);
    return () => clearInterval(timer);
  }, [load, overview?.busy, error]);

  const watchJob = useCallback((job, label) => {
    if (!job || watched.current.has(job.id)) return;
    watched.current.add(job.id);
    const poll = async () => {
      try {
        const cur = await api.job(job.id);
        if (cur.status === "succeeded") {
          const r = cur.result || {};
          if (cur.kind === "retraining") {
            notify(r.should_promote
              ? `${label}: version ${r.challenger_version} (${r.challenger_algorithm}) is now live.`
              : `${label}: the challenger was not better, so the live model was kept.`, r.should_promote ? "ok" : "info");
          } else notify(`${label} finished.`, "ok");
          return refresh();
        }
        if (cur.status === "failed") {
          notify(`${label} failed: ${(cur.error || "").split("\n")[0]}`, "bad");
          return refresh();
        }
        setTimeout(poll, 2000);
      } catch (e) {
        setTimeout(poll, 4000);
      }
    };
    poll();
  }, [notify, refresh]);

  useEffect(() => {
    overview?.diseases?.forEach((d) => d.active_job && watchJob(d.active_job, `Retraining ${d.display_name}`));
  }, [overview, watchJob]);

  const runCheck = async (row) => {
    setChecking((c) => ({ ...c, [row.disease]: true }));
    try {
      const { monitoring: m, retraining_job: job } = await api.runCheck(row.disease, !row.pending_batches?.length);
      if (m.status === "no_new_batch") notify(`${row.display_name}: no new batch to check. Create one in "Simulate data".`);
      else if (job) {
        notify(`${row.display_name}: ${m.reasons.map((r) => r.replace("_", " ").toLowerCase()).join(" and ")} found. Retraining started.`, "warn");
        watchJob(job, `Retraining ${row.display_name}`);
      } else if (m.note) notify(`${row.display_name}: ${m.note}`, "warn");
      else if (m.reasons?.length && m.cooldown_remaining)
        notify(`${row.display_name}: issues found; retraining paused for ${m.cooldown_remaining} more batch(es).`, "warn");
      else if (m.reasons?.length) notify(`${row.display_name}: issues found (${m.reasons.map((r) => r.replace("_", " ").toLowerCase()).join(", ")}).`, "warn");
      else notify(`${row.display_name}: the batch is within limits${m.labelled === false ? " (drift check only, no outcomes)" : ""}.`, "ok");
      refresh();
    } catch (e) {
      notify(e.message, "bad");
    } finally {
      setChecking((c) => ({ ...c, [row.disease]: false }));
    }
  };

  const state = error ? "down" : !overview ? "loading" : overview.busy ? "busy" : "live";
  const selectedRow = overview?.diseases?.find((d) => d.disease === selected);

  return (
    <>
      <Header state={state} updatedAt={updatedAt} onRefresh={refresh} />
      <main className="page">
        {error && (!overview || error.kind === "wrong-service") && <ConnectionProblem error={error} />}
        {overview?.setup_required && !error && <SetupNeeded busy={overview.busy} notify={notify} onStarted={(job) => watchJob(job, "Initial setup")} />}
        {overview && !overview.setup_required && (
          <>
            <KpiRow overview={overview} retrains={retrains} />
            <section className="disease-grid" aria-label="Disease models">
              {overview.diseases.map((row) => (
                <DiseaseCard key={row.disease} row={row} selected={row.disease === selected}
                             onSelect={() => setSelected(row.disease)} onCheck={() => runCheck(row)} checking={!!checking[row.disease]} />
              ))}
            </section>
            <p className="legend">
              <span><i className="lg-line" /> batch AUC of the live model</span>
              <span><i className="lg-dash" /> baseline (live model's test AUC)</span>
              <span><i className="lg-drift" /> drift detected</span>
            </p>
            {selectedRow && (
              <DetailPanel key={selectedRow.disease} row={selectedRow} thresholds={overview.thresholds}
                           refreshKey={refreshKey} notify={notify} onChanged={refresh} />
            )}
          </>
        )}
      </main>
      <Toasts toasts={toasts} />
    </>
  );
}
