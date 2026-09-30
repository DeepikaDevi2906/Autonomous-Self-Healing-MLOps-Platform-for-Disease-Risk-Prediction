import { useId, useState } from "react";
import OverviewTab from "../tabs/OverviewTab.jsx";
import ActivityTab from "../tabs/ActivityTab.jsx";
import VersionsTab from "../tabs/VersionsTab.jsx";
import PredictTab from "../tabs/PredictTab.jsx";
import SimulateTab from "../tabs/SimulateTab.jsx";

const TABS = [["overview", "Overview"], ["activity", "Activity"], ["versions", "Model versions"], ["predict", "Predict"], ["simulate", "Simulate data"]];

export default function DetailPanel({ row, thresholds, refreshKey, notify, onChanged }) {
  const [tab, setTab] = useState("overview");
  const base = useId();
  const onKey = (e) => {
    const i = TABS.findIndex(([k]) => k === tab);
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    const [k] = TABS[(i + d + TABS.length) % TABS.length];
    setTab(k);
    document.getElementById(`${base}-${k}`)?.focus();
  };
  return (
    <section className="card detail" aria-label={`${row.display_name} details`}>
      <div className="detail-head">
        <h2>{row.display_name}</h2>
        <div className="tabs" role="tablist" onKeyDown={onKey}>
          {TABS.map(([k, label]) => (
            <button key={k} id={`${base}-${k}`} role="tab" aria-selected={tab === k} tabIndex={tab === k ? 0 : -1}
                    className="tab" onClick={() => setTab(k)}>{label}</button>
          ))}
        </div>
      </div>
      <div role="tabpanel" className="detail-body">
        {tab === "overview" && <OverviewTab row={row} thresholds={thresholds} refreshKey={refreshKey} />}
        {tab === "activity" && <ActivityTab disease={row.disease} refreshKey={refreshKey} />}
        {tab === "versions" && <VersionsTab row={row} refreshKey={refreshKey} notify={notify} onChanged={onChanged} />}
        {tab === "predict" && <PredictTab disease={row.disease} notify={notify} />}
        {tab === "simulate" && <SimulateTab row={row} notify={notify} onChanged={onChanged} refreshKey={refreshKey} />}
      </div>
    </section>
  );
}
