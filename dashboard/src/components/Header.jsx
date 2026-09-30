import { Activity, RefreshCw } from "lucide-react";
import { shortTime } from "../format.js";

export default function Header({ state, updatedAt, onRefresh }) {
  const label = { live: "Connected", down: "Not connected", busy: "Job running", loading: "Connecting" }[state];
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <Activity size={20} strokeWidth={2.4} />
        </span>
        <div>
          <h1>Self-Healing MLOps</h1>
          <p>Disease risk models that monitor and repair themselves</p>
        </div>
      </div>
      <div className="topbar-right">
        <span className={`conn conn-${state}`} role="status">
          <span className="conn-dot" aria-hidden="true" />
          {label}
        </span>
        <span className="updated">Updated {shortTime(updatedAt)}</span>
        <button className="btn btn-ghost" onClick={onRefresh} aria-label="Refresh now">
          <RefreshCw size={16} /> <span className="hide-sm">Refresh</span>
        </button>
      </div>
    </header>
  );
}
