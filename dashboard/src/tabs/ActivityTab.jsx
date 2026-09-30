import { AlertTriangle, ArrowDownRight, CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { api } from "../api.js";
import { ALERT_TITLE, describeAlert, fmtTime } from "../format.js";
import useLoad from "../useLoad.js";

const ICON = { DATA_DRIFT: AlertTriangle, PERFORMANCE_DROP: ArrowDownRight, MODEL_PROMOTED: CheckCircle2, CHALLENGER_REJECTED: XCircle, ROLLBACK: RotateCcw };
const TONE = { DATA_DRIFT: "warn", PERFORMANCE_DROP: "bad", MODEL_PROMOTED: "ok", CHALLENGER_REJECTED: "neutral", ROLLBACK: "info" };

export default function ActivityTab({ disease, refreshKey }) {
  const { data, error } = useLoad(() => api.alerts(disease), [disease, refreshKey]);
  if (error) return <p className="empty">Could not load activity: {error}</p>;
  if (!data) return <p className="empty">Loading…</p>;
  if (!data.length) return <p className="empty">No events yet. Create a batch in "Simulate data", then check it.</p>;
  return (
    <ol className="timeline">
      {data.map((a, i) => {
        const Icon = ICON[a.reason] || AlertTriangle;
        return (
          <li key={`${a.timestamp}-${i}`} className="event">
            <span className={`event-icon tone-${TONE[a.reason] || "neutral"}`} aria-hidden="true"><Icon size={16} /></span>
            <div className="event-body">
              <div className="event-head">
                <h4>{ALERT_TITLE[a.reason] || a.reason}</h4>
                <time dateTime={a.timestamp}>{fmtTime(a.timestamp)}</time>
              </div>
              <p>{describeAlert(a)}</p>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
