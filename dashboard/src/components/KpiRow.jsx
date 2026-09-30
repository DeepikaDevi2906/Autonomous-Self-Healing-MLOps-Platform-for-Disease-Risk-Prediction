import { ShieldCheck, Layers, GitPullRequestArrow, Stethoscope } from "lucide-react";

function Kpi({ icon: Icon, tone, label, value, sub }) {
  return (
    <div className="card kpi">
      <span className={`kpi-icon tone-${tone}`} aria-hidden="true">
        <Icon size={20} />
      </span>
      <div>
        <p className="kpi-label">{label}</p>
        <p className="kpi-value">{value}</p>
        <p className="kpi-sub">{sub}</p>
      </div>
    </div>
  );
}

export default function KpiRow({ overview, retrains }) {
  const d = overview.diseases;
  const healthy = d.filter((x) => x.status === "healthy").length;
  const checks = d.reduce((n, x) => n + x.trend.length, 0);
  const batches = d.reduce((n, x) => n + x.batches, 0);
  const pending = d.reduce((n, x) => n + x.pending_batches.length, 0);
  const promoted = retrains.filter((r) => r.should_promote).length;
  const predictions = d.reduce((n, x) => n + x.predictions, 0);
  return (
    <section className="kpis" aria-label="Platform summary">
      <Kpi icon={ShieldCheck} tone={healthy === d.length ? "ok" : "warn"} label="Healthy models"
           value={`${healthy} / ${d.length}`} sub={healthy === d.length ? "All models within limits" : "Some models need attention"} />
      <Kpi icon={Layers} tone="info" label="Batches checked" value={checks}
           sub={pending ? `${pending} waiting to be checked` : `${batches} batches received`} />
      <Kpi icon={GitPullRequestArrow} tone="primary" label="Retraining runs" value={retrains.length}
           sub={`${promoted} promoted, ${retrains.length - promoted} kept out`} />
      <Kpi icon={Stethoscope} tone="violet" label="Predictions served" value={predictions} sub="Across all three models" />
    </section>
  );
}
