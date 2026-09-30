import { Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { batchLabel, fmtAuc } from "../format.js";

function DriftDot({ cx, cy, payload }) {
  if (cx == null || cy == null) return null;
  if (payload.drift_detected) {
    return <path d={`M${cx - 6},${cy + 5} L${cx},${cy - 6} L${cx + 6},${cy + 5} Z`} fill="#F79009" stroke="#fff" strokeWidth={1.5} />;
  }
  return <circle cx={cx} cy={cy} r={3.5} fill="#fff" stroke={payload.__color} strokeWidth={2} />;
}

function Tip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="chart-tip">
      <strong>{batchLabel(p.batch)}</strong>
      <span>Batch AUC {fmtAuc(p.batch_auc)}</span>
      <span>Baseline {fmtAuc(p.baseline_auc)}</span>
      {p.drift_detected && <span className="tip-warn">Drift detected</span>}
      {p.performance_dropped && <span className="tip-bad">Performance drop</span>}
    </div>
  );
}

export default function AucChart({ trend, color, id, height = 150 }) {
  const data = (trend || []).slice(-15).map((t) => ({ ...t, name: batchLabel(t.batch).replace("batch ", "#"), __color: color }));
  if (!data.length) return <div className="chart-empty" style={{ height }}>No batches checked yet</div>;
  const lows = data.map((d) => d.batch_auc ?? 1).concat(data.map((d) => d.baseline_auc ?? 1));
  const min = Math.max(0, Math.floor((Math.min(...lows) - 0.05) * 10) / 10);
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}>
          <defs>
            <linearGradient id={`g-${id}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.22} />
              <stop offset="100%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#EEF1F5" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#98A2B3" }} tickLine={false} axisLine={false} />
          <YAxis domain={[min, 1]} tick={{ fontSize: 11, fill: "#98A2B3" }} tickLine={false} axisLine={false} tickCount={4} tickFormatter={(v) => v.toFixed(2)} />
          <Tooltip content={<Tip />} cursor={{ stroke: "#D0D5DD" }} />
          <Line type="stepAfter" dataKey="baseline_auc" stroke="#667085" strokeDasharray="5 4" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Area type="monotone" dataKey="batch_auc" stroke={color} strokeWidth={2.4} fill={`url(#g-${id})`} dot={<DriftDot />} activeDot={{ r: 5 }} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
