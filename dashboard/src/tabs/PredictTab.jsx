import { useEffect, useState } from "react";
import { api } from "../api.js";
import { fmtTime } from "../format.js";
import useLoad from "../useLoad.js";

const LABELS = {
  age: "Age", sex: "Sex", cp: "Chest pain type", trestbps: "Resting blood pressure", chol: "Cholesterol",
  fbs: "Fasting blood sugar > 120", restecg: "Resting ECG", thalach: "Max heart rate", exang: "Exercise angina",
  oldpeak: "ST depression", slope: "ST slope", ca: "Major vessels (0–3)", thal: "Thalassemia",
  DiabetesPedigreeFunction: "Diabetes pedigree", BloodPressure: "Blood pressure", SkinThickness: "Skin thickness",
};
const pretty = (name) => LABELS[name] ||
  name.replace(/_/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase().replace(/^./, (c) => c.toUpperCase()).replace(/\bbmi\b/i, "BMI");

export default function PredictTab({ disease, notify }) {
  const { data: schema, error } = useLoad(() => api.schema(disease), [disease]);
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [sending, setSending] = useState(false);
  const [recentKey, setRecentKey] = useState(0);
  const recent = useLoad(() => api.predictions(disease), [disease, recentKey]);
  const reset = () => {
    if (schema) setValues(Object.fromEntries(Object.entries(schema.example).map(([k, v]) => [k, String(v)])));
  };
  useEffect(() => {
    reset();
  }, [schema]); // eslint-disable-line react-hooks/exhaustive-deps

  if (error) return <p className="empty">Could not load the form: {error}</p>;
  if (!schema) return <p className="empty">Loading…</p>;

  const submit = async (e) => {
    e.preventDefault();
    setSending(true);
    try {
      setResult(await api.predict(disease, Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Number(v)]))));
      setRecentKey((k) => k + 1);
    } catch (err) {
      notify(err.message, "bad");
    } finally {
      setSending(false);
    }
  };
  const pct = result ? Math.round(result.probability * 100) : 0;

  return (
    <div className="grid-2 grid-wide-left">
      <section className="panel">
        <h3>Patient values</h3>
        <form onSubmit={submit}>
          <div className="fields">
            {Object.entries(schema.fields).map(([name, f]) => (
              <label key={name} className="field">
                <span className="field-name">{pretty(name)}</span>
                <input type="number" step="any" required min={f.minimum} max={f.maximum} value={values[name] ?? ""}
                       onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))} />
                {f.description && <span className="field-hint">{f.description}</span>}
              </label>
            ))}
          </div>
          <div className="row-gap form-actions">
            <button className="btn btn-primary" disabled={sending}>{sending ? "Predicting…" : "Run prediction"}</button>
            <button type="button" className="btn btn-ghost" onClick={reset}>Reset to example patient</button>
          </div>
        </form>
      </section>

      <section className="panel">
        <h3>Result</h3>
        {result ? (
          <div className="result" aria-live="polite">
            <div className="gauge" style={{ "--pct": pct, "--gauge": result.prediction ? "#F04438" : "#12B76A" }}>
              <div className="gauge-inner">
                <strong>{pct}%</strong>
                <span>risk</span>
              </div>
            </div>
            <p className={`result-label ${result.prediction ? "is-pos" : "is-neg"}`}>{result.prediction_label}</p>
            <p className="muted small">Scored by version {result.model_version} ({result.model_algorithm}); threshold 50%. A portfolio model, not a clinical tool.</p>
          </div>
        ) : (
          <p className="empty">Adjust the values and run a prediction. Every prediction is logged.</p>
        )}
        {recent.data?.length > 0 && (
          <>
            <h4 className="subhead">Recent predictions</h4>
            <ul className="mini-list">
              {recent.data.slice(0, 6).map((p, i) => (
                <li key={i}><span>{fmtTime(p.timestamp)}</span><span className="num">{Math.round(Number(p.probability) * 100)}%</span><span className="muted">v{p.model_version}</span></li>
              ))}
            </ul>
          </>
        )}
      </section>
    </div>
  );
}
