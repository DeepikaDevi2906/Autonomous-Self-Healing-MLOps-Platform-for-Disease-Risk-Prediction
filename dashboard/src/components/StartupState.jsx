import { PlugZap, ServerCrash, Sparkles } from "lucide-react";
import { useState } from "react";
import { api } from "../api.js";

export function ConnectionProblem({ error }) {
  const wrong = error?.kind === "wrong-service";
  return (
    <section className="card notice" role="alert">
      <span className="notice-icon tone-bad" aria-hidden="true">{wrong ? <ServerCrash size={22} /> : <PlugZap size={22} />}</span>
      <div>
        <h2>{wrong ? "A different program is answering on the API port" : "The dashboard can't reach the API yet"}</h2>
        {wrong ? (
          <>
            <p>Requests reached a server, but it is not this platform's API. This usually means an older container or another app is using port 8000.</p>
            <ol>
              <li>Run <code>docker ps</code> and stop old containers with <code>docker rm -f &lt;id&gt;</code>, or run <code>docker compose down</code> in the old project folder.</li>
              <li>On Windows, find what uses the port with <code>netstat -ano | findstr :8000</code>.</li>
              <li>Or change <code>API_PORT</code> in <code>.env</code> and restart.</li>
            </ol>
          </>
        ) : (
          <>
            <p>If you just ran <code>docker compose up</code>, the models are still training on first start. This page connects automatically when the API is ready.</p>
            <p className="muted">Running without Docker? Start the API from the <code>src</code> folder: <code>uvicorn inference.main:app --port 8000</code></p>
          </>
        )}
      </div>
    </section>
  );
}

export function SetupNeeded({ busy, onStarted, notify }) {
  const [starting, setStarting] = useState(false);
  const start = async () => {
    setStarting(true);
    try {
      onStarted(await api.setup());
    } catch (e) {
      notify(e.message, "bad");
    } finally {
      setStarting(false);
    }
  };
  return (
    <section className="card notice">
      <span className="notice-icon tone-primary" aria-hidden="true"><Sparkles size={22} /></span>
      <div>
        <h2>No models are trained yet</h2>
        <p>Setup validates the data, trains three algorithms for each disease, puts the best one live and creates a first batch to monitor. It takes about a minute.</p>
        <button className="btn btn-primary" onClick={start} disabled={starting || busy}>{busy ? "Setting up…" : "Run initial setup"}</button>
      </div>
    </section>
  );
}
