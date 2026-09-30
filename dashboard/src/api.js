// All requests are same-origin: the Vite dev proxy (local) or nginx (Docker)
// forwards /api, /health and /predict to the FastAPI service.
const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, kind = "error") {
    super(message);
    this.kind = kind; // "unreachable" | "wrong-service" | "error"
  }
}

async function request(method, path, body) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError("The API is not responding.", "unreachable");
  }
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new ApiError(`Unexpected response from ${path} (${response.status}).`, "wrong-service");
  }
  if (response.status === 502 || response.status === 504) throw new ApiError("The API is starting or not running.", "unreachable");
  if (!response.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ")
      : detail || `Request failed (${response.status})`;
    throw new ApiError(message);
  }
  return data;
}

export const api = {
  // Confirms the service on the other end really is this platform's API.
  async identify() {
    let info;
    try {
      info = await request("GET", "/api/info");
    } catch (e) {
      if (e.kind === "unreachable") throw e;
      throw new ApiError("A different service is answering on the API port.", "wrong-service");
    }
    if (info?.platform !== "self-healing-mlops") {
      throw new ApiError("A different service is answering on the API port.", "wrong-service");
    }
    return info;
  },
  overview: () => request("GET", "/api/overview"),
  schema: (d) => request("GET", `/api/diseases/${d}/schema`),
  versions: (d) => request("GET", `/api/models/${d}/versions`),
  rollback: (d) => request("POST", `/api/models/${d}/rollback`),
  history: (d) => request("GET", `/api/monitoring/history?disease=${d}&limit=60`),
  retrainingHistory: () => request("GET", "/api/retraining/history?limit=200"),
  alerts: (d) => request("GET", `/api/alerts?disease=${d}&limit=40`),
  batches: (d) => request("GET", `/api/batches/${d}`),
  createBatch: (d, body) => request("POST", `/api/batches/${d}`, body),
  uploadBatch: (d, filename, csv) => request("POST", `/api/batches/${d}/upload`, { filename, csv }),
  sampleUrl: (name) => `${BASE}/api/samples/${name}`,
  runCheck: (d, force = false) => request("POST", `/api/monitoring/${d}/run?heal=true&force=${force}`),
  predict: (d, body) => request("POST", `/predict/${d}`, body),
  predictions: (d) => request("GET", `/api/predictions/${d}?limit=12`),
  setup: () => request("POST", "/api/setup"),
  job: (id) => request("GET", `/api/jobs/${id}`),
};
