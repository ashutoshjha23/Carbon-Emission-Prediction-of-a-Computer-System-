const API_BASE = "http://localhost:5000/api";

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${res.status})`);
  }
  return res.json();
}

export function getStatus() {
  return apiFetch("/status");
}

export function collectData(runs, interval) {
  return apiFetch("/collect-data", {
    method: "POST",
    body: JSON.stringify({ runs, interval }),
  });
}

export async function uploadCsv(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/upload-csv`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Upload failed (${res.status})`);
  }
  return res.json();
}

export function preprocessData() {
  return apiFetch("/preprocess", { method: "POST" });
}

export function trainModel() {
  return apiFetch("/train", { method: "POST" });
}

export function predictEmissions() {
  return apiFetch("/predict", { method: "POST" });
}

export function getFeatureImportance() {
  return apiFetch("/visualizations/feature-importance");
}

export function getActualVsPredicted() {
  return apiFetch("/visualizations/actual-vs-predicted");
}

export function getErrorDistribution() {
  return apiFetch("/visualizations/error-distribution");
}

export function getModelComparison() {
  return apiFetch("/visualizations/model-comparison");
}

export function getCarbonIntensity() {
  return apiFetch("/visualizations/carbon-intensity");
}

export function getDownloadUrl() {
  return `${API_BASE}/download-predictions`;
}
