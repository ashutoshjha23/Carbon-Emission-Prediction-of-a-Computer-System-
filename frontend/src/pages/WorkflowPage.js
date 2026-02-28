import React, { useState } from "react";
import { collectData, uploadCsv, preprocessData, trainModel, predictEmissions, getDownloadUrl } from "../api";
import DataTable from "../components/DataTable";

export default function WorkflowPage() {
  /* ---- Step 1 state ---- */
  const [runs, setRuns] = useState(5);
  const [interval, setInterval_] = useState(2);
  const [collectLoading, setCollectLoading] = useState(false);
  const [collectResult, setCollectResult] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  /* ---- Step 2 state ---- */
  const [preprocessLoading, setPreprocessLoading] = useState(false);
  const [preprocessResult, setPreprocessResult] = useState(null);

  /* ---- Step 3 state ---- */
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainResult, setTrainResult] = useState(null);

  /* ---- Step 4 state ---- */
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictResult, setPredictResult] = useState(null);

  /* ---- Global error ---- */
  const [error, setError] = useState(null);

  async function handleCollect() {
    setError(null);
    setCollectLoading(true);
    setCollectResult(null);
    try {
      const res = await collectData(runs, interval);
      setCollectResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setCollectLoading(false);
    }
  }

  async function handleUpload() {
    if (!selectedFile) return;
    setError(null);
    setUploadLoading(true);
    setCollectResult(null);
    try {
      const res = await uploadCsv(selectedFile);
      setCollectResult(res);
      setSelectedFile(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploadLoading(false);
    }
  }

  async function handlePreprocess() {
    setError(null);
    setPreprocessLoading(true);
    setPreprocessResult(null);
    try {
      const res = await preprocessData();
      setPreprocessResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setPreprocessLoading(false);
    }
  }

  async function handleTrain() {
    setError(null);
    setTrainLoading(true);
    setTrainResult(null);
    try {
      const res = await trainModel();
      setTrainResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setTrainLoading(false);
    }
  }

  async function handlePredict() {
    setError(null);
    setPredictLoading(true);
    setPredictResult(null);
    try {
      const res = await predictEmissions();
      setPredictResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setPredictLoading(false);
    }
  }

  return (
    <>
      <h1 className="page-title">💻 Carbon Emission Prediction Dashboard</h1>

      {error && <div className="alert alert-error">{error}</div>}

      {/* ---------- Step 1 ---------- */}
      <div className="card">
        <h3>Step 1: Collect Real-Time System Data</h3>
        <div className="input-group">
          <div className="input-field">
            <label>Number of samples (runs)</label>
            <input
              type="number"
              min={1}
              max={100}
              value={runs}
              onChange={(e) => setRuns(Number(e.target.value))}
            />
          </div>
          <div className="input-field">
            <label>Interval (seconds)</label>
            <input
              type="number"
              min={1}
              max={60}
              value={interval}
              onChange={(e) => setInterval_(Number(e.target.value))}
            />
          </div>
          <button className="btn btn-primary" disabled={collectLoading} onClick={handleCollect}>
            {collectLoading ? <><span className="spinner" /> Collecting...</> : "Collect Data"}
          </button>
        </div>

        <div className="divider-or"><span>OR</span></div>

        <h3>Upload Your Own CSV</h3>
        <div className="input-group">
          <div className="input-field">
            <label>Select a system_data CSV file</label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setSelectedFile(e.target.files[0] || null)}
            />
          </div>
          <button
            className="btn btn-secondary"
            disabled={uploadLoading || !selectedFile}
            onClick={handleUpload}
          >
            {uploadLoading ? <><span className="spinner" /> Uploading...</> : "Upload CSV"}
          </button>
        </div>

        {collectResult && (
          <>
            <div className="alert alert-success">{collectResult.message}</div>
            <DataTable columns={collectResult.columns} data={collectResult.data} />
          </>
        )}
      </div>

      {/* ---------- Step 2 ---------- */}
      <div className="card">
        <h3>Step 2: Preprocess Data</h3>
        <button className="btn btn-primary" disabled={preprocessLoading} onClick={handlePreprocess}>
          {preprocessLoading ? <><span className="spinner" /> Processing...</> : "Preprocess Data"}
        </button>
        {preprocessResult && (
          <>
            <div className="alert alert-success" style={{ marginTop: 12 }}>
              {preprocessResult.message}
            </div>
            <DataTable columns={preprocessResult.columns} data={preprocessResult.data} />
          </>
        )}
      </div>

      {/* ---------- Step 3 ---------- */}
      <div className="card">
        <h3>Step 3: Train Model</h3>
        <button className="btn btn-primary" disabled={trainLoading} onClick={handleTrain}>
          {trainLoading ? <><span className="spinner" /> Training (this may take a while)...</> : "Train Model"}
        </button>
        {trainResult && (
          <>
            <div className="alert alert-success" style={{ marginTop: 12 }}>
              {trainResult.message}
            </div>
            <div className="metrics-row">
              <div className="metric-card">
                <div className="label">MAE</div>
                <div className="value">{trainResult.metrics.mae ?? "N/A"}</div>
              </div>
              <div className="metric-card">
                <div className="label">RMSE</div>
                <div className="value">{trainResult.metrics.rmse ?? "N/A"}</div>
              </div>
              <div className="metric-card">
                <div className="label">R² Score</div>
                <div className="value">{trainResult.metrics.r2 ?? "N/A"}</div>
              </div>
              <div className="metric-card">
                <div className="label">MAPE (%)</div>
                <div className="value">{trainResult.metrics.mape ?? "N/A"}</div>
              </div>
              <div className="metric-card">
                <div className="label">MASE</div>
                <div className="value">{trainResult.metrics.mase ?? "N/A"}</div>
              </div>
              <div className="metric-card">
                <div className="label">BIC</div>
                <div className="value">{trainResult.metrics.bic ?? "N/A"}</div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* ---------- Step 4 ---------- */}
      <div className="card">
        <h3>Step 4: Predict Emissions</h3>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <button className="btn btn-primary" disabled={predictLoading} onClick={handlePredict}>
            {predictLoading ? <><span className="spinner" /> Predicting...</> : "Predict Emissions"}
          </button>
          {predictResult && (
            <a className="btn btn-download" href={getDownloadUrl()} download>
              ⬇ Download Predictions CSV
            </a>
          )}
        </div>
        {predictResult && (
          <>
            <div className="alert alert-success" style={{ marginTop: 12 }}>
              {predictResult.message}
            </div>
            <DataTable columns={predictResult.columns} data={predictResult.data} />
          </>
        )}
      </div>
    </>
  );
}
