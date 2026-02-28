import React, { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Scatter } from "react-chartjs-2";
import {
  getFeatureImportance,
  getActualVsPredicted,
  getErrorDistribution,
  getModelComparison,
  getCarbonIntensity,
} from "../api";

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend);

function ChartCard({ title, children, loading, error }) {
  return (
    <div className="card">
      <h3>{title}</h3>
      {loading && <p style={{ color: "#64748b" }}>Loading chart data...</p>}
      {error && <div className="alert alert-error">{error}</div>}
      {!loading && !error && <div className="chart-container">{children}</div>}
    </div>
  );
}

export default function VisualizationsPage() {
  const [fi, setFi] = useState(null);
  const [fiErr, setFiErr] = useState(null);
  const [avp, setAvp] = useState(null);
  const [avpErr, setAvpErr] = useState(null);
  const [ed, setEd] = useState(null);
  const [edErr, setEdErr] = useState(null);
  const [mc, setMc] = useState(null);
  const [mcErr, setMcErr] = useState(null);
  const [ci, setCi] = useState(null);
  const [ciErr, setCiErr] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try { setFi(await getFeatureImportance()); } catch (e) { setFiErr(e.message); }
      try { setAvp(await getActualVsPredicted()); } catch (e) { setAvpErr(e.message); }
      try { setEd(await getErrorDistribution()); } catch (e) { setEdErr(e.message); }
      try { setMc(await getModelComparison()); } catch (e) { setMcErr(e.message); }
      try { setCi(await getCarbonIntensity()); } catch (e) { setCiErr(e.message); }
      setLoading(false);
    }
    load();
  }, []);

  /* ---- Feature Importance ---- */
  const fiData = fi
    ? {
        labels: fi.labels,
        datasets: [
          {
            label: "Gain",
            data: fi.values,
            backgroundColor: "rgba(67,206,162,0.7)",
            borderColor: "#43cea2",
            borderWidth: 1,
          },
        ],
      }
    : null;

  /* ---- Actual vs Predicted ---- */
  const avpData = avp
    ? {
        datasets: [
          {
            label: "Actual vs Predicted",
            data: avp.actual.map((a, i) => ({ x: a, y: avp.predicted[i] })),
            backgroundColor: "rgba(67,206,162,0.6)",
            borderColor: "#185a9d",
            pointRadius: 5,
          },
          {
            label: "Perfect fit",
            data: [
              { x: Math.min(...avp.actual), y: Math.min(...avp.actual) },
              { x: Math.max(...avp.actual), y: Math.max(...avp.actual) },
            ],
            type: "line",
            borderColor: "red",
            borderDash: [6, 4],
            pointRadius: 0,
            borderWidth: 2,
            fill: false,
          },
        ],
      }
    : null;

  /* ---- Error Distribution ---- */
  const edData = ed
    ? (() => {
        const errors = ed.errors;
        const min = Math.min(...errors);
        const max = Math.max(...errors);
        const binCount = 20;
        const binSize = (max - min) / binCount || 1;
        const bins = Array(binCount).fill(0);
        const labels = [];
        for (let i = 0; i < binCount; i++) {
          const lo = min + i * binSize;
          labels.push(lo.toFixed(2));
          errors.forEach((e) => {
            if (e >= lo && e < lo + binSize) bins[i]++;
          });
        }
        return {
          labels,
          datasets: [
            {
              label: "Frequency",
              data: bins,
              backgroundColor: "rgba(24,90,157,0.7)",
              borderColor: "#fff",
              borderWidth: 1,
            },
          ],
        };
      })()
    : null;

  /* ---- Model Comparison — individual metric charts ---- */
  const mcModels = mc ? Object.keys(mc) : [];

  function mcSingle(key, label, color) {
    if (!mc) return null;
    return {
      labels: mcModels,
      datasets: [
        {
          label,
          data: mcModels.map((m) => mc[m][key]),
          backgroundColor: color,
          borderColor: color,
          borderWidth: 1,
        },
      ],
    };
  }

  const mcMAE   = mcSingle("mae",  "MAE",       "#43cea2");
  const mcRMSE  = mcSingle("rmse", "RMSE",      "#185a9d");
  const mcR2    = mcSingle("r2",   "R² Score",  "#fbbf24");
  const mcMAPE  = mcSingle("mape", "MAPE (%)",  "#ef4444");
  const mcMASE  = mcSingle("mase", "MASE",      "#8b5cf6");

  /* ---- Model Comparison BIC ---- */
  const bicData = mc
    ? (() => {
        const models = Object.keys(mc);
        return {
          labels: models,
          datasets: [
            {
              label: "BIC",
              data: models.map((m) => mc[m].bic),
              backgroundColor: "#f97316",
              borderColor: "#ea580c",
              borderWidth: 1,
            },
          ],
        };
      })()
    : null;

  /* ---- Carbon Intensity ---- */
  const ciData = ci
    ? {
        labels: ci.labels,
        datasets: [
          {
            label: "Grid Carbon Intensity (kg CO₂/kWh)",
            data: ci.values,
            backgroundColor: "rgba(16,185,129,0.5)",
            borderColor: "#10b981",
            borderWidth: 2,
            fill: true,
            type: "line",
            tension: 0.3,
            pointRadius: 4,
          },
        ],
      }
    : null;

  return (
    <>
      <h1 className="page-title">📊 Model Visualizations & Insights</h1>

      <ChartCard title="Feature Importance (XGBoost)" loading={loading && !fi} error={fiErr}>
        {fiData && (
          <Bar
            data={fiData}
            options={{
              indexAxis: "y",
              plugins: { title: { display: true, text: "Top Feature Importances (by Gain)" } },
            }}
          />
        )}
      </ChartCard>

      <ChartCard title="Actual vs Predicted" loading={loading && !avp} error={avpErr}>
        {avpData && (
          <Scatter
            data={avpData}
            options={{
              plugins: { title: { display: true, text: "Actual vs Predicted Emissions" } },
              scales: {
                x: { title: { display: true, text: "Actual Emissions (kg CO₂)" } },
                y: { title: { display: true, text: "Predicted Emissions (kg CO₂)" } },
              },
            }}
          />
        )}
      </ChartCard>

      <ChartCard title="Prediction Error Distribution" loading={loading && !ed} error={edErr}>
        {edData && (
          <Bar
            data={edData}
            options={{
              plugins: { title: { display: true, text: "Distribution of Prediction Errors" } },
              scales: {
                x: { title: { display: true, text: "Prediction Error (kg CO₂)" } },
                y: { title: { display: true, text: "Frequency" } },
              },
            }}
          />
        )}
      </ChartCard>

      <ChartCard title="Carbon Intensity Across Samples" loading={loading && !ci} error={ciErr}>
        {ciData && (
          <>
            <div className="metrics-row" style={{ marginBottom: 16 }}>
              <div className="metric-card">
                <div className="label">Mean</div>
                <div className="value">{ci.mean}</div>
              </div>
              <div className="metric-card">
                <div className="label">Min</div>
                <div className="value">{ci.min}</div>
              </div>
              <div className="metric-card">
                <div className="label">Max</div>
                <div className="value">{ci.max}</div>
              </div>
            </div>
            <Bar
              data={ciData}
              options={{
                plugins: { title: { display: true, text: "Grid Carbon Intensity per Sample" } },
                scales: {
                  y: { title: { display: true, text: "kg CO₂/kWh" } },
                },
              }}
            />
          </>
        )}
      </ChartCard>

      <ChartCard title="Model Comparison — MAE" loading={loading && !mc} error={mcErr}>
        {mcMAE && (
          <Bar data={mcMAE} options={{ plugins: { title: { display: true, text: "Mean Absolute Error (lower is better)" } }, scales: { y: { title: { display: true, text: "MAE" } } } }} />
        )}
      </ChartCard>

      <ChartCard title="Model Comparison — RMSE" loading={loading && !mc} error={mcErr}>
        {mcRMSE && (
          <Bar data={mcRMSE} options={{ plugins: { title: { display: true, text: "Root Mean Squared Error (lower is better)" } }, scales: { y: { title: { display: true, text: "RMSE" } } } }} />
        )}
      </ChartCard>

      <ChartCard title="Model Comparison — R² Score" loading={loading && !mc} error={mcErr}>
        {mcR2 && (
          <Bar data={mcR2} options={{ plugins: { title: { display: true, text: "R² Score (closer to 1 is better)" } }, scales: { y: { title: { display: true, text: "R²" } } } }} />
        )}
      </ChartCard>

      <ChartCard title="Model Comparison — MAPE (%)" loading={loading && !mc} error={mcErr}>
        {mcMAPE && (
          <Bar data={mcMAPE} options={{ plugins: { title: { display: true, text: "Mean Absolute Percentage Error (lower is better)" } }, scales: { y: { title: { display: true, text: "MAPE (%)" } } } }} />
        )}
      </ChartCard>

      <ChartCard title="Model Comparison — MASE" loading={loading && !mc} error={mcErr}>
        {mcMASE && (
          <Bar data={mcMASE} options={{ plugins: { title: { display: true, text: "Mean Absolute Scaled Error (lower is better)" } }, scales: { y: { title: { display: true, text: "MASE" } } } }} />
        )}
      </ChartCard>

      <ChartCard title="Model Comparison — Bayesian Information Criterion (BIC)" loading={loading && !mc} error={mcErr}>
        {bicData && (
          <Bar
            data={bicData}
            options={{
              plugins: {
                title: { display: true, text: "BIC by Model (lower is better)" },
              },
              scales: {
                y: { title: { display: true, text: "BIC" } },
              },
            }}
          />
        )}
      </ChartCard>
    </>
  );
}
