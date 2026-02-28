import React from "react";

export default function AboutPage() {
  return (
    <>
      <h1 className="page-title">ℹ️ About This Project</h1>
      <div className="card about-section">
        <p style={{ fontSize: "1.1rem" }}>
          <strong>Carbon Emission Prediction of a Computer System</strong> is a professional
          dashboard for collecting, processing, and analyzing the carbon footprint of your
          computing device using machine learning.
        </p>

        <h3 style={{ marginTop: 20 }}>Features</h3>
        <ul>
          <li>Collect real-time system data with customizable intervals and sample size</li>
          <li>Preprocess and clean your data with one click</li>
          <li>Train advanced ML models (XGBoost, Random Forest, etc.) and compare their performance</li>
          <li>Predict emissions and download results</li>
          <li>Visualize feature importance, error distributions, and model comparisons</li>
        </ul>

        <h3 style={{ marginTop: 20 }}>Developed by</h3>
        <p>
          <a href="https://github.com/ashutoshjha23" target="_blank" rel="noreferrer">
            ashutoshjha23
          </a>
        </p>

        <h3 style={{ marginTop: 20 }}>Powered by</h3>
        <p>React, Flask, scikit-learn, XGBoost, Chart.js, Python</p>
      </div>
    </>
  );
}
