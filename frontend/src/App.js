import React, { useState } from "react";
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from "react-router-dom";
import WorkflowPage from "./pages/WorkflowPage";
import VisualizationsPage from "./pages/VisualizationsPage";
import AboutPage from "./pages/AboutPage";

const NAV_ITEMS = [
  { path: "/", label: "🟢 Full Workflow" },
  { path: "/visualizations", label: "📊 Visualizations" },
  { path: "/about", label: "ℹ️ About" },
];

function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <aside className="sidebar">
      <img
        className="sidebar-logo"
        src="https://cdn-icons-png.flaticon.com/512/2909/2909765.png"
        alt="logo"
      />
      <h2>Navigation</h2>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.path}
            className={`nav-btn ${location.pathname === item.path ? "active" : ""}`}
            onClick={() => navigate(item.path)}
          >
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<WorkflowPage />} />
            <Route path="/visualizations" element={<VisualizationsPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
