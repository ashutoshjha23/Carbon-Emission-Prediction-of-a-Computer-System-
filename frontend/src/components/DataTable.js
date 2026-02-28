import React from "react";

export default function DataTable({ columns, data }) {
  if (!data || data.length === 0) return null;
  const cols = columns || Object.keys(data[0]);

  return (
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              {cols.map((c) => (
                <td key={c}>{row[c] != null ? String(row[c]) : ""}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
