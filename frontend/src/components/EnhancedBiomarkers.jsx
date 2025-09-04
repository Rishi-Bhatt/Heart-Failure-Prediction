import React from "react";
import "./EnhancedFeatures.css";

const EnhancedBiomarkers = ({ biomarkers, handleChange, age }) => {
  return (
    <div className="enhanced-biomarkers">
      <div className="form-row">
        <div className="form-group">
          <label className="form-label">NT-proBNP (pg/mL)</label>
          <input
            type="number"
            name="biomarkers.nt_probnp"
            value={biomarkers?.nt_probnp || ""}
            onChange={handleChange}
            className="form-input"
            min="0"
            placeholder="e.g., 125"
          />
          <small
            className="form-text"
            style={{
              display: "block",
              marginTop: "5px",
              fontSize: "0.8rem",
              color: "#666",
            }}
          >
            Reference range:{" "}
            {parseInt(age) < 50
              ? "0-450"
              : parseInt(age) <= 75
              ? "0-900"
              : "0-1800"}{" "}
            pg/mL (age-adjusted)
          </small>
        </div>

        <div className="form-group">
          <label className="form-label">Troponin (ng/mL)</label>
          <input
            type="number"
            name="biomarkers.troponin"
            value={biomarkers?.troponin || ""}
            onChange={handleChange}
            className="form-input"
            min="0"
            step="0.01"
            placeholder="e.g., 0.01"
          />
          <small
            className="form-text"
            style={{
              display: "block",
              marginTop: "5px",
              fontSize: "0.8rem",
              color: "#666",
            }}
          >
            Reference range: 0-0.04 ng/mL
          </small>
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label className="form-label">CRP (mg/L)</label>
          <input
            type="number"
            name="biomarkers.crp"
            value={biomarkers?.crp || ""}
            onChange={handleChange}
            className="form-input"
            min="0"
            step="0.1"
            placeholder="e.g., 1.0"
          />
          <small
            className="form-text"
            style={{
              display: "block",
              marginTop: "5px",
              fontSize: "0.8rem",
              color: "#666",
            }}
          >
            Reference range: 0-3.0 mg/L
          </small>
        </div>

        <div className="form-group">
          <label className="form-label">BNP (pg/mL)</label>
          <input
            type="number"
            name="biomarkers.bnp"
            value={biomarkers?.bnp || ""}
            onChange={handleChange}
            className="form-input"
            min="0"
            placeholder="e.g., 50"
          />
          <small
            className="form-text"
            style={{
              display: "block",
              marginTop: "5px",
              fontSize: "0.8rem",
              color: "#666",
            }}
          >
            Reference range: 0-100 pg/mL
          </small>
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label className="form-label">Creatinine (mg/dL)</label>
          <input
            type="number"
            name="biomarkers.creatinine"
            value={biomarkers?.creatinine || ""}
            onChange={handleChange}
            className="form-input"
            min="0"
            step="0.1"
            placeholder="e.g., 0.8"
          />
          <small
            className="form-text"
            style={{
              display: "block",
              marginTop: "5px",
              fontSize: "0.8rem",
              color: "#666",
            }}
          >
            Reference range: 0.7-1.3 mg/dL (male), 0.6-1.1 mg/dL (female)
          </small>
        </div>
      </div>
    </div>
  );
};

export default EnhancedBiomarkers;
