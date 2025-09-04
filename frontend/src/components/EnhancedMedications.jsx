import React from "react";
import "./EnhancedFeatures.css";

const EnhancedMedications = ({ medications, handleChange }) => {
  return (
    <div className="enhanced-medications">
      <div className="form-row">
        <div className="form-group checkbox-group">
          <label className="form-label">ACE Inhibitors</label>
          <input
            type="checkbox"
            name="medications.ace_inhibitor"
            checked={medications?.ace_inhibitor || false}
            onChange={handleChange}
          />
          <small className="form-text">
            (e.g., Lisinopril, Enalapril, Ramipril)
          </small>
        </div>

        <div className="form-group checkbox-group">
          <label className="form-label">ARBs</label>
          <input
            type="checkbox"
            name="medications.arb"
            checked={medications?.arb || false}
            onChange={handleChange}
          />
          <small className="form-text">
            (e.g., Losartan, Valsartan, Candesartan)
          </small>
        </div>
      </div>

      <div className="form-row">
        <div className="form-group checkbox-group">
          <label className="form-label">Beta Blockers</label>
          <input
            type="checkbox"
            name="medications.beta_blocker"
            checked={medications?.beta_blocker || false}
            onChange={handleChange}
          />
          <small className="form-text">
            (e.g., Metoprolol, Carvedilol, Bisoprolol)
          </small>
        </div>

        <div className="form-group checkbox-group">
          <label className="form-label">Statins</label>
          <input
            type="checkbox"
            name="medications.statin"
            checked={medications?.statin || false}
            onChange={handleChange}
          />
          <small className="form-text">
            (e.g., Atorvastatin, Simvastatin, Rosuvastatin)
          </small>
        </div>
      </div>

      <div className="form-row">
        <div className="form-group checkbox-group">
          <label className="form-label">Antiplatelet</label>
          <input
            type="checkbox"
            name="medications.antiplatelet"
            checked={medications?.antiplatelet || false}
            onChange={handleChange}
          />
          <small className="form-text">
            (e.g., Aspirin, Clopidogrel, Ticagrelor)
          </small>
        </div>

        <div className="form-group checkbox-group">
          <label className="form-label">Diuretics</label>
          <input
            type="checkbox"
            name="medications.diuretic"
            checked={medications?.diuretic || false}
            onChange={handleChange}
          />
          <small className="form-text">
            (e.g., Furosemide, Hydrochlorothiazide)
          </small>
        </div>
      </div>

      <div className="form-row">
        <div className="form-group checkbox-group">
          <label className="form-label">Calcium Channel Blockers</label>
          <input
            type="checkbox"
            name="medications.calcium_channel_blocker"
            checked={medications?.calcium_channel_blocker || false}
            onChange={handleChange}
          />
          <small className="form-text">
            (e.g., Amlodipine, Diltiazem, Verapamil)
          </small>
        </div>
      </div>
    </div>
  );
};

export default EnhancedMedications;
