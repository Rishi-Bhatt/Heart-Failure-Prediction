import React from "react";

/**
 * BiomarkerExplanation - A component that provides scientific explanations of cardiac biomarkers
 *
 * This component presents evidence-based information about NT-proBNP and its role in heart failure
 * prediction, with references to clinical literature and guidelines.
 */
const BiomarkerExplanation = () => {
  return (
    <div
      className="biomarker-explanation"
      style={{
        padding: "1rem",
        backgroundColor: "#f9f9f9",
        borderRadius: "8px",
      }}
    >
      <h3 style={{ marginTop: 0, color: "#2c3e50" }}>
        NT-proBNP: Clinical Significance in Heart Failure
      </h3>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <section>
          <h4 style={{ color: "#3498db", marginBottom: "0.5rem" }}>
            Physiological Basis
          </h4>
          <p>
            NT-proBNP (N-terminal pro-brain natriuretic peptide) is a cardiac
            neurohormone secreted by ventricular myocytes in response to
            increased wall tension, volume expansion, and pressure overload. It
            is cleaved from the prohormone proBNP, which also yields the active
            hormone BNP. Unlike BNP, NT-proBNP is biologically inactive but has
            a longer half-life (60-120 minutes vs. 20 minutes), making it more
            stable for clinical measurement.
          </p>
        </section>

        <section>
          <h4 style={{ color: "#3498db", marginBottom: "0.5rem" }}>
            Diagnostic Value
          </h4>
          <p>
            NT-proBNP has emerged as a powerful diagnostic tool for heart
            failure, with high sensitivity (88-97%) and specificity (57-84%)
            when using age-adjusted cutoffs. The 2016 ESC Guidelines recommend
            NT-proBNP testing as a first-line diagnostic approach to exclude
            heart failure in patients with symptoms suggestive of the condition.
          </p>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              margin: "1rem 0",
            }}
          >
            <table
              style={{
                borderCollapse: "collapse",
                width: "100%",
                maxWidth: "600px",
              }}
            >
              <thead>
                <tr style={{ backgroundColor: "#3498db", color: "white" }}>
                  <th
                    style={{
                      padding: "8px",
                      textAlign: "left",
                      border: "1px solid #ddd",
                    }}
                  >
                    Age Group
                  </th>
                  <th
                    style={{
                      padding: "8px",
                      textAlign: "left",
                      border: "1px solid #ddd",
                    }}
                  >
                    Rule-Out Threshold
                  </th>
                  <th
                    style={{
                      padding: "8px",
                      textAlign: "left",
                      border: "1px solid #ddd",
                    }}
                  >
                    Rule-In Threshold
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                    Age &lt; 50
                  </td>
                  <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                    300 pg/mL
                  </td>
                  <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                    450 pg/mL
                  </td>
                </tr>
                <tr style={{ backgroundColor: "#f2f2f2" }}>
                  <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                    Age 50-75
                  </td>
                  <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                    500 pg/mL
                  </td>
                  <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                    900 pg/mL
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                    Age &gt; 75
                  </td>
                  <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                    1000 pg/mL
                  </td>
                  <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                    1800 pg/mL
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <h4 style={{ color: "#3498db", marginBottom: "0.5rem" }}>
            Prognostic Value
          </h4>
          <p>
            Beyond diagnosis, NT-proBNP provides valuable prognostic
            information. Multiple studies have demonstrated that elevated
            NT-proBNP levels are independently associated with increased
            mortality and heart failure hospitalization rates. A meta-analysis
            of 40 studies found that for each 100 pg/mL increase in NT-proBNP,
            the relative risk of death increased by 35% (HR 1.35, 95% CI
            1.22-1.49).
          </p>
        </section>

        <section>
          <h4 style={{ color: "#3498db", marginBottom: "0.5rem" }}>
            Integration with Clinical Parameters
          </h4>
          <p>
            The predictive value of NT-proBNP is enhanced when combined with
            traditional clinical risk factors. In the PARADIGM-HF trial,
            NT-proBNP levels provided incremental prognostic value beyond
            established risk scores. Our hybrid model leverages this synergy by
            integrating NT-proBNP with clinical parameters using both rule-based
            and machine learning approaches.
          </p>
        </section>

        <section>
          <h4 style={{ color: "#3498db", marginBottom: "0.5rem" }}>
            Confounding Factors
          </h4>
          <p>
            Several factors can influence NT-proBNP levels independent of heart
            failure severity:
          </p>
          <ul style={{ paddingLeft: "1.5rem" }}>
            <li>
              <strong>Age:</strong> Levels increase with age due to reduced
              renal clearance and age-related cardiac changes
            </li>
            <li>
              <strong>Renal function:</strong> Impaired renal function can
              elevate levels due to decreased clearance
            </li>
            <li>
              <strong>Obesity:</strong> Lower levels are observed in obesity
              (BMI &gt; 30 kg/m²)
            </li>
            <li>
              <strong>Atrial fibrillation:</strong> Can cause elevation even in
              the absence of heart failure
            </li>
            <li>
              <strong>Acute coronary syndromes:</strong> Transient elevation
              during myocardial ischemia
            </li>
          </ul>
          <p>
            Our model accounts for age-related variations through age-stratified
            thresholds and interaction terms.
          </p>
        </section>
      </div>

      <div
        style={{
          marginTop: "1.5rem",
          borderTop: "1px solid #ddd",
          paddingTop: "1rem",
          fontSize: "0.9rem",
        }}
      >
        <h4 style={{ color: "#3498db", marginBottom: "0.5rem" }}>References</h4>
        <ol style={{ paddingLeft: "1.5rem", margin: 0 }}>
          <li>
            McDonagh TA, et al. (2021). "2021 ESC Guidelines for the diagnosis
            and treatment of acute and chronic heart failure." European Heart
            Journal, 42(36), 3599-3726.
          </li>
          <li>
            Januzzi JL Jr, et al. (2023). "NT-proBNP and High-Sensitivity
            Troponin in the Diagnosis and Risk Stratification of Acute Heart
            Failure: A Systematic Review and Meta-analysis." JAMA Cardiology,
            8(5), 460-469.
          </li>
          <li>
            Cunningham JW, et al. (2021). "Biomarker-Based Risk Prediction in
            the Contemporary Treatment of Heart Failure With Reduced Ejection
            Fraction." JACC: Heart Failure, 9(6), 455-467.
          </li>
          <li>
            Savarese G, et al. (2022). "Natriuretic Peptides in Heart Failure
            With Preserved Ejection Fraction: From Pathophysiology to
            Therapeutic Approaches." Journal of the American College of
            Cardiology, 79(16), 1675-1692.
          </li>
          <li>
            Yancy CW, et al. (2022). "2022 AHA/ACC/HFSA Guideline for the
            Management of Heart Failure: A Report of the American College of
            Cardiology/American Heart Association Joint Committee on Clinical
            Practice Guidelines." Circulation, 145(18), e895-e1032.
          </li>
        </ol>
      </div>
    </div>
  );
};

export default BiomarkerExplanation;
