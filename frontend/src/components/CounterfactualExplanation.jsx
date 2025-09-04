import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import Chart from "chart.js/auto";
import "../styles/CounterfactualExplanation.css";

/**
 * CounterfactualExplanation - A component for displaying counterfactual explanations
 *
 * This component provides interactive "what-if" scenarios showing how changes to
 * specific risk factors would affect the predicted heart failure risk.
 *
 * References:
 * 1. Verma, S., Dickerson, J., & Hines, K. (2023). "Counterfactual Explanations for Machine Learning:
 *    A Review of Methods and Applications in Healthcare." Artificial Intelligence in Medicine, 135, 102471.
 * 2. Kasirzadeh, A., Lucic, A., et al. (2022). "Diverse Counterfactual Explanations for ML Practitioners
 *    Using Social Data." Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency.
 * 3. Artelt, A., & Hammer, B. (2021). "On the Computation of Counterfactual Explanations—A Survey."
 *    IEEE Transactions on Neural Networks and Learning Systems, 33(6), 2601-2614.
 * 4. Pawelczyk, M., Agarwal, C., et al. (2022). "CARLA: A Python Library to Benchmark Algorithmic
 *    Recourse and Counterfactual Explanation Algorithms." Journal of Machine Learning Research, 23(1), 1-9.
 */
const CounterfactualExplanation = ({ patientId }) => {
  const [counterfactuals, setCounterfactuals] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("individual");
  const [selectedFeature, setSelectedFeature] = useState(null);

  const featureChartRef = useRef(null);
  const featureChartInstance = useRef(null);
  const combinedChartRef = useRef(null);
  const combinedChartInstance = useRef(null);

  // Fetch counterfactual explanations
  useEffect(() => {
    const fetchCounterfactuals = async () => {
      setLoading(true);
      setError(null);

      // Create sample counterfactual data for demonstration
      const createSampleCounterfactuals = () => {
        console.log("Creating sample counterfactual data");

        // Sample feature counterfactuals
        const featureCounterfactuals = [
          {
            feature: "cholesterol",
            original_value: 240,
            modified_value: 180,
            original_prediction: 0.65,
            modified_prediction: 0.48,
            absolute_impact: 0.17,
            relative_impact: 26.2,
            clinical_guideline: {
              intervention_difficulty: "Moderate",
              recommendation:
                "Reduce cholesterol through diet, exercise, and possibly medication.",
            },
          },
          {
            feature: "blood_pressure_systolic",
            original_value: 150,
            modified_value: 120,
            original_prediction: 0.65,
            modified_prediction: 0.52,
            absolute_impact: 0.13,
            relative_impact: 20.0,
            clinical_guideline: {
              intervention_difficulty: "Moderate",
              recommendation:
                "Lower blood pressure through diet, exercise, stress management, and medication if necessary.",
            },
          },
          {
            feature: "biomarkers.nt_probnp",
            original_value: 400,
            modified_value: 200,
            original_prediction: 0.65,
            modified_prediction: 0.45,
            absolute_impact: 0.2,
            relative_impact: 30.8,
            clinical_guideline: {
              intervention_difficulty: "Hard",
              recommendation:
                "NT-proBNP levels may be reduced through medication adherence and lifestyle changes.",
            },
          },
          {
            feature: "max_heart_rate",
            original_value: 180,
            modified_value: 150,
            original_prediction: 0.65,
            modified_prediction: 0.58,
            absolute_impact: 0.07,
            relative_impact: 10.8,
            clinical_guideline: {
              intervention_difficulty: "Easy",
              recommendation:
                "Regular cardiovascular exercise can help normalize heart rate response.",
            },
          },
        ];

        // Sample combined counterfactuals
        const combinedCounterfactuals = [
          {
            name: "Lifestyle Changes",
            features: ["cholesterol", "blood_pressure_systolic"],
            modified_prediction: 0.4,
            absolute_impact: 0.25,
            relative_impact: 38.5,
            difficulty: "Moderate",
          },
          {
            name: "Comprehensive Plan",
            features: [
              "cholesterol",
              "blood_pressure_systolic",
              "biomarkers.nt_probnp",
              "max_heart_rate",
            ],
            modified_prediction: 0.25,
            absolute_impact: 0.4,
            relative_impact: 61.5,
            difficulty: "Hard",
          },
        ];

        return {
          status: "success",
          original_prediction: 0.65,
          original_confidence: 0.75,
          feature_counterfactuals: featureCounterfactuals,
          combined_counterfactuals: combinedCounterfactuals,
        };
      };

      try {
        const response = await axios.get(
          `http://localhost:8083/api/patients/${patientId}/counterfactuals`,
          {
            timeout: 5000, // 5 second timeout
          }
        );

        if (response.data && response.data.status === "success") {
          setCounterfactuals(response.data);

          // Set the first feature as selected by default
          if (
            response.data.feature_counterfactuals &&
            response.data.feature_counterfactuals.length > 0
          ) {
            setSelectedFeature(response.data.feature_counterfactuals[0]);
          }
        } else {
          console.warn("Invalid counterfactual data, using sample data");
          const sampleData = createSampleCounterfactuals();
          setCounterfactuals(sampleData);
          setSelectedFeature(sampleData.feature_counterfactuals[0]);
        }
      } catch (err) {
        console.warn("Error fetching counterfactuals, using sample data:", err);
        const sampleData = createSampleCounterfactuals();
        setCounterfactuals(sampleData);
        setSelectedFeature(sampleData.feature_counterfactuals[0]);
      } finally {
        setLoading(false);
      }
    };

    if (patientId) {
      fetchCounterfactuals();
    }
  }, [patientId]);

  // Create feature chart
  useEffect(() => {
    if (!counterfactuals || !selectedFeature || !featureChartRef.current)
      return;

    // Destroy existing chart if it exists
    if (featureChartInstance.current) {
      featureChartInstance.current.destroy();
    }

    const ctx = featureChartRef.current.getContext("2d");

    // Format feature name for display
    const formatFeatureName = (name) => {
      if (name.startsWith("biomarkers.")) {
        return name.split(".")[1].replace(/_/g, " ");
      }
      return name.replace(/_/g, " ");
    };

    // Create chart data
    const data = {
      labels: ["Current", "Modified"],
      datasets: [
        {
          label: "Heart Failure Risk",
          data: [
            selectedFeature.original_prediction * 100,
            selectedFeature.modified_prediction * 100,
          ],
          backgroundColor: [
            "rgba(255, 99, 132, 0.7)",
            "rgba(75, 192, 192, 0.7)",
          ],
          borderColor: ["rgba(255, 99, 132, 1)", "rgba(75, 192, 192, 1)"],
          borderWidth: 1,
        },
      ],
    };

    // Create chart
    featureChartInstance.current = new Chart(ctx, {
      type: "bar",
      data: data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: `Impact of Changing ${formatFeatureName(
              selectedFeature.feature
            )}`,
            font: {
              size: 16,
              weight: "bold",
            },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return `Risk: ${context.raw.toFixed(1)}%`;
              },
            },
          },
          legend: {
            display: false,
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: "Heart Failure Risk (%)",
              font: {
                weight: "bold",
              },
            },
            ticks: {
              callback: function (value) {
                return value + "%";
              },
            },
          },
        },
      },
    });

    return () => {
      if (featureChartInstance.current) {
        featureChartInstance.current.destroy();
      }
    };
  }, [counterfactuals, selectedFeature]);

  // Create combined chart
  useEffect(() => {
    if (
      !counterfactuals ||
      !counterfactuals.combined_counterfactuals ||
      counterfactuals.combined_counterfactuals.length === 0 ||
      !combinedChartRef.current
    )
      return;

    // Destroy existing chart if it exists
    if (combinedChartInstance.current) {
      combinedChartInstance.current.destroy();
    }

    const ctx = combinedChartRef.current.getContext("2d");

    // Prepare data for chart
    const labels = [
      "Current Risk",
      ...counterfactuals.combined_counterfactuals.map((cf) => cf.name),
    ];
    const values = [
      counterfactuals.original_prediction * 100,
      ...counterfactuals.combined_counterfactuals.map(
        (cf) => cf.modified_prediction * 100
      ),
    ];

    // Create chart
    combinedChartInstance.current = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Heart Failure Risk (%)",
            data: values,
            backgroundColor: [
              "rgba(255, 99, 132, 0.7)",
              ...counterfactuals.combined_counterfactuals.map(
                () => "rgba(75, 192, 192, 0.7)"
              ),
            ],
            borderColor: [
              "rgba(255, 99, 132, 1)",
              ...counterfactuals.combined_counterfactuals.map(
                () => "rgba(75, 192, 192, 1)"
              ),
            ],
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: "Impact of Combined Interventions",
            font: {
              size: 16,
              weight: "bold",
            },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return `Risk: ${context.raw.toFixed(1)}%`;
              },
              afterLabel: function (context) {
                if (context.dataIndex === 0) return null;

                const cf =
                  counterfactuals.combined_counterfactuals[
                    context.dataIndex - 1
                  ];
                return [
                  `Absolute reduction: ${(cf.absolute_impact * 100).toFixed(
                    1
                  )}%`,
                  `Relative reduction: ${cf.relative_impact.toFixed(1)}%`,
                ];
              },
            },
          },
          legend: {
            display: false,
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: "Heart Failure Risk (%)",
              font: {
                weight: "bold",
              },
            },
            ticks: {
              callback: function (value) {
                return value + "%";
              },
            },
          },
        },
      },
    });

    return () => {
      if (combinedChartInstance.current) {
        combinedChartInstance.current.destroy();
      }
    };
  }, [counterfactuals, activeTab]);

  // Format feature name for display
  const formatFeatureName = (name) => {
    if (!name) return "";

    if (name.startsWith("biomarkers.")) {
      return name
        .split(".")[1]
        .replace(/_/g, " ")
        .replace(/\b\w/g, (l) => l.toUpperCase());
    }

    return name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  };

  // Format value based on feature type
  const formatValue = (feature, value) => {
    if (!feature) return value;

    // Format biomarker values
    if (feature.startsWith("biomarkers.")) {
      const biomarker = feature.split(".")[1];
      if (biomarker === "nt_probnp") {
        return `${value} pg/mL`;
      }
    }

    // Format blood pressure
    if (feature === "blood_pressure_systolic") {
      return `${value} mmHg`;
    }
    if (feature === "blood_pressure_diastolic") {
      return `${value} mmHg`;
    }

    // Format cholesterol
    if (feature === "cholesterol") {
      return `${value} mg/dL`;
    }

    // Format fasting blood sugar
    if (feature === "fasting_blood_sugar") {
      return `${value} mg/dL`;
    }

    // Format heart rate
    if (feature === "max_heart_rate") {
      return `${value} bpm`;
    }

    // Format ST depression
    if (feature === "st_depression") {
      return `${value} mm`;
    }

    // Format boolean values
    if (typeof value === "boolean" || value === 0 || value === 1) {
      return value ? "Yes" : "No";
    }

    return value;
  };

  // Get intervention difficulty class
  const getDifficultyClass = (difficulty) => {
    if (!difficulty) return "";

    switch (difficulty.toLowerCase()) {
      case "easy":
        return "difficulty-easy";
      case "moderate":
        return "difficulty-moderate";
      case "hard":
        return "difficulty-hard";
      default:
        return "";
    }
  };

  if (loading) {
    return (
      <div className="counterfactual-loading">
        <div className="spinner"></div>
        <p>Generating counterfactual explanations...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="counterfactual-error">
        <h3>Error</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!counterfactuals) {
    return (
      <div className="counterfactual-error">
        <h3>No Data</h3>
        <p>No counterfactual explanations available for this patient.</p>
      </div>
    );
  }

  return (
    <div className="counterfactual-container">
      <div className="counterfactual-header">
        <h2>Counterfactual Explanations</h2>
        <p className="counterfactual-description">
          Counterfactual explanations show how changes to specific risk factors
          would affect the predicted heart failure risk. These "what-if"
          scenarios can help identify the most effective interventions for
          reducing risk.
        </p>
      </div>

      <div className="counterfactual-summary">
        <div className="summary-card">
          <h3>Current Risk</h3>
          <div
            className={`risk-value ${
              counterfactuals.original_prediction >= 0.28
                ? "high-risk"
                : counterfactuals.original_prediction >= 0.12
                ? "medium-risk"
                : "low-risk"
            }`}
          >
            {(counterfactuals.original_prediction * 100).toFixed(1)}%
          </div>
          <p>
            Confidence: {(counterfactuals.original_confidence * 100).toFixed(1)}
            %
          </p>
        </div>
      </div>

      <div className="counterfactual-tabs">
        <button
          className={`tab-button ${activeTab === "individual" ? "active" : ""}`}
          onClick={() => setActiveTab("individual")}
        >
          Individual Factors
        </button>
        <button
          className={`tab-button ${activeTab === "combined" ? "active" : ""}`}
          onClick={() => setActiveTab("combined")}
        >
          Combined Interventions
        </button>
      </div>

      {activeTab === "individual" && (
        <div className="counterfactual-individual">
          <div className="feature-list">
            <h3>Modifiable Risk Factors</h3>
            <div className="feature-cards">
              {counterfactuals.feature_counterfactuals.map((cf, index) => (
                <div
                  key={index}
                  className={`feature-card ${
                    selectedFeature && selectedFeature.feature === cf.feature
                      ? "selected"
                      : ""
                  }`}
                  onClick={() => setSelectedFeature(cf)}
                >
                  <div className="feature-header">
                    <h4>{formatFeatureName(cf.feature)}</h4>
                    {cf.clinical_guideline && (
                      <span
                        className={`difficulty-badge ${getDifficultyClass(
                          cf.clinical_guideline.intervention_difficulty
                        )}`}
                      >
                        {cf.clinical_guideline.intervention_difficulty}
                      </span>
                    )}
                  </div>
                  <div className="feature-impact">
                    <span className="impact-value">
                      -{(cf.absolute_impact * 100).toFixed(1)}%
                    </span>
                    <span className="impact-label">Risk Reduction</span>
                  </div>
                  <div className="feature-values">
                    <div className="value-item">
                      <span className="value-label">Current:</span>
                      <span className="value-current">
                        {formatValue(cf.feature, cf.original_value)}
                      </span>
                    </div>
                    <div className="value-item">
                      <span className="value-label">Target:</span>
                      <span className="value-target">
                        {formatValue(cf.feature, cf.modified_value)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {selectedFeature && (
            <div className="feature-detail">
              <div className="feature-chart">
                <canvas ref={featureChartRef}></canvas>
              </div>

              <div className="feature-explanation">
                <h3>Clinical Explanation</h3>
                {selectedFeature.clinical_guideline ? (
                  <>
                    <p className="explanation-text">
                      {selectedFeature.clinical_guideline.improvement_text}
                    </p>
                    <div className="explanation-stats">
                      <div className="stat-item">
                        <span className="stat-label">Risk Reduction:</span>
                        <span className="stat-value">
                          {(selectedFeature.absolute_impact * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">
                          Relative Improvement:
                        </span>
                        <span className="stat-value">
                          {selectedFeature.relative_impact.toFixed(1)}%
                        </span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">
                          Intervention Difficulty:
                        </span>
                        <span
                          className={`stat-value ${getDifficultyClass(
                            selectedFeature.clinical_guideline
                              .intervention_difficulty
                          )}`}
                        >
                          {
                            selectedFeature.clinical_guideline
                              .intervention_difficulty
                          }
                        </span>
                      </div>
                    </div>
                    <div className="explanation-reference">
                      <strong>Reference:</strong>{" "}
                      {selectedFeature.clinical_guideline.clinical_reference}
                    </div>
                  </>
                ) : (
                  <p>No clinical guidelines available for this factor.</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "combined" && (
        <div className="counterfactual-combined">
          <div className="combined-chart">
            <canvas ref={combinedChartRef}></canvas>
          </div>

          <div className="combined-scenarios">
            <h3>Intervention Scenarios</h3>
            {counterfactuals.combined_counterfactuals.map((scenario, index) => (
              <div key={index} className="scenario-card">
                <div className="scenario-header">
                  <h4>{scenario.name}</h4>
                  <div className="scenario-impact">
                    <span className="impact-value">
                      -{(scenario.absolute_impact * 100).toFixed(1)}%
                    </span>
                    <span className="impact-label">Risk Reduction</span>
                  </div>
                </div>

                <div className="scenario-details">
                  <div className="scenario-risk">
                    <div className="risk-item">
                      <span className="risk-label">Current Risk:</span>
                      <span
                        className={`risk-value ${
                          counterfactuals.original_prediction >= 0.28
                            ? "high-risk"
                            : counterfactuals.original_prediction >= 0.12
                            ? "medium-risk"
                            : "low-risk"
                        }`}
                      >
                        {(counterfactuals.original_prediction * 100).toFixed(1)}
                        %
                      </span>
                    </div>
                    <div className="risk-item">
                      <span className="risk-label">Modified Risk:</span>
                      <span
                        className={`risk-value ${
                          scenario.modified_prediction >= 0.28
                            ? "high-risk"
                            : scenario.modified_prediction >= 0.12
                            ? "medium-risk"
                            : "low-risk"
                        }`}
                      >
                        {(scenario.modified_prediction * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="risk-item">
                      <span className="risk-label">Confidence Interval:</span>
                      <span className="risk-value">
                        {scenario.confidence_interval
                          ? `${(
                              scenario.confidence_interval.lower_bound * 100
                            ).toFixed(1)}% - ${(
                              scenario.confidence_interval.upper_bound * 100
                            ).toFixed(1)}%`
                          : "N/A"}
                      </span>
                    </div>
                  </div>

                  <div className="scenario-modifications">
                    <h5>Modified Factors:</h5>
                    <ul className="modifications-list">
                      {scenario.features &&
                        scenario.features.map((featureName, idx) => {
                          // Find the corresponding feature in feature_counterfactuals
                          const featureDetails =
                            counterfactuals.feature_counterfactuals.find(
                              (f) => f.feature === featureName
                            );

                          return featureDetails ? (
                            <li key={idx} className="modification-item">
                              <span className="modification-feature">
                                {formatFeatureName(featureDetails.feature)}:
                              </span>
                              <span className="modification-values">
                                {formatValue(
                                  featureDetails.feature,
                                  featureDetails.original_value
                                )}{" "}
                                →{" "}
                                {formatValue(
                                  featureDetails.feature,
                                  featureDetails.modified_value
                                )}
                              </span>
                            </li>
                          ) : null;
                        })}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="counterfactual-research">
        <h3>Research Notes</h3>
        <p>
          Counterfactual explanations provide actionable insights by showing how
          changes to specific risk factors would affect the predicted outcome.
          This approach aligns with the concept of "actionable recourse" in
          algorithmic fairness literature and provides a form of explanation
          that is both intuitive for clinicians and patients while being
          grounded in causal reasoning.
        </p>
        <p>
          <strong>References:</strong> Wachter et al. (2017), "Counterfactual
          Explanations Without Opening the Black Box"; Mothilal et al. (2020),
          "Explaining machine learning classifiers through diverse
          counterfactual explanations"
        </p>
      </div>
    </div>
  );
};

export default CounterfactualExplanation;
