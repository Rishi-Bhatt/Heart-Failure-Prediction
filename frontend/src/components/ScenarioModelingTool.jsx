/**
 * ScenarioModelingTool - A component for creating and comparing intervention scenarios
 *
 * This component allows users to create "what-if" scenarios by modifying risk factors
 * and visualizing the impact on predicted risk trajectories.
 *
 * References:
 * 1. Verma, S., et al. (2023). "Counterfactual Explanations for Machine Learning: A Review of Methods and Applications in Healthcare"
 * 2. Prosperi, M., et al. (2020). "Causal inference and counterfactual prediction in machine learning for actionable healthcare"
 */
import React, { useState } from "react";
import ForecastService from "../services/ForecastService";
import "../styles/ScenarioModelingTool.css";

const ScenarioModelingTool = ({
  patientId,
  baselineForecast,
  onScenarioUpdate,
}) => {
  const [scenarios, setScenarios] = useState([]);
  const [isCreatingScenario, setIsCreatingScenario] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // New scenario form state
  const [scenarioName, setScenarioName] = useState("");
  const [interventions, setInterventions] = useState([]);

  // Available intervention types
  const interventionTypes = [
    {
      id: "nt_probnp",
      name: "NT-proBNP",
      type: "biomarker",
      unit: "pg/mL",
      min: 0,
      max: 5000,
      step: 50,
    },
    {
      id: "blood_pressure",
      name: "Blood Pressure",
      type: "clinical",
      unit: "mmHg",
      format: "systolic/diastolic",
      systolicMin: 90,
      systolicMax: 200,
      systolicStep: 5,
      diastolicMin: 60,
      diastolicMax: 120,
      diastolicStep: 5,
    },
    {
      id: "cholesterol",
      name: "Cholesterol",
      type: "clinical",
      unit: "mg/dL",
      min: 100,
      max: 300,
      step: 5,
    },
    {
      id: "max_heart_rate",
      name: "Max Heart Rate",
      type: "clinical",
      unit: "bpm",
      min: 60,
      max: 200,
      step: 5,
    },
    {
      id: "exercise",
      name: "Exercise",
      type: "lifestyle",
      options: [
        { value: "none", label: "None" },
        { value: "light", label: "Light (1-2 days/week)" },
        { value: "moderate", label: "Moderate (3-5 days/week)" },
        { value: "intense", label: "Intense (6-7 days/week)" },
      ],
    },
    {
      id: "diet",
      name: "Diet",
      type: "lifestyle",
      options: [
        { value: "poor", label: "Poor" },
        { value: "average", label: "Average" },
        { value: "good", label: "Good" },
        { value: "excellent", label: "Excellent (Mediterranean/DASH)" },
      ],
    },
  ];

  // Add an intervention to the current scenario
  const addIntervention = (type, value) => {
    // Check if intervention of this type already exists
    const existingIndex = interventions.findIndex((i) => i.type === type);

    if (existingIndex >= 0) {
      // Update existing intervention
      const updatedInterventions = [...interventions];
      updatedInterventions[existingIndex] = { type, value };
      setInterventions(updatedInterventions);
    } else {
      // Add new intervention
      setInterventions([...interventions, { type, value }]);
    }
  };

  // Remove an intervention from the current scenario
  const removeIntervention = (type) => {
    setInterventions(interventions.filter((i) => i.type !== type));
  };

  // Handle intervention value change
  const handleInterventionChange = (type, value) => {
    addIntervention(type, value);
  };

  // Handle blood pressure change (special case)
  const handleBPChange = (systolic, diastolic) => {
    addIntervention("blood_pressure", `${systolic}/${diastolic}`);
  };

  // Create a new scenario
  const createScenario = async () => {
    if (!scenarioName) {
      setError("Please provide a name for the scenario");
      return;
    }

    if (interventions.length === 0) {
      setError("Please add at least one intervention to the scenario");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const scenarioData = {
        name: scenarioName,
        interventions: interventions,
      };

      console.log("Creating scenario:", scenarioData);

      // Use a consistent cache key for this scenario to prevent duplicate requests
      const cacheKey = `scenario-${patientId}-${scenarioName
        .replace(/\s+/g, "-")
        .toLowerCase()}`;

      const response = await ForecastService.getScenarioForecast(
        patientId,
        scenarioData,
        undefined, // Use default horizon
        cacheKey
      );

      console.log("Scenario response:", response.data);

      if (response.data.status === "success") {
        // Add scenario to list
        const newScenario = {
          ...response.data,
          scenario_name: scenarioName,
          interventions: interventions,
        };

        const updatedScenarios = [...scenarios, newScenario];
        setScenarios(updatedScenarios);

        // Call the onScenarioUpdate callback
        if (onScenarioUpdate) {
          onScenarioUpdate(updatedScenarios);
        }

        // Reset form
        setScenarioName("");
        setInterventions([]);
        setIsCreatingScenario(false);
      } else {
        setError(response.data.message || "Failed to create scenario");
      }
    } catch (err) {
      console.error("Error creating scenario:", err);

      // Extract error message from response if available
      let errorMessage = "An error occurred while creating the scenario";

      if (err.response?.data?.message) {
        errorMessage = err.response.data.message;
      } else if (err.response?.data?.error) {
        errorMessage = err.response.data.error;
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // Remove a scenario
  const removeScenario = (index) => {
    const updatedScenarios = scenarios.filter((_, i) => i !== index);
    setScenarios(updatedScenarios);

    // Call the onScenarioUpdate callback
    if (onScenarioUpdate) {
      onScenarioUpdate(updatedScenarios);
    }
  };

  // Calculate risk reduction for a scenario
  const calculateRiskReduction = (scenario) => {
    if (!baselineForecast || !scenario) {
      return "N/A";
    }

    // Get peak risks
    const baselinePeak = baselineForecast.peak_risk;
    const scenarioPeak = scenario.peak_risk;

    // Calculate reduction
    const reduction = baselinePeak - scenarioPeak;
    const percentReduction = (reduction / baselinePeak) * 100;

    return percentReduction > 0
      ? `${percentReduction.toFixed(1)}% reduction`
      : `${Math.abs(percentReduction).toFixed(1)}% increase`;
  };

  // Render intervention form based on type
  const renderInterventionForm = (interventionType) => {
    const currentValue = interventions.find(
      (i) => i.type === interventionType.id
    )?.value;

    switch (interventionType.id) {
      case "blood_pressure":
        // Extract systolic and diastolic from current value
        let systolic = 120;
        let diastolic = 80;

        if (currentValue) {
          const [sys, dia] = currentValue.split("/");
          systolic = parseInt(sys) || 120;
          diastolic = parseInt(dia) || 80;
        }

        return (
          <div className="intervention-form-item" key={interventionType.id}>
            <div className="intervention-header">
              <label>{interventionType.name}</label>
              {currentValue && (
                <button
                  className="remove-intervention-button"
                  onClick={() => removeIntervention(interventionType.id)}
                >
                  Remove
                </button>
              )}
            </div>
            <div className="blood-pressure-inputs">
              <div className="bp-input-group">
                <label>Systolic:</label>
                <input
                  type="number"
                  min={interventionType.systolicMin}
                  max={interventionType.systolicMax}
                  step={interventionType.systolicStep}
                  value={systolic}
                  onChange={(e) =>
                    handleBPChange(parseInt(e.target.value), diastolic)
                  }
                />
              </div>
              <div className="bp-input-group">
                <label>Diastolic:</label>
                <input
                  type="number"
                  min={interventionType.diastolicMin}
                  max={interventionType.diastolicMax}
                  step={interventionType.diastolicStep}
                  value={diastolic}
                  onChange={(e) =>
                    handleBPChange(systolic, parseInt(e.target.value))
                  }
                />
              </div>
              <span className="unit-label">{interventionType.unit}</span>
            </div>
          </div>
        );

      case "exercise":
      case "diet":
        return (
          <div className="intervention-form-item" key={interventionType.id}>
            <div className="intervention-header">
              <label>{interventionType.name}</label>
              {currentValue && (
                <button
                  className="remove-intervention-button"
                  onClick={() => removeIntervention(interventionType.id)}
                >
                  Remove
                </button>
              )}
            </div>
            <select
              value={currentValue || ""}
              onChange={(e) =>
                handleInterventionChange(interventionType.id, e.target.value)
              }
            >
              <option value="">Select {interventionType.name}</option>
              {interventionType.options.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        );

      default:
        // Default numeric input
        return (
          <div className="intervention-form-item" key={interventionType.id}>
            <div className="intervention-header">
              <label>{interventionType.name}</label>
              {currentValue && (
                <button
                  className="remove-intervention-button"
                  onClick={() => removeIntervention(interventionType.id)}
                >
                  Remove
                </button>
              )}
            </div>
            <div className="numeric-input-container">
              <input
                type="number"
                min={interventionType.min}
                max={interventionType.max}
                step={interventionType.step}
                value={currentValue || ""}
                onChange={(e) =>
                  handleInterventionChange(interventionType.id, e.target.value)
                }
                placeholder={`Enter ${interventionType.name}`}
              />
              <span className="unit-label">{interventionType.unit}</span>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="scenario-modeling-tool">
      <div className="scenario-header">
        <h3>What-If Scenario Modeling</h3>
        {!isCreatingScenario && (
          <button
            className="create-scenario-button"
            onClick={() => setIsCreatingScenario(true)}
          >
            Create New Scenario
          </button>
        )}
      </div>

      {error && <div className="error-message">{error}</div>}

      {isCreatingScenario && (
        <div className="scenario-form">
          <h4>Create New Scenario</h4>

          <div className="form-group">
            <label>Scenario Name:</label>
            <input
              type="text"
              value={scenarioName}
              onChange={(e) => setScenarioName(e.target.value)}
              placeholder="e.g., Improved Diet and Exercise"
            />
          </div>

          <div className="intervention-groups">
            <div className="intervention-group">
              <h5>Biomarkers</h5>
              {interventionTypes
                .filter((type) => type.type === "biomarker")
                .map(renderInterventionForm)}
            </div>

            <div className="intervention-group">
              <h5>Clinical Parameters</h5>
              {interventionTypes
                .filter((type) => type.type === "clinical")
                .map(renderInterventionForm)}
            </div>

            <div className="intervention-group">
              <h5>Lifestyle Factors</h5>
              {interventionTypes
                .filter((type) => type.type === "lifestyle")
                .map(renderInterventionForm)}
            </div>
          </div>

          <div className="form-actions">
            <button
              className="cancel-button"
              onClick={() => {
                setIsCreatingScenario(false);
                setScenarioName("");
                setInterventions([]);
                setError(null);
              }}
            >
              Cancel
            </button>
            <button
              className="create-button"
              onClick={createScenario}
              disabled={isLoading}
            >
              {isLoading ? "Creating..." : "Create Scenario"}
            </button>
          </div>
        </div>
      )}

      {scenarios.length > 0 && (
        <div className="scenarios-list">
          <h4>Scenarios</h4>
          <div className="scenario-cards">
            {scenarios.map((scenario, index) => (
              <div className="scenario-card" key={index}>
                <div className="scenario-card-header">
                  <h5>{scenario.scenario_name}</h5>
                  <button
                    className="remove-scenario-button"
                    onClick={() => removeScenario(index)}
                  >
                    Remove
                  </button>
                </div>

                <div className="scenario-details">
                  <div className="scenario-metric">
                    <span className="metric-label">Risk Reduction:</span>
                    <span className="metric-value">
                      {calculateRiskReduction(scenario)}
                    </span>
                  </div>

                  <div className="scenario-metric">
                    <span className="metric-label">Peak Risk:</span>
                    <span className="metric-value">
                      {(scenario.peak_risk * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="scenario-interventions">
                    <span className="interventions-label">Interventions:</span>
                    <ul className="interventions-list">
                      {scenario.interventions.map((intervention, i) => {
                        const interventionType = interventionTypes.find(
                          (t) => t.id === intervention.type
                        );
                        let displayValue = intervention.value;

                        if (interventionType) {
                          if (interventionType.options) {
                            const option = interventionType.options.find(
                              (o) => o.value === intervention.value
                            );
                            displayValue = option
                              ? option.label
                              : intervention.value;
                          } else if (interventionType.unit) {
                            displayValue = `${intervention.value} ${interventionType.unit}`;
                          }
                        }

                        return (
                          <li key={i}>
                            <strong>
                              {interventionType
                                ? interventionType.name
                                : intervention.type}
                              :
                            </strong>{" "}
                            {displayValue}
                          </li>
                        );
                      })}
                    </ul>
                  </div>

                  {/* Display scenario-specific insights */}
                  {scenario.insights &&
                    scenario.insights.recommendations &&
                    scenario.insights.recommendations.length > 0 && (
                      <div className="scenario-insights">
                        <span className="insights-label">Key Insights:</span>
                        <ul className="insights-list">
                          {scenario.insights.recommendations
                            .slice(0, 3)
                            .map((recommendation, i) => (
                              <li key={i}>{recommendation}</li>
                            ))}
                        </ul>
                      </div>
                    )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="research-notes">
        <h5>Research Notes</h5>
        <p>
          Scenario modeling enables clinicians and patients to explore potential
          interventions and their impact on future risk. This approach supports
          shared decision-making by quantifying the expected benefits of
          different treatment strategies.
        </p>
        <p>
          <strong>References:</strong> Verma, S., et al. (2023). "Counterfactual
          Explanations for Machine Learning: A Review of Methods and
          Applications in Healthcare"
        </p>
      </div>
    </div>
  );
};

export default ScenarioModelingTool;
