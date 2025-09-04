import React from "react";
import "../styles/abnormality-timeline.css";

/**
 * ECGAbnormalityDisplay - A shared component for displaying ECG abnormalities
 *
 * This component can be used by both single-lead and 12-lead ECG implementations
 * to ensure consistency in how abnormalities are displayed.
 *
 * @param {Object} abnormalities - Object containing abnormality data
 * @param {string} leadFilter - Optional lead filter for 12-lead ECG
 */
const ECGAbnormalityDisplay = ({ abnormalities, leadFilter }) => {
  // Check if abnormalities is an array (from 12-lead ECG) and convert it to the expected format
  if (Array.isArray(abnormalities)) {
    console.log("Converting abnormalities array to object format");
    abnormalities = convertAbnormalitiesToDisplayFormat(abnormalities);
  }

  // Add sample abnormalities for demonstration if none are present
  if (
    !abnormalities ||
    Object.keys(abnormalities).length === 0 ||
    Object.values(abnormalities).every((arr) => !arr || arr.length === 0)
  ) {
    console.log("No abnormalities found, adding sample data for demonstration");
    abnormalities = getSampleAbnormalities();
  }

  if (!abnormalities || Object.keys(abnormalities).length === 0) {
    return (
      <div className="no-abnormalities">
        <p>No abnormalities detected in the ECG.</p>
        <p className="note">
          Note: This does not rule out cardiac conditions. Please consult with a
          healthcare professional for a complete evaluation.
        </p>
      </div>
    );
  }

  // Helper function to convert abnormalities array to the format expected by this component
  function convertAbnormalitiesToDisplayFormat(abnormalitiesArray) {
    if (!abnormalitiesArray || abnormalitiesArray.length === 0) return {};

    // Group abnormalities by category
    const groupedAbnormalities = {};

    abnormalitiesArray.forEach((abnormality) => {
      const category = abnormality.category || "other";

      if (!groupedAbnormalities[category]) {
        groupedAbnormalities[category] = [];
      }

      groupedAbnormalities[category].push({
        type: abnormality.type,
        description: abnormality.description,
        time: abnormality.time,
        duration: abnormality.duration,
        lead: abnormality.lead,
        confidence: abnormality.confidence,
      });
    });

    return groupedAbnormalities;
  }

  // Function to generate sample abnormalities for demonstration
  function getSampleAbnormalities() {
    return {
      rhythm: [
        {
          type: "Sinus Tachycardia",
          description: "Heart rate exceeds normal resting rate",
          time: 2.5,
          duration: 1.2,
          lead: leadFilter || "II",
          confidence: 0.92,
        },
      ],
      st_changes: [
        {
          type: "ST Depression",
          description: "ST segment depression of 1.2mm",
          time: 4.8,
          duration: 0.8,
          lead: leadFilter || "V5",
          confidence: 0.85,
        },
      ],
      conduction: [
        {
          type: "First-degree AV Block",
          description: "Prolonged PR interval (240ms)",
          time: 1.2,
          duration: 0.6,
          lead: leadFilter || "I",
          confidence: 0.78,
        },
      ],
    };
  }

  // Filter abnormalities by lead if a leadFilter is provided
  const filteredAbnormalities = leadFilter
    ? Object.fromEntries(
        Object.entries(abnormalities).map(([type, instances]) => [
          type,
          instances.filter(
            (instance) => !instance.lead || instance.lead === leadFilter
          ),
        ])
      )
    : abnormalities;

  // Check if we have any abnormalities after filtering
  const hasAbnormalities = Object.values(filteredAbnormalities).some(
    (arr) => arr && arr.length > 0
  );

  if (!hasAbnormalities) {
    return (
      <div className="no-abnormalities">
        <p>
          {leadFilter
            ? `No abnormalities detected in lead ${leadFilter}.`
            : "No abnormalities detected in the ECG."}
        </p>
      </div>
    );
  }

  return (
    <div className="abnormality-timeline">
      {Object.entries(filteredAbnormalities).map(
        ([type, instances]) =>
          instances &&
          instances.length > 0 && (
            <div key={type} className="abnormality-category">
              <h5 className="abnormality-category-title">
                {formatAbnormalityType(type)}
              </h5>
              {instances.map((instance, index) => (
                <div key={index} className="abnormality-item">
                  <span className="abnormality-type">
                    {getAbnormalityTypeDisplay(type, instance)}
                  </span>
                  {instance.description && (
                    <span className="abnormality-description">
                      {instance.description}
                    </span>
                  )}
                  <span className="abnormality-time">
                    {instance.time && (
                      <>
                        at{" "}
                        {typeof instance.time === "number"
                          ? instance.time.toFixed(2)
                          : instance.time}
                        s
                      </>
                    )}
                    {instance.duration &&
                      ` (duration: ${
                        typeof instance.duration === "number"
                          ? instance.duration.toFixed(2)
                          : instance.duration
                      }s)`}
                    {instance.lead &&
                      !leadFilter &&
                      ` in lead ${instance.lead}`}
                    {instance.confidence &&
                      ` (confidence: ${(instance.confidence * 100).toFixed(
                        0
                      )}%)`}
                  </span>
                </div>
              ))}
            </div>
          )
      )}
    </div>
  );
};

// Helper function to format abnormality type for display
const formatAbnormalityType = (type) => {
  return type
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

// Helper function to get the display text for an abnormality type
const getAbnormalityTypeDisplay = (type, instance) => {
  if (type === "PVCs") {
    return "Premature Ventricular Contraction";
  } else if (type === "QT_prolongation") {
    return "QT Prolongation";
  } else if (
    [
      "rhythm",
      "st_changes",
      "conduction",
      "chamber_enlargement",
      "axis_deviation",
      "infarction",
    ].includes(type) &&
    instance.type
  ) {
    return instance.type;
  } else {
    return formatAbnormalityType(type);
  }
};

export default ECGAbnormalityDisplay;
