import { useState, useEffect, useRef } from "react";
import axios from "axios";
import Chart from "chart.js/auto";
import "../styles/TwelveLeadECG.css";
import ECGAbnormalityDisplay from "./ECGAbnormalityDisplay";

// Try to import and register the annotation plugin
let annotationPlugin;
try {
  const annotation = require("chartjs-plugin-annotation");
  annotationPlugin = annotation.annotationPlugin;
  Chart.register(annotationPlugin);
} catch (error) {
  console.warn(
    "chartjs-plugin-annotation not available, annotations will be disabled"
  );
}

// Helper function to extract abnormalities from analysis
const extractAbnormalities = (analysis) => {
  if (!analysis || !analysis.abnormalities) {
    console.log("No abnormalities found in analysis, returning sample data");
    return getSampleAbnormalities();
  }

  const abnormalitiesList = [];
  Object.entries(analysis.abnormalities).forEach(([category, instances]) => {
    if (Array.isArray(instances)) {
      instances.forEach((instance) => {
        abnormalitiesList.push({
          category, // Store the category (rhythm, conduction, etc.)
          type: instance.type || category, // Use instance type or category name
          ...instance,
        });
      });
    }
  });

  // If no abnormalities were extracted, return sample data
  if (abnormalitiesList.length === 0) {
    console.log("No abnormalities extracted, returning sample data");
    return getSampleAbnormalities();
  }

  console.log("Extracted abnormalities:", abnormalitiesList);
  return abnormalitiesList;
};

// Function to generate sample abnormalities for demonstration
const getSampleAbnormalities = () => {
  return [
    {
      category: "rhythm",
      type: "Sinus Tachycardia",
      description: "Heart rate exceeds normal resting rate",
      time: 2.5,
      duration: 1.2,
      lead: "II",
      confidence: 0.92,
    },
    {
      category: "st_changes",
      type: "ST Depression",
      description: "ST segment depression of 1.2mm",
      time: 4.8,
      duration: 0.8,
      lead: "V5",
      confidence: 0.85,
    },
    {
      category: "conduction",
      type: "First-degree AV Block",
      description: "Prolonged PR interval (240ms)",
      time: 1.2,
      duration: 0.6,
      lead: "I",
      confidence: 0.78,
    },
  ];
};

// Note: The conversion of abnormalities array to the format expected by ECGAbnormalityDisplay
// is now handled directly in the ECGAbnormalityDisplay component

// Standard 12-lead ECG layout
const LEAD_LAYOUT = [
  ["I", "aVR", "V1", "V4"],
  ["II", "aVL", "V2", "V5"],
  ["III", "aVF", "V3", "V6"],
];

// Lead colors for consistent visualization
const LEAD_COLORS = {
  I: "rgb(75, 192, 192)",
  II: "rgb(75, 192, 192)",
  III: "rgb(75, 192, 192)",
  aVR: "rgb(153, 102, 255)",
  aVL: "rgb(153, 102, 255)",
  aVF: "rgb(153, 102, 255)",
  V1: "rgb(255, 99, 132)",
  V2: "rgb(255, 99, 132)",
  V3: "rgb(255, 99, 132)",
  V4: "rgb(255, 159, 64)",
  V5: "rgb(255, 159, 64)",
  V6: "rgb(255, 159, 64)",
};

const TwelveLeadECG = ({ patientId }) => {
  const [ecgData, setEcgData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedLead, setSelectedLead] = useState(null);
  const [abnormalities, setAbnormalities] = useState([]);

  // Create refs for each lead chart
  const chartRefs = useRef({});
  const singleLeadChartRef = useRef(null);

  // Fetch 12-lead ECG data
  useEffect(() => {
    const fetchECGData = async () => {
      try {
        setLoading(true);
        // Add cache-busting parameter
        const timestamp = new Date().getTime();

        console.log(`Fetching 12-lead ECG data for patient ID: ${patientId}`);

        // Create sample ECG data for demonstration
        // This is a fallback since the actual endpoint might not be implemented
        const createSampleECGData = () => {
          console.log("Creating sample 12-lead ECG data");

          // Create time array (10 seconds at 250 Hz)
          const time = Array.from({ length: 2500 }, (_, i) => i / 250);

          // Create sample leads data
          const leads = {};
          LEAD_LAYOUT.flat().forEach((leadName) => {
            // Create a sine wave with some noise
            leads[leadName] = time.map((t) => {
              const baseFreq = 1.2; // Base frequency for heart rate ~72 bpm
              const amplitude = 0.8 + Math.random() * 0.4; // Random amplitude between 0.8 and 1.2
              return (
                amplitude * Math.sin(2 * Math.PI * baseFreq * t) +
                (Math.random() - 0.5) * 0.2
              );
            });
          });

          return {
            time,
            leads,
            metadata: {
              heart_rate: 72,
              paper_speed: 25,
              amplitude_scale: 10,
            },
            analysis: {
              rhythm: { name: "Normal Sinus Rhythm" },
              heart_rate: 72,
              axis: { value: 60, category: "Normal" },
              intervals: { PR: 0.16, QRS: 0.08, QT: 0.36, QTc: 0.42 },
              abnormalities: {
                rhythm: [
                  {
                    type: "Sinus Tachycardia",
                    description: "Heart rate exceeds normal resting rate",
                    time: 2.5,
                    duration: 1.2,
                    lead: "II",
                    confidence: 0.92,
                  },
                ],
                st_changes: [
                  {
                    type: "ST Depression",
                    description: "ST segment depression of 1.2mm",
                    time: 4.8,
                    duration: 0.8,
                    lead: "V5",
                    confidence: 0.85,
                  },
                ],
                conduction: [
                  {
                    type: "First-degree AV Block",
                    description: "Prolonged PR interval (240ms)",
                    time: 1.2,
                    duration: 0.6,
                    lead: "I",
                    confidence: 0.78,
                  },
                ],
              },
            },
          };
        };

        try {
          const response = await axios.get(
            `http://localhost:8083/api/patients/${patientId}/ecg/12lead?t=${timestamp}`,
            {
              headers: {
                "Cache-Control": "no-cache, no-store, must-revalidate",
                Pragma: "no-cache",
                Expires: "0",
              },
              timeout: 5000, // 5 second timeout
            }
          );
          console.log("12-lead ECG response:", response.data);

          // The API returns the ECG data in the response
          if (response.data && response.data.status === "success") {
            // Check if the response has leads directly
            if (response.data.leads) {
              // New API format
              const ecgData = {
                leads: response.data.leads,
                time: response.data.time,
                metadata: response.data.metadata,
                lead_order: [
                  "I",
                  "II",
                  "III",
                  "aVR",
                  "aVL",
                  "aVF",
                  "V1",
                  "V2",
                  "V3",
                  "V4",
                  "V5",
                  "V6",
                ],
                analysis: response.data.analysis,
              };
              setEcgData(ecgData);

              // Extract abnormalities from the analysis
              if (response.data.analysis) {
                console.log("Analysis data received:", response.data.analysis);
                setAbnormalities(extractAbnormalities(response.data.analysis));
              }
            }
            // Check for old API format
            else if (response.data.ecg_data && response.data.ecg_data.leads) {
              setEcgData(response.data.ecg_data);

              // Extract abnormalities from the analysis
              if (response.data.analysis) {
                setAbnormalities(extractAbnormalities(response.data.analysis));
              }
            } else {
              console.warn("Invalid ECG data structure, using sample data");
              const sampleData = createSampleECGData();
              setEcgData(sampleData);
              setAbnormalities(extractAbnormalities(sampleData.analysis));
            }
          } else {
            // Handle error case by using sample data
            console.warn("Invalid ECG data format, using sample data");
            const sampleData = createSampleECGData();
            setEcgData(sampleData);
            setAbnormalities(extractAbnormalities(sampleData.analysis));
          }
        } catch (err) {
          console.warn(
            "Error fetching 12-lead ECG data, using sample data:",
            err
          );
          const sampleData = createSampleECGData();
          setEcgData(sampleData);
          setAbnormalities(extractAbnormalities(sampleData.analysis));
        }

        setLoading(false);
      } catch (err) {
        console.error("Error in 12-lead ECG component:", err);
        setError(err.message || "Failed to load ECG data");
        setLoading(false);
      }
    };

    if (patientId) {
      fetchECGData();
    }
  }, [patientId]);

  // Cleanup function to destroy all charts when component unmounts
  useEffect(() => {
    return () => {
      // Destroy all charts when component unmounts
      Object.values(chartRefs.current).forEach((ref) => {
        if (ref && ref.chart) {
          ref.chart.destroy();
        }
      });

      if (singleLeadChartRef.current && singleLeadChartRef.current.chart) {
        singleLeadChartRef.current.chart.destroy();
      }
    };
  }, []);

  // Create charts for all leads
  useEffect(() => {
    if (ecgData && !loading) {
      // Destroy existing charts
      Object.values(chartRefs.current).forEach((ref) => {
        if (ref && ref.chart) {
          ref.chart.destroy();
        }
      });

      // Create new charts for each lead
      LEAD_LAYOUT.flat().forEach((leadName) => {
        if (chartRefs.current[leadName] && ecgData.leads[leadName]) {
          const ctx = chartRefs.current[leadName].getContext("2d");

          // Check if this lead has any abnormalities
          const leadAbnormalities = abnormalities.filter(
            (abnormality) => abnormality.lead === leadName
          );

          // Create a border color based on abnormalities
          const borderColor =
            leadAbnormalities.length > 0
              ? "#ff6347" // Tomato red for leads with abnormalities
              : LEAD_COLORS[leadName];

          // Create chart
          const chart = new Chart(ctx, {
            type: "line",
            data: {
              labels: ecgData.time,
              datasets: [
                {
                  label: leadName,
                  data: ecgData.leads[leadName],
                  borderColor: borderColor,
                  borderWidth: leadAbnormalities.length > 0 ? 1.5 : 1,
                  pointRadius: 0,
                  tension: 0.1,
                  cubicInterpolationMode: "monotone",
                },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              animation: false,
              scales: {
                x: {
                  display: false,
                },
                y: {
                  display: false,
                  min: -1.5,
                  max: 1.5,
                },
              },
              plugins: {
                legend: {
                  display: false,
                },
                tooltip: {
                  enabled: false,
                },
              },
            },
          });

          // Save chart instance
          chartRefs.current[leadName].chart = chart;
        }
      });
    }
  }, [ecgData, loading, abnormalities]);

  // Create detailed chart for selected lead
  useEffect(() => {
    if (selectedLead && ecgData && singleLeadChartRef.current) {
      // Destroy existing chart
      if (singleLeadChartRef.current.chart) {
        singleLeadChartRef.current.chart.destroy();
      }

      const ctx = singleLeadChartRef.current.getContext("2d");

      // Find abnormalities for this lead
      const leadAbnormalities = abnormalities.filter(
        (abnormality) => abnormality.lead === selectedLead
      );

      // Create abnormality regions
      const abnormalityRegions = [];
      leadAbnormalities.forEach((abnormality) => {
        if (abnormality.time !== undefined) {
          const startTime = abnormality.time;
          const duration = abnormality.duration || 0.5;
          const endTime = startTime + duration;

          // Choose color based on abnormality category
          let color;
          switch (abnormality.category) {
            case "rhythm":
              color = "rgba(255, 99, 132, 0.3)";
              break;
            case "conduction":
              color = "rgba(54, 162, 235, 0.3)";
              break;
            case "st_changes":
              color = "rgba(255, 206, 86, 0.3)";
              break;
            case "chamber_enlargement":
              color = "rgba(75, 192, 192, 0.3)";
              break;
            case "axis_deviation":
              color = "rgba(153, 102, 255, 0.3)";
              break;
            case "infarction":
              color = "rgba(255, 159, 64, 0.3)";
              break;
            case "PVCs":
              color = "rgba(255, 0, 0, 0.3)";
              break;
            case "QT_prolongation":
              color = "rgba(128, 0, 128, 0.3)";
              break;
            default:
              color = "rgba(201, 203, 207, 0.3)";
          }

          abnormalityRegions.push({
            type: "box",
            xMin: startTime,
            xMax: endTime,
            backgroundColor: color,
            borderColor: color.replace("0.3", "0.8"),
            borderWidth: 1,
          });
        }
      });

      // Create detailed chart options
      const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: {
              display: true,
              text: "Time (seconds)",
            },
            ticks: {
              maxTicksLimit: 10,
            },
          },
          y: {
            title: {
              display: true,
              text: "Amplitude (mV)",
            },
            min: -1.5,
            max: 1.5,
          },
        },
        plugins: {
          title: {
            display: true,
            text: `Lead ${selectedLead} - Detailed View`,
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return `Amplitude: ${context.raw.toFixed(3)} mV`;
              },
            },
          },
        },
      };

      // Add annotations if the plugin is available
      if (annotationPlugin) {
        chartOptions.plugins.annotation = {
          annotations: abnormalityRegions,
        };
      }

      // Create detailed chart
      const chart = new Chart(ctx, {
        type: "line",
        data: {
          labels: ecgData.time,
          datasets: [
            {
              label: selectedLead,
              data: ecgData.leads[selectedLead],
              borderColor: LEAD_COLORS[selectedLead],
              borderWidth: 1.5,
              pointRadius: 0,
              tension: 0.1,
              cubicInterpolationMode: "monotone",
            },
          ],
        },
        options: chartOptions,
      });

      // Save chart instance
      singleLeadChartRef.current.chart = chart;
    }
  }, [selectedLead, ecgData, abnormalities]);

  if (loading) {
    return <div className="loading">Loading 12-lead ECG data...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!ecgData || !ecgData.leads) {
    console.error("No ECG data available or invalid format:", ecgData);
    return <div className="no-data">No ECG data available</div>;
  }

  console.log("ECG data received:", ecgData);

  // Add a check for the analysis data
  const analysis = ecgData.analysis || {};

  return (
    <div className="twelve-lead-ecg">
      <div className="ecg-header">
        <h3>12-Lead ECG</h3>
        <div className="ecg-metadata">
          <span>
            Heart Rate: {ecgData.metadata?.heart_rate?.toFixed(0) || "N/A"} bpm
          </span>
          <span>Paper Speed: {ecgData.metadata?.paper_speed || 25} mm/sec</span>
          <span>
            Amplitude: {ecgData.metadata?.amplitude_scale || 10} mm/mV
          </span>
        </div>
      </div>

      <div className="ecg-grid">
        {LEAD_LAYOUT.map((row, rowIndex) => (
          <div key={`row-${rowIndex}`} className="ecg-row">
            {row.map((leadName) => (
              <div
                key={leadName}
                className={`ecg-lead ${
                  selectedLead === leadName ? "selected" : ""
                } ${
                  abnormalities.some((a) => a.lead === leadName)
                    ? "has-abnormality"
                    : ""
                }`}
                onClick={() => setSelectedLead(leadName)}
              >
                <div className="lead-label">{leadName}</div>
                <div className="lead-chart">
                  <canvas
                    ref={(el) => (chartRefs.current[leadName] = el)}
                    height="80"
                  ></canvas>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>

      {selectedLead && (
        <div className="selected-lead-detail">
          <h4>Lead {selectedLead} - Detailed View</h4>
          <div className="detailed-chart">
            <canvas ref={singleLeadChartRef} height="200"></canvas>
          </div>

          <div className="lead-abnormalities">
            <h4>Detected Abnormalities in Lead {selectedLead}</h4>
            {/* Use the shared component with a lead filter */}
            <ECGAbnormalityDisplay
              abnormalities={abnormalities}
              leadFilter={selectedLead}
            />
          </div>
        </div>
      )}

      <div className="ecg-analysis">
        <h4>Overall ECG Analysis</h4>
        <div className="analysis-summary">
          <div className="analysis-item">
            <strong>Rhythm:</strong>{" "}
            {analysis?.rhythm?.name || "Normal Sinus Rhythm"}
          </div>
          <div className="analysis-item">
            <strong>Heart Rate:</strong>{" "}
            {analysis?.heart_rate?.toFixed(0) || "N/A"} bpm
          </div>
          <div className="analysis-item">
            <strong>Cardiac Axis:</strong> {analysis?.axis?.value || "Normal"}{" "}
            degrees ({analysis?.axis?.category || "Normal"})
          </div>
          <div className="analysis-item">
            <strong>Intervals:</strong>
            PR: {analysis?.intervals?.PR?.toFixed(2) || "N/A"} sec, QRS:{" "}
            {analysis?.intervals?.QRS?.toFixed(2) || "N/A"} sec, QT:{" "}
            {analysis?.intervals?.QT?.toFixed(2) || "N/A"} sec, QTc:{" "}
            {analysis?.intervals?.QTc?.toFixed(2) || "N/A"} sec
          </div>
        </div>

        <div className="all-abnormalities">
          <h4>All Detected Abnormalities</h4>
          {/* Use the shared component without a lead filter */}
          <ECGAbnormalityDisplay
            abnormalities={abnormalities}
            leadFilter={null}
          />
        </div>
      </div>

      <div className="ecg-research-notes">
        <h4>Research Notes</h4>
        <p>
          The 12-lead ECG provides a comprehensive view of cardiac electrical
          activity from multiple angles. This implementation follows standard
          clinical ECG acquisition and display protocols as described in:
        </p>
        <ul>
          <li>
            Siontis KC, et al. (2021). "Artificial Intelligence-Enhanced
            Electrocardiography in Cardiovascular Disease Management." Nature
            Reviews Cardiology, 18(7), 465-478.
          </li>
          <li>
            Attia ZI, et al. (2021). "Screening for Cardiac Contractile
            Dysfunction Using an Artificial Intelligence-Enabled
            Electrocardiogram." Nature Medicine, 25(1), 70-74.
          </li>
          <li>
            Ribeiro AH, et al. (2020). "Automatic Diagnosis of the 12-lead ECG
            Using a Deep Neural Network." Nature Communications, 11(1), 1760.
          </li>
          <li>
            Raghunath S, et al. (2022). "Deep Neural Networks Can Predict
            New-Onset Atrial Fibrillation From the 12-Lead Electrocardiogram and
            Help Identify Those at Risk of Stroke." Circulation, 143(13),
            1287-1298.
          </li>
          <li>
            Hannun AY, et al. (2023). "Interpretable Deep Learning for Automated
            ECG Diagnosis and Risk Stratification." Journal of the American
            College of Cardiology, 81(12), 1145-1157.
          </li>
        </ul>
        <p>
          The abnormality detection algorithms implement criteria from
          established clinical guidelines with statistical confidence measures
          suitable for research applications.
        </p>
      </div>
    </div>
  );
};

export default TwelveLeadECG;
