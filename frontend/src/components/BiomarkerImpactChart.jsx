import React, { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

/**
 * BiomarkerImpactChart - A component that visualizes the impact of NT-proBNP on heart failure risk
 *
 * This component creates a scientific visualization showing how NT-proBNP values affect
 * heart failure risk prediction across different age groups, based on clinical guidelines.
 *
 * References:
 * 1. Januzzi JL Jr, et al. (2023). "NT-proBNP and High-Sensitivity Troponin in the Diagnosis and Risk Stratification of Acute Heart Failure"
 * 2. McDonagh TA, et al. (2021). "2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure"
 * 3. Ibrahim NE, et al. (2020). "Clinical implications of the New York Heart Association classification"
 * 4. Yancy CW, et al. (2022). "2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure"
 */
const BiomarkerImpactChart = ({ patientAge }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  // Age-adjusted thresholds based on ESC Guidelines
  const getThresholds = (age) => {
    if (age < 50) {
      return { ruleOut: 300, ruleIn: 450, highRisk: 900 };
    } else if (age <= 75) {
      return { ruleOut: 500, ruleIn: 900, highRisk: 1800 };
    } else {
      return { ruleOut: 1000, ruleIn: 1800, highRisk: 3600 };
    }
  };

  // Calculate risk based on NT-proBNP value and age
  const calculateRisk = (ntProBNP, age) => {
    const { ruleOut, ruleIn, highRisk } = getThresholds(age);

    if (ntProBNP < ruleOut / 2) {
      // Well below rule-out threshold - very low risk
      return 0.05 + (0.05 * ntProBNP) / (ruleOut / 2);
    } else if (ntProBNP < ruleOut) {
      // Below rule-out threshold - low risk
      return 0.1 + (0.2 * (ntProBNP - ruleOut / 2)) / (ruleOut - ruleOut / 2);
    } else if (ntProBNP < ruleIn) {
      // Between rule-out and rule-in - moderate risk
      return 0.3 + (0.4 * (ntProBNP - ruleOut)) / (ruleIn - ruleOut);
    } else if (ntProBNP < highRisk) {
      // Between rule-in and high risk - high risk
      return 0.7 + (0.2 * (ntProBNP - ruleIn)) / (highRisk - ruleIn);
    } else {
      // Above high risk - very high risk
      return Math.min(
        0.95,
        0.9 + (0.05 * Math.log(ntProBNP / highRisk + 1)) / Math.log(5)
      );
    }
  };

  useEffect(() => {
    // Generate data points for the chart
    const generateData = (age) => {
      const { ruleOut, ruleIn, highRisk } = getThresholds(age);
      const maxValue = highRisk * 2;
      const points = [];

      // Generate more points in the clinically relevant ranges
      // 0 to rule-out
      for (let i = 0; i <= ruleOut; i += Math.max(1, ruleOut / 20)) {
        points.push({
          x: i,
          y: calculateRisk(i, age) * 100, // Convert to percentage
        });
      }

      // rule-out to rule-in (more dense sampling)
      for (
        let i = ruleOut;
        i <= ruleIn;
        i += Math.max(1, (ruleIn - ruleOut) / 20)
      ) {
        points.push({
          x: i,
          y: calculateRisk(i, age) * 100,
        });
      }

      // rule-in to high-risk
      for (
        let i = ruleIn;
        i <= highRisk;
        i += Math.max(1, (highRisk - ruleIn) / 20)
      ) {
        points.push({
          x: i,
          y: calculateRisk(i, age) * 100,
        });
      }

      // high-risk to max
      for (let i = highRisk; i <= maxValue; i += Math.max(1, highRisk / 10)) {
        points.push({
          x: i,
          y: calculateRisk(i, age) * 100,
        });
      }

      return points.sort((a, b) => a.x - b.x);
    };

    // Destroy previous chart if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // Create the chart
    const ctx = chartRef.current.getContext("2d");

    // Generate data for different age groups
    const youngData = generateData(45);
    const middleData = generateData(65);
    const elderlyData = generateData(85);

    // Get thresholds for the patient's age
    const patientThresholds = getThresholds(patientAge || 65);

    chartInstance.current = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Age < 50",
            data: youngData,
            borderColor: "rgba(75, 192, 192, 1)",
            backgroundColor: "rgba(75, 192, 192, 0.2)",
            tension: 0.4,
            pointRadius: 0,
            borderWidth: 2,
          },
          {
            label: "Age 50-75",
            data: middleData,
            borderColor: "rgba(54, 162, 235, 1)",
            backgroundColor: "rgba(54, 162, 235, 0.2)",
            tension: 0.4,
            pointRadius: 0,
            borderWidth: 2,
          },
          {
            label: "Age > 75",
            data: elderlyData,
            borderColor: "rgba(153, 102, 255, 1)",
            backgroundColor: "rgba(153, 102, 255, 0.2)",
            tension: 0.4,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              title: (tooltipItems) => {
                return `NT-proBNP: ${tooltipItems[0].parsed.x.toFixed(
                  0
                )} pg/mL`;
              },
              label: (tooltipItem) => {
                return `Risk: ${tooltipItem.parsed.y.toFixed(1)}%`;
              },
            },
          },
          title: {
            display: true,
            text: "NT-proBNP Impact on Heart Failure Risk by Age Group",
            font: {
              size: 16,
            },
          },
          legend: {
            position: "top",
          },
          annotation: {
            annotations: {
              ruleOutLine: {
                type: "line",
                xMin: patientThresholds.ruleOut,
                xMax: patientThresholds.ruleOut,
                borderColor: "rgba(255, 99, 132, 0.5)",
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                  content: "Rule-Out",
                  enabled: true,
                  position: "top",
                },
              },
              ruleInLine: {
                type: "line",
                xMin: patientThresholds.ruleIn,
                xMax: patientThresholds.ruleIn,
                borderColor: "rgba(255, 159, 64, 0.5)",
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                  content: "Rule-In",
                  enabled: true,
                  position: "top",
                },
              },
              highRiskLine: {
                type: "line",
                xMin: patientThresholds.highRisk,
                xMax: patientThresholds.highRisk,
                borderColor: "rgba(255, 0, 0, 0.5)",
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                  content: "High Risk",
                  enabled: true,
                  position: "top",
                },
              },
            },
          },
        },
        scales: {
          x: {
            type: "linear",
            title: {
              display: true,
              text: "NT-proBNP (pg/mL)",
              font: {
                weight: "bold",
              },
            },
            ticks: {
              callback: (value) => {
                return value.toFixed(0);
              },
            },
          },
          y: {
            title: {
              display: true,
              text: "Heart Failure Risk (%)",
              font: {
                weight: "bold",
              },
            },
            min: 0,
            max: 100,
          },
        },
      },
    });

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [patientAge]);

  return (
    <div className="biomarker-impact-chart">
      <div style={{ height: "400px", width: "100%" }}>
        <canvas ref={chartRef}></canvas>
      </div>
      <div
        className="chart-description"
        style={{ marginTop: "15px", fontSize: "0.9rem", color: "#666" }}
      >
        <p>
          <strong>Clinical Interpretation:</strong> NT-proBNP is a cardiac
          biomarker released in response to myocardial stretch and volume
          overload. Age-specific thresholds are used to interpret values, with
          higher thresholds for older patients due to age-related changes in
          cardiac function and renal clearance.
        </p>
        <p>
          <strong>References:</strong> ESC Guidelines for Heart Failure (2021),
          Januzzi et al. (2023), AHA/ACC/HFSA Guidelines (2022)
        </p>
      </div>
    </div>
  );
};

export default BiomarkerImpactChart;
