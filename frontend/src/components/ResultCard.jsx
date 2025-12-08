import React from "react";

export default function ResultCard({ result }) {
  if (!result) return null;

  return (
    <div style={styles.card}>
      <h3 style={styles.title}>분석 결과</h3>

      <p style={styles.text}>
        <strong>패턴: </strong> 
        {result.prediction ? result.prediction : "값 없음"}
      </p>

      <p style={styles.text}>
        <strong>정확도: </strong>
        {(result.confidence * 100).toFixed(2)}%
      </p>
    </div>
  );
}

const styles = {
  card: {
    marginTop: "20px",
    padding: "20px",
    border: "1px solid #ddd",
    borderRadius: "10px",
    width: "350px",
    marginLeft: "auto",
    marginRight: "auto",
    backgroundColor: "#ffffff",
    fontSize: "18px",
  },
  title: {
    marginBottom: "20px",
  },
  text: {
    fontSize: "18px",
    marginBottom: "10px",
  },
};
