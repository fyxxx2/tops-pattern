import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { sendImageToSpring } from "../api/springClient";
import ResultCard from "../components/ResultCard";

export default function Analyze() {
  const navigate = useNavigate();
  const { state } = useLocation();
  const { file, imageURL } = state || {};

  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    if (!file) return;

    try {
      const response = await sendImageToSpring(file);
      setResult(response);
    } catch (error) {
      alert("분석 중 오류가 발생했습니다.");
    }
  };

  const goHome = () => {
    navigate("/");
  };

  return (
    <>
      <div style={styles.container}>
        <h1 style={styles.title}>이미지 분석</h1>

        {/* 이미지 카드 */}
        {imageURL && (
          <div style={styles.imageWrapper}>
            <img src={imageURL} alt="uploaded" style={styles.image} />
          </div>
        )}

        {/* 분석하기 버튼 */}
        {!result && (
          <button style={styles.button} onClick={handleAnalyze}>
            분석하기
          </button>
        )}

        {/* 결과 카드 */}
        {result && (
          <>
            <div style={styles.resultWrapper}>
              <ResultCard result={result} />
            </div>

            <button style={styles.retryButton} onClick={goHome}>
              다른 이미지 분석하기
            </button>
          </>
        )}
      </div>

      {/* Fade-Up Animation Keyframes */}
      <style>
        {`
          @keyframes fadeUp {
            from {
              opacity: 0;
              transform: translateY(20px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>
    </>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fafafa",
    textAlign: "center",
  },

  title: {
    fontSize: "38px",
    fontWeight: "700",
    marginBottom: "25px",
  },

  imageWrapper: {
    backgroundColor: "#fff",
    padding: "20px",
    borderRadius: "12px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
    marginBottom: "30px",
  },

  image: {
    width: "350px",
    borderRadius: "10px",
  },

  button: {
    padding: "12px 25px",
    backgroundColor: "#111",
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "16px",
    marginBottom: "30px",
    transition: "0.2s",
  },

  resultWrapper: {
    display: "flex",
    justifyContent: "center",
    width: "100%",
    opacity: 0,
    transform: "translateY(20px)",
    animation: "fadeUp 0.45s ease-out forwards",
  },

  retryButton: {
    padding: "12px 25px",
    backgroundColor: "#666",
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "16px",
    marginTop: "20px",
  },
};
