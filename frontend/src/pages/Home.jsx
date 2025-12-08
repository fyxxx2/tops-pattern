import React from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const imageURL = URL.createObjectURL(file);

    navigate("/analyze", {
      state: { file, imageURL },
    });
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Pattern Project</h1>

      <p style={styles.subtitle}>
        당신의 패션을 패턴 분석으로 더 정확하게.
      </p>

      <div style={styles.uploadBox}>
        <label style={styles.uploadLabel}>
          📷 이미지 선택하기
          <input
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
        </label>
      </div>
    </div>
  );
}

const styles = {
  container: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fafafa",
  },
  title: {
    fontSize: "38px",
    fontWeight: "700",
    marginBottom: "12px",
    letterSpacing: "-0.5px",
  },
  subtitle: {
    fontSize: "16px",
    color: "#666",
    marginBottom: "40px",
  },
  uploadBox: {
    border: "2px dashed #bbb",
    padding: "40px 60px",
    borderRadius: "12px",
    backgroundColor: "white",
    boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
    transition: "0.2s",
  },
  uploadLabel: {
    cursor: "pointer",
    fontSize: "18px",
    color: "#333",
  },
};
