import React from "react";

export default function ImageUploader({ onFileSelect }) {
  const handleChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    onFileSelect(file);
  };

  return (
    <label style={styles.uploadButton}>
      이미지 선택하기
      <input
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleChange}
      />
    </label>
  );
}

const styles = {
  uploadButton: {
    padding: "12px 18px",
    backgroundColor: "#f2f2f2",
    border: "1px solid #ccc",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "16px",
    display: "inline-block",
  },
};
