import axios from "axios";

export async function sendImageToSpring(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await axios.post(
    "http://localhost:8080/pattern/predict",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return res.data; // { prediction: "stripe", confidence: 0.97 }
}