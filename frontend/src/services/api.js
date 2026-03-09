// ══════════════════════════════════════════════════════════
// api.js — Service de communication avec la FastAPI
// ══════════════════════════════════════════════════════════

import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: { "Content-Type": "application/json" },
});

// Intercepteur de réponse
api.interceptors.response.use(
  (res) => res,
  (err) => {
    console.error("[API Error]", err.response?.data || err.message);
    return Promise.reject(err);
  }
);

export const classifyNOTAM = (text, model_version = "latest") =>
  api.post("/classify", { text, model_version });

export const batchClassify = (texts) =>
  api.post("/classify/batch", { texts });

export const getHealth = () =>
  api.get("/health");

export const getModelInfo = () =>
  api.get("/health/model");

export const getRecentPredictions = (limit = 50) =>
  api.get(`/monitoring/predictions?limit=${limit}`);

export const getStats = () =>
  api.get("/monitoring/stats");

export const submitFeedback = (prediction_id, true_label) =>
  api.post("/classify/feedback", { prediction_id, true_label });

export default api;