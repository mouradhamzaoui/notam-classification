import { useEffect, useState } from "react";
import { getHealth, getModelInfo } from "../services/api";
import { CheckCircle, XCircle, Radio, Cpu, Database, Zap } from "lucide-react";

export default function StatusPage() {
  const [health, setHealth]   = useState(null);
  const [model,  setModel]    = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getHealth(), getModelInfo()])
      .then(([h, m]) => { setHealth(h.data); setModel(m.data); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const StatusIcon = ({ ok }) => ok
    ? <CheckCircle size={14} color="var(--low)" />
    : <XCircle    size={14} color="var(--critical)" />;

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <h1 style={styles.title}>SYSTEM STATUS</h1>
        <p style={styles.subtitle}>Real-time health monitoring of all services</p>
      </div>

      {loading ? (
        <div style={styles.loading}>
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--amber)" }}>
            QUERYING SYSTEMS...
          </span>
        </div>
      ) : (
        <div style={styles.grid}>
          {/* API Status */}
          <div style={styles.card}>
            <div style={styles.cardHeader}>
              <Radio size={16} color="var(--amber)" />
              <span style={styles.cardTitle}>API SERVICE</span>
              <StatusIcon ok={health?.status === "healthy"} />
            </div>
            <div style={styles.cardBody}>
              {[
                { label: "STATUS",     value: health?.status?.toUpperCase() || "UNKNOWN" },
                { label: "VERSION",    value: health?.version || "—" },
                { label: "UPTIME",     value: health ? `${health.uptime_s}s` : "—" },
                { label: "MODEL",      value: health?.model_loaded ? "LOADED ✓" : "NOT LOADED" },
                { label: "DATABASE",   value: health?.db_connected ? "CONNECTED ✓" : "OFFLINE" },
              ].map(({ label, value }) => (
                <div key={label} style={styles.row}>
                  <span style={styles.rowLabel}>{label}</span>
                  <span style={styles.rowValue}>{value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Model Info */}
          <div style={styles.card}>
            <div style={styles.cardHeader}>
              <Cpu size={16} color="var(--amber)" />
              <span style={styles.cardTitle}>MODEL INFO</span>
              <StatusIcon ok={!!model} />
            </div>
            <div style={styles.cardBody}>
              {[
                { label: "ALGORITHM",  value: model?.model_name || "—" },
                { label: "VERSION",    value: model?.version || "—" },
                { label: "FEATURES",   value: model?.n_features?.toLocaleString() || "—" },
                { label: "F1-MACRO",   value: model?.f1_macro ? `${(model.f1_macro * 100).toFixed(2)}%` : "—" },
                { label: "ACCURACY",   value: model?.accuracy  ? `${(model.accuracy  * 100).toFixed(2)}%` : "—" },
              ].map(({ label, value }) => (
                <div key={label} style={styles.row}>
                  <span style={styles.rowLabel}>{label}</span>
                  <span style={{
                    ...styles.rowValue,
                    color: label === "F1-MACRO" || label === "ACCURACY"
                      ? "var(--low)" : "var(--text-primary)",
                  }}>{value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Classes */}
          <div style={{ ...styles.card, gridColumn: "span 2" }}>
            <div style={styles.cardHeader}>
              <Zap size={16} color="var(--amber)" />
              <span style={styles.cardTitle}>SUPPORTED CLASSES</span>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, padding: 20 }}>
              {(model?.classes || []).map(cls => {
                const colors = {
                  RUNWAY_CLOSURE:"var(--runway)", NAVIGATION_AID:"var(--navaid)",
                  AIRSPACE_RESTRICTION:"var(--airspace)", LIGHTING:"var(--lighting)",
                  OBSTACLE:"var(--obstacle)", AERODROME_PROCEDURE:"var(--aerodrome)",
                };
                const c = colors[cls] || "#555";
                return (
                  <div key={cls} style={{
                    padding: "10px 20px", borderRadius: 6,
                    background: `${c}12`, border: `1px solid ${c}33`,
                    color: c, fontFamily: "var(--font-display)",
                    fontWeight: 700, fontSize: 12, letterSpacing: "0.08em",
                  }}>
                    {cls.replace(/_/g, " ")}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  page: { padding: "80px 24px 40px", maxWidth: 1400, margin: "0 auto" },
  header: { marginBottom: 32 },
  title: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 32, letterSpacing: "0.1em",
  },
  subtitle: {
    fontFamily: "var(--font-mono)", fontSize: 11,
    color: "var(--text-secondary)", marginTop: 4,
  },
  loading: {
    display: "flex", justifyContent: "center",
    padding: "80px 0",
  },
  grid: {
    display: "grid", gridTemplateColumns: "1fr 1fr",
    gap: 20,
  },
  card: {
    background: "var(--bg-card)", border: "1px solid var(--border)",
    borderRadius: 8,
  },
  cardHeader: {
    display: "flex", alignItems: "center", gap: 10,
    padding: "16px 20px", borderBottom: "1px solid var(--border)",
  },
  cardTitle: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 12, letterSpacing: "0.12em", color: "var(--amber)", flex: 1,
  },
  cardBody: { padding: "8px 0" },
  row: {
    display: "flex", justifyContent: "space-between",
    padding: "10px 20px",
    borderBottom: "1px solid var(--border)",
  },
  rowLabel: {
    fontFamily: "var(--font-mono)", fontSize: 10,
    color: "var(--text-dim)", letterSpacing: "0.08em",
  },
  rowValue: {
    fontFamily: "var(--font-mono)", fontSize: 12,
    color: "var(--text-primary)", fontWeight: 600,
  },
};