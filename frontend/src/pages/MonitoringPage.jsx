import { useEffect, useState } from "react";
import { getStats } from "../services/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import { Activity, TrendingUp, CheckCircle, AlertTriangle } from "lucide-react";

const CATEGORY_COLORS = {
  RUNWAY_CLOSURE:"#ef4444", NAVIGATION_AID:"#3b82f6",
  AIRSPACE_RESTRICTION:"#f59e0b", LIGHTING:"#8b5cf6",
  OBSTACLE:"#10b981", AERODROME_PROCEDURE:"#ec4899",
};

export default function MonitoringPage() {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getStats()
      .then(r => setStats(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const chartData = stats?.category_distribution
    ? Object.entries(stats.category_distribution).map(([name, value]) => ({
        name: name.replace("_", "\n"), fullName: name, value,
      }))
    : [];

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <h1 style={styles.title}>MONITORING</h1>
        <p style={styles.subtitle}>Production prediction statistics and model performance</p>
      </div>

      {loading ? (
        <p style={{ fontFamily: "var(--font-mono)", color: "var(--amber)" }}>LOADING STATS...</p>
      ) : !stats || stats.total_predictions === 0 ? (
        <div style={styles.empty}>
          <Activity size={40} color="var(--border-bright)" />
          <p>No predictions logged yet.</p>
          <p style={{ fontSize: 11, color: "var(--text-dim)" }}>
            Use the Classify or Batch page to generate predictions.
          </p>
        </div>
      ) : (
        <>
          {/* KPI Row */}
          <div style={styles.kpiRow}>
            {[
              {
                icon: <Activity size={16} color="var(--amber)" />,
                label: "TOTAL PREDICTIONS",
                value: stats.total_predictions?.toLocaleString(),
                color: "var(--amber)",
              },
              {
                icon: <TrendingUp size={16} color="var(--info)" />,
                label: "AVG CONFIDENCE",
                value: stats.avg_confidence ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "—",
                color: stats.avg_confidence > 0.8 ? "var(--low)"
                     : stats.avg_confidence > 0.6 ? "var(--amber)"
                     : "var(--critical)",
              },
              {
                icon: <CheckCircle size={16} color="var(--low)" />,
                label: "AVG LATENCY",
                value: stats.avg_latency_ms ? `${stats.avg_latency_ms.toFixed(1)}ms` : "—",
                color: stats.avg_latency_ms < 50 ? "var(--low)"
                     : stats.avg_latency_ms < 200 ? "var(--amber)"
                     : "var(--critical)",
              },
              {
                icon: <AlertTriangle size={16} color="var(--medium)" />,
                label: "FEEDBACK ACCURACY",
                value: stats.feedback_accuracy
                  ? `${(stats.feedback_accuracy * 100).toFixed(1)}%`
                  : "N/A",
                color: "var(--medium)",
              },
            ].map(({ icon, label, value, color }) => (
              <div key={label} style={styles.kpiCard}>
                <div style={styles.kpiTop}>{icon}<span style={styles.kpiLabel}>{label}</span></div>
                <div style={{ ...styles.kpiValue, color }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Chart */}
          <div style={styles.chartCard}>
            <div style={styles.panelHeader}>
              <span style={styles.panelLabel}>◈ PREDICTION DISTRIBUTION BY CATEGORY</span>
            </div>
            <div style={{ padding: "24px 8px" }}>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} margin={{ top: 0, right: 20, bottom: 40, left: 0 }}>
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#8b949e", fontSize: 10, fontFamily: "var(--font-mono)" }}
                    axisLine={{ stroke: "var(--border)" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#8b949e", fontSize: 10, fontFamily: "var(--font-mono)" }}
                    axisLine={false} tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--bg-card)", border: "1px solid var(--border)",
                      borderRadius: 6, fontFamily: "var(--font-mono)", fontSize: 11,
                    }}
                    labelStyle={{ color: "var(--amber)", fontWeight: 700 }}
                    itemStyle={{ color: "var(--text-primary)" }}
                    formatter={(v, n, p) => [v, p.payload.fullName]}
                  />
                  <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                    {chartData.map((entry) => (
                      <Cell
                        key={entry.fullName}
                        fill={CATEGORY_COLORS[entry.fullName] || "#555"}
                        opacity={0.85}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confidence range */}
          <div style={styles.chartCard}>
            <div style={styles.panelHeader}>
              <span style={styles.panelLabel}>◈ CONFIDENCE RANGE</span>
            </div>
            <div style={{ padding: 20, display: "flex", gap: 16 }}>
              {[
                { label: "MIN", value: stats.min_confidence, color: "var(--critical)" },
                { label: "AVG", value: stats.avg_confidence, color: "var(--amber)"    },
                { label: "MAX", value: stats.max_confidence, color: "var(--low)"      },
              ].map(({ label, value, color }) => (
                <div key={label} style={styles.confCard}>
                  <div style={styles.confLabel}>{label} CONFIDENCE</div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 28, color, fontWeight: 700 }}>
                    {value ? `${(value * 100).toFixed(1)}%` : "—"}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
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
  empty: {
    display: "flex", flexDirection: "column", alignItems: "center",
    justifyContent: "center", gap: 12, padding: "80px 0",
    color: "var(--text-secondary)", fontFamily: "var(--font-mono)", fontSize: 13,
  },
  kpiRow: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 20 },
  kpiCard: {
    background: "var(--bg-card)", border: "1px solid var(--border)",
    borderRadius: 8, padding: 20,
  },
  kpiTop: { display: "flex", alignItems: "center", gap: 8, marginBottom: 12 },
  kpiLabel: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 10, letterSpacing: "0.1em", color: "var(--text-dim)",
  },
  kpiValue: { fontFamily: "var(--font-mono)", fontSize: 28, fontWeight: 700 },
  chartCard: {
    background: "var(--bg-card)", border: "1px solid var(--border)",
    borderRadius: 8, marginBottom: 20,
  },
  panelHeader: {
    padding: "14px 20px", borderBottom: "1px solid var(--border)",
  },
  panelLabel: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 11, letterSpacing: "0.12em", color: "var(--amber)",
  },
  confCard: {
    flex: 1, padding: "16px 20px", background: "var(--bg-secondary)",
    borderRadius: 6, border: "1px solid var(--border)",
  },
  confLabel: {
    fontFamily: "var(--font-mono)", fontSize: 10,
    color: "var(--text-dim)", marginBottom: 8,
  },
};