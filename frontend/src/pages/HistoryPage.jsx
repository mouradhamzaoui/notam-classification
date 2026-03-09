import { useEffect, useState } from "react";
import { getRecentPredictions } from "../services/api";
import CategoryBadge from "../components/CategoryBadge";
import { Clock, RefreshCw } from "lucide-react";

export default function HistoryPage() {
  const [logs,    setLogs]    = useState([]);
  const [loading, setLoading] = useState(true);
  const [limit,   setLimit]   = useState(50);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const { data } = await getRecentPredictions(limit);
      setLogs(data);
    } catch { setLogs([]); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchLogs(); }, [limit]);

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>PREDICTION HISTORY</h1>
          <p style={styles.subtitle}>{logs.length} recent predictions from database</p>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <select
            value={limit}
            onChange={e => setLimit(Number(e.target.value))}
            style={styles.select}
          >
            {[25, 50, 100, 200].map(v => (
              <option key={v} value={v}>{v} records</option>
            ))}
          </select>
          <button style={styles.refreshBtn} onClick={fetchLogs}>
            <RefreshCw size={13} /> REFRESH
          </button>
        </div>
      </div>

      <div style={styles.tableWrap}>
        <table style={styles.table}>
          <thead>
            <tr style={styles.thead}>
              {["#", "NOTAM TEXT", "CATEGORY", "CONFIDENCE", "LATENCY", "TIMESTAMP"].map(h => (
                <th key={h} style={styles.th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={6} style={styles.loadingCell}>LOADING...</td></tr>
            ) : logs.length === 0 ? (
              <tr><td colSpan={6} style={styles.loadingCell}>NO PREDICTIONS YET</td></tr>
            ) : logs.map((log, i) => (
              <tr key={log.id || i} style={styles.tr}>
                <td style={{ ...styles.td, color: "var(--text-dim)", width: 40 }}>
                  {String(i + 1).padStart(3, "0")}
                </td>
                <td style={{ ...styles.td, maxWidth: 300 }}>
                  <span style={styles.notamText}>
                    {log.input_text?.substring(0, 60)}{log.input_text?.length > 60 ? "…" : ""}
                  </span>
                </td>
                <td style={styles.td}>
                  <CategoryBadge category={log.predicted} size="sm" />
                </td>
                <td style={{ ...styles.td, fontFamily: "var(--font-mono)", fontSize: 12 }}>
                  <span style={{
                    color: log.confidence > 0.85 ? "var(--low)"
                         : log.confidence > 0.65 ? "var(--amber)"
                         : "var(--critical)",
                  }}>
                    {(log.confidence * 100).toFixed(1)}%
                  </span>
                </td>
                <td style={{ ...styles.td, fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-secondary)" }}>
                  {log.latency_ms ? `${log.latency_ms.toFixed(1)}ms` : "—"}
                </td>
                <td style={{ ...styles.td, fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-dim)" }}>
                  <Clock size={9} style={{ marginRight: 4 }} />
                  {log.created_at ? new Date(log.created_at).toLocaleString() : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const styles = {
  page: { padding: "80px 24px 40px", maxWidth: 1400, margin: "0 auto" },
  header: {
    display: "flex", justifyContent: "space-between",
    alignItems: "flex-start", marginBottom: 24,
  },
  title: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 32, letterSpacing: "0.1em",
  },
  subtitle: {
    fontFamily: "var(--font-mono)", fontSize: 11,
    color: "var(--text-secondary)", marginTop: 4,
  },
  select: {
    padding: "7px 12px", background: "var(--bg-card)",
    border: "1px solid var(--border)", borderRadius: 4,
    color: "var(--text-primary)", fontFamily: "var(--font-mono)",
    fontSize: 11, cursor: "pointer", outline: "none",
  },
  refreshBtn: {
    display: "flex", alignItems: "center", gap: 6,
    padding: "7px 14px", background: "transparent",
    border: "1px solid var(--border)", borderRadius: 4,
    color: "var(--text-secondary)", cursor: "pointer",
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 11, letterSpacing: "0.08em",
  },
  tableWrap: {
    border: "1px solid var(--border)", borderRadius: 8,
    overflow: "hidden", overflowX: "auto",
  },
  table: { width: "100%", borderCollapse: "collapse" },
  thead: { background: "var(--bg-tertiary)" },
  th: {
    padding: "10px 16px", textAlign: "left",
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 10, letterSpacing: "0.1em", color: "var(--text-dim)",
    borderBottom: "1px solid var(--border)",
  },
  tr: {
    borderBottom: "1px solid var(--border)",
    transition: "background 0.1s",
  },
  td: { padding: "12px 16px", verticalAlign: "middle" },
  notamText: {
    fontFamily: "var(--font-mono)", fontSize: 11,
    color: "var(--text-secondary)",
  },
  loadingCell: {
    padding: "40px", textAlign: "center",
    fontFamily: "var(--font-mono)", fontSize: 12,
    color: "var(--text-dim)",
  },
};