import { useState } from "react";
import { Layers, Download, AlertTriangle, CheckCircle } from "lucide-react";
import toast from "react-hot-toast";
import { batchClassify } from "../services/api";
import CategoryBadge, { PriorityBadge } from "../components/CategoryBadge";

const DEFAULT_BATCH = [
  "RWY 10L CLSD DUE TO MAINTENANCE",
  "ILS RWY 28R NOT AVAILABLE",
  "RESTRICTED AREA R-4009 ACTIVE SFC-10000FT",
  "PAPI RWY 18 OTS",
  "NEW CRANE 480FT AGL WITHIN 2NM OF EHAM ARP",
  "FUEL NOT AVBL 0600-1400 DAILY",
].join("\n");

const PRIORITY_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };

export default function BatchPage() {
  const [input,   setInput]   = useState(DEFAULT_BATCH);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleBatch = async () => {
    const lines = input.split("\n").map(l => l.trim()).filter(Boolean);
    if (!lines.length) return;
    setLoading(true);
    try {
      const { data } = await batchClassify(lines);
      const sorted = data.results
        .map((r, i) => ({ ...r, text: lines[i] }))
        .sort((a, b) => (PRIORITY_ORDER[a.priority] || 4) - (PRIORITY_ORDER[b.priority] || 4));
      setResults(sorted);
      toast.success(`${sorted.length} NOTAMs classified`);
    } catch (err) {
      toast.error("Batch classification failed");
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
    const header = "text,category,priority,confidence\n";
    const rows = results.map(r =>
      `"${r.text}","${r.category}","${r.priority}","${r.confidence.toFixed(4)}"`
    ).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = "notams_classified.csv"; a.click();
  };

  const criticalCount = results.filter(r => r.priority === "CRITICAL").length;

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>BATCH PROCESSING</h1>
          <p style={styles.subtitle}>Classify multiple NOTAMs · max 100 per request</p>
        </div>
        {results.length > 0 && (
          <button style={styles.downloadBtn} onClick={downloadCSV}>
            <Download size={13} /> EXPORT CSV
          </button>
        )}
      </div>

      <div style={styles.grid}>
        {/* Input */}
        <div style={styles.panel}>
          <div style={styles.panelHeader}>
            <span style={styles.panelLabel}>⌨ INPUT — ONE NOTAM PER LINE</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-dim)" }}>
              {input.split("\n").filter(Boolean).length} NOTAMs
            </span>
          </div>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value.toUpperCase())}
            style={styles.textarea}
            rows={14}
            placeholder={"RWY 28L CLSD...\nILS RWY 10 OTS...\n..."}
          />
          <button
            style={{ ...styles.classifyBtn, opacity: loading ? 0.5 : 1 }}
            onClick={handleBatch}
            disabled={loading}
          >
            <Layers size={14} />
            {loading ? "PROCESSING..." : `CLASSIFY ${input.split("\n").filter(Boolean).length} NOTAMs`}
          </button>
        </div>

        {/* Results */}
        <div style={styles.panel}>
          <div style={styles.panelHeader}>
            <span style={styles.panelLabel}>◈ RESULTS — SORTED BY PRIORITY</span>
            {results.length > 0 && (
              <div style={{ display: "flex", gap: 10 }}>
                {criticalCount > 0 && (
                  <span style={styles.criticalTag}>
                    <AlertTriangle size={10} /> {criticalCount} CRITICAL
                  </span>
                )}
                <span style={styles.okTag}>
                  <CheckCircle size={10} /> {results.length} TOTAL
                </span>
              </div>
            )}
          </div>

          {results.length > 0 ? (
            <div style={styles.resultsList}>
              {results.map((r, i) => (
                <div key={i} style={{
                  ...styles.resultRow,
                  borderLeft: `3px solid ${
                    r.priority === "CRITICAL" ? "var(--critical)"
                    : r.priority === "HIGH"   ? "var(--high)"
                    : r.priority === "MEDIUM" ? "var(--medium)"
                    : "var(--low)"
                  }`,
                }}>
                  <div style={styles.resultRowTop}>
                    <CategoryBadge category={r.category} size="sm" />
                    <PriorityBadge priority={r.priority} />
                    <span style={styles.confSmall}>{(r.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div style={styles.resultRowText}>{r.text}</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={styles.empty}>
              <Layers size={32} color="var(--border-bright)" />
              <p>Results will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: { padding: "80px 24px 40px", maxWidth: 1400, margin: "0 auto" },
  header: {
    display: "flex", justifyContent: "space-between", alignItems: "flex-start",
    marginBottom: 32,
  },
  title: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 32, letterSpacing: "0.1em",
  },
  subtitle: {
    fontFamily: "var(--font-mono)", fontSize: 11,
    color: "var(--text-secondary)", marginTop: 4,
  },
  downloadBtn: {
    display: "flex", alignItems: "center", gap: 6,
    padding: "8px 16px", background: "transparent",
    border: "1px solid var(--low)", borderRadius: 4,
    color: "var(--low)", cursor: "pointer",
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 12, letterSpacing: "0.08em",
  },
  grid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 },
  panel: {
    background: "var(--bg-card)", border: "1px solid var(--border)",
    borderRadius: 8, padding: 24,
    display: "flex", flexDirection: "column", gap: 16,
  },
  panelHeader: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    paddingBottom: 12, borderBottom: "1px solid var(--border)",
  },
  panelLabel: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 11, letterSpacing: "0.12em", color: "var(--amber)",
  },
  textarea: {
    width: "100%", background: "var(--bg-primary)",
    border: "1px solid var(--border)", borderRadius: 6,
    color: "var(--text-primary)", fontFamily: "var(--font-mono)",
    fontSize: 12, padding: 16, resize: "vertical", outline: "none",
    lineHeight: 1.8, flex: 1,
  },
  classifyBtn: {
    display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
    padding: "12px 24px", background: "var(--amber)", border: "none",
    borderRadius: 6, cursor: "pointer", fontFamily: "var(--font-display)",
    fontWeight: 700, fontSize: 13, letterSpacing: "0.1em", color: "#000",
    boxShadow: "0 0 20px var(--amber-glow)",
  },
  criticalTag: {
    display: "flex", alignItems: "center", gap: 4,
    padding: "2px 8px", background: "rgba(239,68,68,0.12)",
    border: "1px solid rgba(239,68,68,0.3)", borderRadius: 3,
    color: "var(--critical)", fontFamily: "var(--font-mono)", fontSize: 10,
  },
  okTag: {
    display: "flex", alignItems: "center", gap: 4,
    padding: "2px 8px", background: "rgba(16,185,129,0.1)",
    border: "1px solid rgba(16,185,129,0.2)", borderRadius: 3,
    color: "var(--low)", fontFamily: "var(--font-mono)", fontSize: 10,
  },
  resultsList: {
    display: "flex", flexDirection: "column", gap: 8,
    overflowY: "auto", maxHeight: 500,
  },
  resultRow: {
    padding: "10px 14px", background: "var(--bg-secondary)",
    borderRadius: 4, display: "flex", flexDirection: "column", gap: 6,
  },
  resultRowTop: { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" },
  resultRowText: {
    fontFamily: "var(--font-mono)", fontSize: 11,
    color: "var(--text-secondary)", lineHeight: 1.4,
  },
  confSmall: {
    marginLeft: "auto", fontFamily: "var(--font-mono)",
    fontSize: 11, color: "var(--amber)",
  },
  empty: {
    flex: 1, display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center", gap: 12,
    color: "var(--text-dim)", fontFamily: "var(--font-mono)",
    fontSize: 12, minHeight: 300,
  },
};