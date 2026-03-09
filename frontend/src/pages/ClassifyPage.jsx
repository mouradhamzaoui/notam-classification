import { useState } from "react";
import { Send, Zap, Clock, Hash, AlertTriangle } from "lucide-react";
import toast from "react-hot-toast";
import { classifyNOTAM } from "../services/api";
import CategoryBadge, { PriorityBadge } from "../components/CategoryBadge";
import ConfidenceBar from "../components/ConfidenceBar";

const EXAMPLES = [
  "RWY 28L CLSD DUE TO CONSTRUCTION WIP",
  "ILS CAT II RWY 10R NOT AVAILABLE",
  "RESTRICTED AREA R-2508 ACTIVE SFC-18000FT MSL",
  "PAPI RWY 36 OTS",
  "NEW CRANE 520FT AGL WITHIN 3NM OF LFPG ARP",
  "FUEL NOT AVBL 2H DAILY DUE MAINTENANCE",
];

export default function ClassifyPage() {
  const [text, setText]       = useState("");
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);

  const handleClassify = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const { data } = await classifyNOTAM(text);
      setResult(data);
      toast.success("Classification complete", { duration: 2000 });
    } catch (err) {
      toast.error(err.response?.data?.detail || "API unreachable");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>NOTAM CLASSIFIER</h1>
          <p style={styles.subtitle}>
            Automatic classification of aeronautical notices · ICAO standard
          </p>
        </div>
        <div style={styles.headerBadge}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--amber)" }}>
            LinearSVC · TF-IDF · 6 CLASSES
          </span>
        </div>
      </div>

      <div style={styles.grid}>
        {/* Input panel */}
        <div style={styles.panel}>
          <div style={styles.panelHeader}>
            <span style={styles.panelLabel}>⌨ INPUT NOTAM</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-dim)" }}>
              {text.length}/2000
            </span>
          </div>

          {/* Examples */}
          <div style={styles.examplesRow}>
            {EXAMPLES.map((ex, i) => (
              <button key={i} style={styles.exBtn} onClick={() => setText(ex)}>
                EX-{String(i+1).padStart(2,"0")}
              </button>
            ))}
          </div>

          {/* Textarea */}
          <div style={styles.textareaWrap}>
            <textarea
              value={text}
              onChange={e => setText(e.target.value.toUpperCase())}
              placeholder="ENTER NOTAM TEXT..."
              style={styles.textarea}
              rows={5}
              onKeyDown={e => e.key === "Enter" && e.ctrlKey && handleClassify()}
            />
            <div style={styles.textareaCorner} />
          </div>

          {text && (
            <div style={styles.textMeta}>
              <span><Hash size={10} /> {text.length} chars</span>
              <span><Hash size={10} /> {text.split(" ").filter(Boolean).length} words</span>
              <span style={{ color: "var(--text-dim)" }}>CTRL+ENTER to classify</span>
            </div>
          )}

          <button
            style={{ ...styles.classifyBtn, opacity: loading || !text.trim() ? 0.5 : 1 }}
            onClick={handleClassify}
            disabled={loading || !text.trim()}
          >
            {loading ? (
              <span style={{ fontFamily: "var(--font-mono)" }}>PROCESSING...</span>
            ) : (
              <>
                <Zap size={14} />
                <span>CLASSIFY NOTAM</span>
                <Send size={12} />
              </>
            )}
          </button>
        </div>

        {/* Result panel */}
        <div style={styles.panel}>
          <div style={styles.panelHeader}>
            <span style={styles.panelLabel}>◈ CLASSIFICATION RESULT</span>
          </div>

          {result ? (
            <div style={styles.resultContent} className="fade-in">
              {/* Main result */}
              <div style={styles.resultMain}>
                <CategoryBadge category={result.category} size="lg" />
                <PriorityBadge priority={result.priority} />
              </div>

              {/* Confidence score */}
              <div style={styles.confidenceScore}>
                <span style={styles.confLabel}>CONFIDENCE</span>
                <span style={{
                  ...styles.confValue,
                  color: result.confidence > 0.85 ? "var(--low)"
                       : result.confidence > 0.60 ? "var(--amber)"
                       : "var(--critical)",
                }}>
                  {(result.confidence * 100).toFixed(2)}%
                </span>
              </div>

              {/* Low confidence warning */}
              {result.confidence < 0.65 && (
                <div style={styles.warning}>
                  <AlertTriangle size={12} color="var(--amber)" />
                  <span>Low confidence — manual review recommended</span>
                </div>
              )}

              {/* Stats row */}
              <div style={styles.statsRow}>
                {[
                  { label: "LATENCY",  value: `${result.latency_ms?.toFixed(1)}ms`, icon: <Clock size={10}/> },
                  { label: "MODEL",    value: result.model_version?.toUpperCase(), icon: <Zap size={10}/> },
                  { label: "CLASSES",  value: Object.keys(result.probabilities || {}).length, icon: <Hash size={10}/> },
                ].map(({ label, value, icon }) => (
                  <div key={label} style={styles.statPill}>
                    <div style={styles.statLabel}>{icon} {label}</div>
                    <div style={styles.statValue}>{value}</div>
                  </div>
                ))}
              </div>

              {/* Probability distribution */}
              <div style={styles.probSection}>
                <div style={styles.probLabel}>PROBABILITY DISTRIBUTION</div>
                <ConfidenceBar probabilities={result.probabilities} />
              </div>
            </div>
          ) : (
            <div style={styles.emptyResult}>
              <div style={styles.emptyIcon}>◈</div>
              <p>Awaiting input...</p>
              <p style={{ fontSize: 11, marginTop: 4, color: "var(--text-dim)" }}>
                Select an example or type a NOTAM
              </p>
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
    fontSize: 32, letterSpacing: "0.1em", color: "var(--text-primary)",
  },
  subtitle: {
    fontFamily: "var(--font-mono)", fontSize: 11,
    color: "var(--text-secondary)", marginTop: 4,
  },
  headerBadge: {
    padding: "6px 14px", border: "1px solid var(--amber-dim)",
    borderRadius: 4, background: "var(--amber-glow)",
  },
  grid: {
    display: "grid", gridTemplateColumns: "1fr 1fr",
    gap: 20,
  },
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
  examplesRow: { display: "flex", gap: 6, flexWrap: "wrap" },
  exBtn: {
    padding: "4px 10px", background: "var(--bg-secondary)",
    border: "1px solid var(--border)", borderRadius: 3,
    color: "var(--text-secondary)", fontFamily: "var(--font-mono)",
    fontSize: 10, cursor: "pointer", letterSpacing: "0.06em",
    transition: "all 0.15s",
  },
  textareaWrap: { position: "relative" },
  textarea: {
    width: "100%", background: "var(--bg-primary)",
    border: "1px solid var(--border)", borderRadius: 6,
    color: "var(--text-primary)", fontFamily: "var(--font-mono)",
    fontSize: 13, padding: 16, resize: "vertical",
    outline: "none", lineHeight: 1.6,
  },
  textareaCorner: {
    position: "absolute", bottom: 6, right: 6,
    width: 10, height: 10,
    borderBottom: "2px solid var(--amber)",
    borderRight: "2px solid var(--amber)",
    borderRadius: 1,
  },
  textMeta: {
    display: "flex", gap: 16, fontFamily: "var(--font-mono)",
    fontSize: 10, color: "var(--text-secondary)", alignItems: "center",
  },
  classifyBtn: {
    display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
    padding: "12px 24px", background: "var(--amber)",
    border: "none", borderRadius: 6, cursor: "pointer",
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 13, letterSpacing: "0.1em", color: "#000",
    transition: "all 0.2s", boxShadow: "0 0 20px var(--amber-glow)",
  },
  resultContent: { display: "flex", flexDirection: "column", gap: 20 },
  resultMain: { display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" },
  confidenceScore: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "12px 16px", background: "var(--bg-secondary)",
    borderRadius: 6, border: "1px solid var(--border)",
  },
  confLabel: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 11, letterSpacing: "0.1em", color: "var(--text-secondary)",
  },
  confValue: {
    fontFamily: "var(--font-mono)", fontSize: 28, fontWeight: 700,
  },
  warning: {
    display: "flex", alignItems: "center", gap: 8,
    padding: "8px 12px", background: "rgba(245,158,11,0.08)",
    border: "1px solid rgba(245,158,11,0.2)", borderRadius: 4,
    color: "var(--amber)", fontFamily: "var(--font-mono)", fontSize: 11,
  },
  statsRow: { display: "flex", gap: 10 },
  statPill: {
    flex: 1, padding: "10px 12px", background: "var(--bg-secondary)",
    borderRadius: 6, border: "1px solid var(--border)",
    textAlign: "center",
  },
  statLabel: {
    fontFamily: "var(--font-mono)", fontSize: 9,
    color: "var(--text-dim)", letterSpacing: "0.08em",
    display: "flex", alignItems: "center", justifyContent: "center", gap: 4,
    marginBottom: 4,
  },
  statValue: {
    fontFamily: "var(--font-mono)", fontSize: 14,
    color: "var(--amber)", fontWeight: 700,
  },
  probSection: {
    padding: 16, background: "var(--bg-secondary)",
    borderRadius: 6, border: "1px solid var(--border)",
    display: "flex", flexDirection: "column", gap: 12,
  },
  probLabel: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 10, letterSpacing: "0.1em", color: "var(--text-dim)",
  },
  emptyResult: {
    flex: 1, display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    color: "var(--text-dim)", fontFamily: "var(--font-mono)",
    fontSize: 12, minHeight: 300, gap: 8,
  },
  emptyIcon: { fontSize: 48, color: "var(--border-bright)", marginBottom: 8 },
};