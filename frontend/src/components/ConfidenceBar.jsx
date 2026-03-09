export default function ConfidenceBar({ probabilities }) {
  if (!probabilities) return null;

  const COLORS = {
    RUNWAY_CLOSURE:       "var(--runway)",
    NAVIGATION_AID:       "var(--navaid)",
    AIRSPACE_RESTRICTION: "var(--airspace)",
    LIGHTING:             "var(--lighting)",
    OBSTACLE:             "var(--obstacle)",
    AERODROME_PROCEDURE:  "var(--aerodrome)",
  };

  const sorted = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {sorted.map(([cat, prob]) => (
        <div key={cat} style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{
            width: 140, fontFamily: "var(--font-mono)",
            fontSize: 10, color: "var(--text-secondary)",
            flexShrink: 0,
          }}>
            {cat.replace("_", " ")}
          </span>
          <div style={{
            flex: 1, height: 6, background: "var(--bg-primary)",
            borderRadius: 3, overflow: "hidden",
          }}>
            <div style={{
              height: "100%",
              width: `${prob * 100}%`,
              background: COLORS[cat] || "#555",
              borderRadius: 3,
              transition: "width 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
              boxShadow: prob > 0.5 ? `0 0 8px ${COLORS[cat]}66` : "none",
            }} />
          </div>
          <span style={{
            width: 42, textAlign: "right",
            fontFamily: "var(--font-mono)", fontSize: 11,
            color: prob > 0.5 ? COLORS[cat] : "var(--text-secondary)",
            fontWeight: prob > 0.5 ? 700 : 400,
          }}>
            {(prob * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}