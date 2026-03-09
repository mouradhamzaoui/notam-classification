const CATEGORY_CONFIG = {
  RUNWAY_CLOSURE:       { color: "var(--runway)",    icon: "🛬", label: "RUNWAY CLOSURE",    priority: "CRITICAL" },
  NAVIGATION_AID:       { color: "var(--navaid)",    icon: "📡", label: "NAV AID",           priority: "HIGH"     },
  AIRSPACE_RESTRICTION: { color: "var(--airspace)",  icon: "🚫", label: "AIRSPACE",          priority: "CRITICAL" },
  LIGHTING:             { color: "var(--lighting)",  icon: "💡", label: "LIGHTING",          priority: "MEDIUM"   },
  OBSTACLE:             { color: "var(--obstacle)",  icon: "🏗️", label: "OBSTACLE",          priority: "HIGH"     },
  AERODROME_PROCEDURE:  { color: "var(--aerodrome)", icon: "📋", label: "AERODROME PROC",    priority: "LOW"      },
};

export const CATEGORY_CONFIG_EXPORT = CATEGORY_CONFIG;

export default function CategoryBadge({ category, size = "md" }) {
  const cfg = CATEGORY_CONFIG[category] || {
    color: "#555", icon: "✈️", label: category, priority: "—"
  };

  const sizes = {
    sm: { fontSize: 10, padding: "2px 8px", gap: 4 },
    md: { fontSize: 12, padding: "4px 12px", gap: 6 },
    lg: { fontSize: 14, padding: "6px 16px", gap: 8 },
  };

  return (
    <span style={{
      display: "inline-flex", alignItems: "center",
      gap: sizes[size].gap,
      padding: sizes[size].padding,
      borderRadius: 4,
      background: `${cfg.color}18`,
      border: `1px solid ${cfg.color}44`,
      color: cfg.color,
      fontFamily: "var(--font-display)",
      fontWeight: 700,
      fontSize: sizes[size].fontSize,
      letterSpacing: "0.06em",
      whiteSpace: "nowrap",
    }}>
      <span>{cfg.icon}</span>
      <span>{cfg.label}</span>
    </span>
  );
}

export function PriorityBadge({ priority }) {
  const colors = {
    CRITICAL: "var(--critical)",
    HIGH:     "var(--high)",
    MEDIUM:   "var(--medium)",
    LOW:      "var(--low)",
  };
  const color = colors[priority] || "#555";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 8px", borderRadius: 2,
      background: `${color}18`,
      border: `1px solid ${color}44`,
      color, fontFamily: "var(--font-mono)",
      fontSize: 10, letterSpacing: "0.1em",
    }}>
      ⚡ {priority}
    </span>
  );
}