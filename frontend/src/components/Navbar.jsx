import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  Plane, Radio, Layers, Activity,
  Clock, Menu, X, Zap
} from "lucide-react";

const NAV_ITEMS = [
  { to: "/",           icon: Zap,      label: "CLASSIFY"   },
  { to: "/batch",      icon: Layers,   label: "BATCH"      },
  { to: "/history",    icon: Clock,    label: "HISTORY"    },
  { to: "/monitoring", icon: Activity, label: "MONITOR"    },
  { to: "/status",     icon: Radio,    label: "STATUS"     },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav style={styles.nav}>
      {/* Logo */}
      <div style={styles.logo}>
        <Plane size={18} color="var(--amber)" strokeWidth={1.5} />
        <span style={styles.logoText}>NOTAM</span>
        <span style={styles.logoSub}>CLASSIFIER</span>
        <span style={styles.logoVersion}>v1.0</span>
      </div>

      {/* Desktop nav */}
      <div style={styles.links}>
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            style={({ isActive }) => ({
              ...styles.link,
              ...(isActive ? styles.linkActive : {}),
            })}
          >
            <Icon size={13} strokeWidth={2} />
            <span>{label}</span>
          </NavLink>
        ))}
      </div>

      {/* Status indicator */}
      <div style={styles.statusBar}>
        <span style={styles.statusDot} />
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--low)" }}>
          SYS ONLINE
        </span>
      </div>

      {/* Mobile toggle */}
      <button style={styles.toggle} onClick={() => setOpen(!open)}>
        {open ? <X size={18} /> : <Menu size={18} />}
      </button>

      {/* Mobile menu */}
      {open && (
        <div style={styles.mobileMenu}>
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              style={({ isActive }) => ({
                ...styles.mobileLink,
                ...(isActive ? styles.mobileLinkActive : {}),
              })}
              onClick={() => setOpen(false)}
            >
              <Icon size={14} />
              <span>{label}</span>
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  );
}

const styles = {
  nav: {
    position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
    display: "flex", alignItems: "center", gap: 8,
    padding: "0 24px", height: 52,
    background: "rgba(8, 11, 15, 0.95)",
    borderBottom: "1px solid var(--border)",
    backdropFilter: "blur(12px)",
  },
  logo: {
    display: "flex", alignItems: "center", gap: 8,
    marginRight: 24, flexShrink: 0,
  },
  logoText: {
    fontFamily: "var(--font-display)", fontWeight: 700,
    fontSize: 16, letterSpacing: "0.12em", color: "var(--amber)",
  },
  logoSub: {
    fontFamily: "var(--font-display)", fontWeight: 400,
    fontSize: 13, letterSpacing: "0.08em", color: "var(--text-secondary)",
  },
  logoVersion: {
    fontFamily: "var(--font-mono)", fontSize: 10,
    color: "var(--text-dim)", marginLeft: 4,
  },
  links: { display: "flex", gap: 2, flex: 1 },
  link: {
    display: "flex", alignItems: "center", gap: 6,
    padding: "6px 12px", borderRadius: 4,
    textDecoration: "none", fontFamily: "var(--font-display)",
    fontWeight: 600, fontSize: 12, letterSpacing: "0.08em",
    color: "var(--text-secondary)",
    transition: "all 0.15s ease",
  },
  linkActive: {
    color: "var(--amber)",
    background: "var(--amber-glow)",
    borderBottom: "1px solid var(--amber)",
  },
  statusBar: { display: "flex", alignItems: "center", gap: 6, flexShrink: 0 },
  statusDot: {
    width: 6, height: 6, borderRadius: "50%",
    background: "var(--low)",
    animation: "pulse-amber 2s infinite",
    boxShadow: "0 0 6px var(--low)",
  },
  toggle: {
    display: "none", background: "none", border: "none",
    color: "var(--text-primary)", cursor: "pointer",
  },
  mobileMenu: {
    display: "none",
    position: "absolute", top: 52, left: 0, right: 0,
    background: "var(--bg-secondary)",
    borderBottom: "1px solid var(--border)",
    padding: 12, gap: 4, flexDirection: "column",
  },
  mobileLink: {
    display: "flex", alignItems: "center", gap: 8,
    padding: "10px 16px", borderRadius: 4,
    textDecoration: "none",
    fontFamily: "var(--font-display)", fontWeight: 600, fontSize: 13,
    color: "var(--text-secondary)",
  },
  mobileLinkActive: { color: "var(--amber)", background: "var(--amber-glow)" },
};