import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/",           label: "Dashboard" },
  { to: "/text",       label: "Text" },
  { to: "/voice",      label: "Voice" },
  { to: "/facial",     label: "Facial" },
  { to: "/multimodal", label: "Multimodal" },
  { to: "/explain",    label: "Explainability" },
  { to: "/suggest",    label: "Suggestions" },
  { to: "/model",      label: "Model Info" },
  { to: "/ethics",     label: "Ethics & Privacy" },
];

export function NavBar() {
  return (
    <header
      style={{
        background: "#1e3a5f",
        color: "#fff",
        padding: "0 24px",
        display: "flex",
        alignItems: "center",
        gap: 0,
        flexWrap: "wrap",
        minHeight: 52,
      }}
    >
      <span
        style={{ fontWeight: 700, fontSize: 17, marginRight: 24, letterSpacing: "-0.02em", flexShrink: 0 }}
        aria-label="EmotiSense AI home"
      >
        EmotiSense<span style={{ color: "#60a5fa" }}>AI</span>
      </span>
      <nav aria-label="Main navigation" style={{ display: "flex", flexWrap: "wrap", gap: 0 }}>
        {LINKS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            style={({ isActive }) => ({
              color: isActive ? "#bfdbfe" : "#cbd5e1",
              textDecoration: "none",
              padding: "14px 12px",
              fontSize: 13,
              fontWeight: isActive ? 700 : 400,
              borderBottom: isActive ? "2px solid #60a5fa" : "2px solid transparent",
              display: "inline-block",
            })}
            aria-current={undefined}
          >
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
