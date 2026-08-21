import { Routes, Route } from "react-router-dom";
import { NavBar } from "@/components/NavBar";
import Dashboard        from "@/pages/Dashboard";
import TextAnalysis     from "@/pages/TextAnalysis";
import VoiceAnalysis    from "@/pages/VoiceAnalysis";
import FacialAnalysis   from "@/pages/FacialAnalysis";
import MultimodalAnalysis from "@/pages/MultimodalAnalysis";
import Explainability   from "@/pages/Explainability";
import Suggestions      from "@/pages/Suggestions";
import ModelInfo        from "@/pages/ModelInfo";
import EthicsPrivacy    from "@/pages/EthicsPrivacy";

export default function App() {
  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", fontFamily: "system-ui, sans-serif" }}>
      <NavBar />
      <Routes>
        <Route path="/"           element={<Dashboard />} />
        <Route path="/text"       element={<TextAnalysis />} />
        <Route path="/voice"      element={<VoiceAnalysis />} />
        <Route path="/facial"     element={<FacialAnalysis />} />
        <Route path="/multimodal" element={<MultimodalAnalysis />} />
        <Route path="/explain"    element={<Explainability />} />
        <Route path="/suggest"    element={<Suggestions />} />
        <Route path="/model"      element={<ModelInfo />} />
        <Route path="/ethics"     element={<EthicsPrivacy />} />
        <Route path="*"           element={<NotFound />} />
      </Routes>
    </div>
  );
}

function NotFound() {
  return (
    <main style={{ textAlign: "center", padding: "80px 16px" }}>
      <h2 style={{ fontSize: 22, marginBottom: 8 }}>Page not found</h2>
      <a href="/" style={{ color: "#2563eb" }}>Return to dashboard</a>
    </main>
  );
}
