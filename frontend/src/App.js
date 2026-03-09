import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import Navbar       from "./components/Navbar";
import ClassifyPage from "./pages/ClassifyPage";
import BatchPage    from "./pages/BatchPage";
import HistoryPage  from "./pages/HistoryPage";
import MonitoringPage from "./pages/MonitoringPage";
import StatusPage   from "./pages/StatusPage";
import "./index.css";

export default function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "var(--bg-card)",
            color: "var(--text-primary)",
            border: "1px solid var(--border)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
          },
          success: { iconTheme: { primary: "var(--low)", secondary: "#000" } },
          error:   { iconTheme: { primary: "var(--critical)", secondary: "#000" } },
        }}
      />
      <Navbar />
      <main>
        <Routes>
          <Route path="/"           element={<ClassifyPage />} />
          <Route path="/batch"      element={<BatchPage />} />
          <Route path="/history"    element={<HistoryPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/status"     element={<StatusPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}