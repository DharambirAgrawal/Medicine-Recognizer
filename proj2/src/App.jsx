import { useState, useEffect } from "react";
import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import "./App.css";
import TranslationTab from "./components/TranslationTab";
import DiseaseTab from "./components/DiseaseTab";
import { translations } from "./translations";

function App() {
  const [language, setLanguage] = useState(() => {
    return localStorage.getItem("appLanguage") || "en";
  });

  useEffect(() => {
    localStorage.setItem("appLanguage", language);
  }, [language]);

  const t = translations[language] || translations["en"];

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-top">
          <h1>{t.appTitle}</h1>
          <div className="language-selector">
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="lang-select"
            >
              <option value="en">English</option>
              <option value="es">Español</option>
              <option value="hi">हिंदी</option>
              <option value="ne">नेपाली</option>
            </select>
          </div>
        </div>
        <nav className="tab-nav">
          <NavLink
            to="/translation"
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            {t.navTranslator}
          </NavLink>
          <NavLink
            to="/diseases"
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            {t.navDiseases}
          </NavLink>
        </nav>
      </header>

      <main className="app-content">
        <Routes>
          <Route path="/" element={<Navigate to="/translation" replace />} />
          <Route
            path="/translation"
            element={<TranslationTab t={t} language={language} />}
          />
          <Route
            path="/diseases"
            element={<DiseaseTab t={t} language={language} />}
          />
        </Routes>
      </main>
    </div>
  );
}

export default App;
