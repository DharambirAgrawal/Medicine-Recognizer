import { useState } from "react";
import "./App.css";
import TranslationTab from "./components/TranslationTab";
import DiseaseTab from "./components/DiseaseTab";

function App() {
  const [activeTab, setActiveTab] = useState("translation");

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Health & Language Assistant</h1>
        <nav className="tab-nav">
          <button
            className={activeTab === "translation" ? "active" : ""}
            onClick={() => setActiveTab("translation")}
          >
            🗣️ Translator
          </button>
          <button
            className={activeTab === "diseases" ? "active" : ""}
            onClick={() => setActiveTab("diseases")}
          >
            🏥 Disease Info
          </button>
        </nav>
      </header>

      <main className="app-content">
        {activeTab === "translation" ? <TranslationTab /> : <DiseaseTab />}
      </main>
    </div>
  );
}

export default App;
