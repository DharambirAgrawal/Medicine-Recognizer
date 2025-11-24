import React, { useState, useEffect } from "react";

const DiseaseTab = ({ t }) => {
  const diseases = t.diseases;
  const categories = [t.all, ...new Set(diseases.map((d) => d.category))];

  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState(t.all);

  useEffect(() => {
    setSelectedCategory(t.all);
  }, [t.all]);

  const filteredDiseases = diseases.filter((disease) => {
    const matchesSearch = disease.name
      .toLowerCase()
      .includes(searchTerm.toLowerCase());
    const matchesCategory =
      selectedCategory === t.all || disease.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="tab-content disease-tab">
      <h2>{t.diseaseEncyclopedia}</h2>

      <div className="disease-controls">
        <input
          type="text"
          placeholder={t.searchPlaceholder}
          className="search-bar"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <div className="category-filters">
          {categories.map((cat) => (
            <button
              key={cat}
              className={`filter-btn ${
                selectedCategory === cat ? "active" : ""
              }`}
              onClick={() => setSelectedCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div className="disease-grid">
        {filteredDiseases.map((disease) => (
          <div key={disease.id} className="disease-card">
            <div className="card-header-row">
              <span className="disease-icon">{disease.icon}</span>
              <div className="disease-title-group">
                <h3>{disease.name}</h3>
                <span className="category-badge">{disease.category}</span>
              </div>
            </div>

            <div className="card-body">
              <div className="disease-section">
                <strong>{t.symptoms}</strong>
                <ul>
                  {disease.symptoms.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>

              <div className="disease-section fun-fact">
                <strong>{t.funFact}</strong>
                <p>{disease.funFact}</p>
              </div>

              <div className="disease-section solution">
                <strong>{t.solution}</strong>
                <p>{disease.solution}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DiseaseTab;
