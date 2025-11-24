import React, { useState } from "react";

const diseases = [
  {
    id: 1,
    name: "Common Cold",
    category: "Viral",
    icon: "🤧",
    symptoms: ["Runny nose", "Sore throat", "Cough", "Congestion"],
    funFact:
      "Adults get an average of 2-3 colds per year, while children can get 12 or more!",
    solution: "Rest, hydration, and over-the-counter cold remedies.",
  },
  {
    id: 2,
    name: "Influenza (Flu)",
    category: "Viral",
    icon: "🤒",
    symptoms: ["Fever", "Chills", "Muscle aches", "Cough", "Congestion"],
    funFact:
      "The word 'influenza' comes from the Italian word for 'influence' of the stars.",
    solution:
      "Antiviral drugs, rest, and plenty of fluids. Vaccination is the best prevention.",
  },
  {
    id: 3,
    name: "Migraine",
    category: "Neurological",
    icon: "🧠",
    symptoms: ["Severe throbbing pain", "Sensitivity to light/sound", "Nausea"],
    funFact: "Migraines are the 3rd most prevalent illness in the world.",
    solution: "Pain relievers, triptans, resting in a dark quiet room.",
  },
  {
    id: 4,
    name: "Allergies",
    category: "Immune",
    icon: "🌸",
    symptoms: ["Sneezing", "Itchy eyes", "Runny nose", "Hives"],
    funFact: "The term 'allergy' was coined in 1906 by Clemens von Pirquet.",
    solution: "Antihistamines, avoiding triggers, and immunotherapy.",
  },
  {
    id: 5,
    name: "Insomnia",
    category: "Sleep",
    icon: "😴",
    symptoms: [
      "Difficulty falling asleep",
      "Waking up often",
      "Daytime tiredness",
    ],
    funFact: "Humans are the only mammals that willingly delay sleep.",
    solution:
      "Sleep hygiene, cognitive behavioral therapy, and sometimes medication.",
  },
  {
    id: 6,
    name: "COVID-19",
    category: "Viral",
    icon: "🦠",
    symptoms: ["Fever", "Cough", "Loss of taste/smell", "Shortness of breath"],
    funFact:
      "The '19' in COVID-19 stands for 2019, the year it was discovered.",
    solution: "Vaccination, isolation, antivirals, and supportive care.",
  },
  {
    id: 7,
    name: "Diabetes (Type 2)",
    category: "Chronic",
    icon: "🩸",
    symptoms: ["Increased thirst", "Frequent urination", "Hunger", "Fatigue"],
    funFact:
      "Diabetes was first identified by the ancient Egyptians around 1500 BC.",
    solution: "Diet, exercise, medication (metformin), and insulin therapy.",
  },
  {
    id: 8,
    name: "Hypertension",
    category: "Chronic",
    icon: "❤️",
    symptoms: ["Headaches", "Shortness of breath", "Nosebleeds", "Often none"],
    funFact:
      "It is often called the 'silent killer' because it rarely has symptoms.",
    solution:
      "Lifestyle changes, low sodium diet, and antihypertensive medication.",
  },
  {
    id: 9,
    name: "Asthma",
    category: "Respiratory",
    icon: "🫁",
    symptoms: [
      "Wheezing",
      "Coughing",
      "Chest tightness",
      "Shortness of breath",
    ],
    funFact:
      "Swimming is often recommended for asthmatics because of the warm, moist air.",
    solution: "Inhalers (bronchodilators, steroids) and avoiding triggers.",
  },
  {
    id: 10,
    name: "Chickenpox",
    category: "Viral",
    icon: "🐔",
    symptoms: ["Itchy rash", "Fever", "Tiredness", "Loss of appetite"],
    funFact:
      "Once you've had chickenpox, the virus stays in your body and can cause shingles later.",
    solution:
      "Calamine lotion, oatmeal baths, and antiviral medication if severe.",
  },
];

const categories = ["All", ...new Set(diseases.map((d) => d.category))];

const DiseaseTab = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");

  const filteredDiseases = diseases.filter((disease) => {
    const matchesSearch = disease.name
      .toLowerCase()
      .includes(searchTerm.toLowerCase());
    const matchesCategory =
      selectedCategory === "All" || disease.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="tab-content disease-tab">
      <h2>Disease Encyclopedia</h2>

      <div className="disease-controls">
        <input
          type="text"
          placeholder="Search diseases..."
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
                <strong>⚠️ Symptoms</strong>
                <ul>
                  {disease.symptoms.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>

              <div className="disease-section fun-fact">
                <strong>💡 Fun Fact</strong>
                <p>{disease.funFact}</p>
              </div>

              <div className="disease-section solution">
                <strong>💊 Solution</strong>
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
