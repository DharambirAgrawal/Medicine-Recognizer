import React, { useState, useEffect } from "react";

const languages = [
  { code: "en-US", name: "English", transCode: "en" },
  { code: "es-ES", name: "Spanish", transCode: "es" },
  { code: "fr-FR", name: "French", transCode: "fr" },
  { code: "de-DE", name: "German", transCode: "de" },
  { code: "ja-JP", name: "Japanese", transCode: "ja" },
  { code: "it-IT", name: "Italian", transCode: "it" },
  { code: "hi-IN", name: "Hindi", transCode: "hi" },
  { code: "zh-CN", name: "Chinese", transCode: "zh" },
  { code: "ne-NP", name: "Nepali", transCode: "ne" },
];

const TranslationTab = ({ t }) => {
  const [isListening, setIsListening] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [spokenText, setSpokenText] = useState("");
  const [translatedText, setTranslatedText] = useState("");

  const [sourceLang, setSourceLang] = useState("en-US");
  const [targetLang, setTargetLang] = useState("es-ES");

  // Speech Recognition Setup
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = SpeechRecognition ? new SpeechRecognition() : null;

  useEffect(() => {
    if (recognition) {
      recognition.continuous = false;
      recognition.lang = sourceLang;
      recognition.interimResults = false;

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setSpokenText(transcript);
        handleTranslate(transcript);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error", event.error);
        setIsListening(false);
      };
    }
  }, [sourceLang]); // Re-configure when source language changes

  const toggleListening = () => {
    if (!recognition) {
      alert("Browser does not support Speech Recognition.");
      return;
    }

    if (isListening) {
      recognition.stop();
    } else {
      // Ensure lang is set before starting
      recognition.lang = sourceLang;
      recognition.start();
      setIsListening(true);
      setSpokenText("");
      setTranslatedText("");
    }
  };

  const handleTranslate = async (text) => {
    if (!text) return;
    setIsTranslating(true);

    const sourceCode =
      languages.find((l) => l.code === sourceLang)?.transCode || "en";
    const targetCode =
      languages.find((l) => l.code === targetLang)?.transCode || "es";

    try {
      const response = await fetch(
        `https://api.mymemory.translated.net/get?q=${encodeURIComponent(
          text
        )}&langpair=${sourceCode}|${targetCode}`
      );
      const data = await response.json();

      if (data.responseData) {
        setTranslatedText(data.responseData.translatedText);
      } else {
        setTranslatedText("Could not translate.");
      }
    } catch (error) {
      console.error("Translation error:", error);
      setTranslatedText("Error connecting to translation service.");
    } finally {
      setIsTranslating(false);
    }
  };

  const handleSpeak = () => {
    if (!translatedText) return;
    const utterance = new SpeechSynthesisUtterance(translatedText);
    utterance.lang = targetLang;
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="tab-content translation-tab">
      <h2>{t.voiceTranslator}</h2>

      <div className="controls">
        <div className="control-group">
          <label>{t.speakIn}</label>
          <select
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
          >
            {languages.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>

        <span className="arrow">➡️</span>

        <div className="control-group">
          <label>{t.translateTo}</label>
          <select
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
          >
            {languages.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="translation-area">
        <div className="card input-card">
          <div className="card-header">
            <h3>{t.input}</h3>
            <button
              className="clear-button"
              onClick={() => {
                setSpokenText("");
                setTranslatedText("");
              }}
              title="Clear text"
            >
              ✕
            </button>
          </div>

          <textarea
            className="modern-textarea"
            value={spokenText}
            onChange={(e) => setSpokenText(e.target.value)}
            placeholder={t.placeholder}
          />

          <div className="action-row">
            <button
              className={`mic-button ${isListening ? "listening" : ""}`}
              onClick={toggleListening}
              title="Toggle Microphone"
            >
              {isListening ? t.stop : t.speak}
            </button>

            <button
              className="translate-button"
              onClick={() => handleTranslate(spokenText)}
              disabled={!spokenText}
            >
              {t.translateBtn}
            </button>
          </div>
        </div>

        <div className="card output-card">
          <div className="card-header">
            <h3>{t.translation}</h3>
          </div>

          <div className="translation-result">
            {isTranslating ? (
              <div className="loading-pulse">{t.translating}</div>
            ) : (
              <p className="result-text">
                {translatedText || t.translationPlaceholder}
              </p>
            )}
          </div>

          <button
            className="speak-button"
            onClick={handleSpeak}
            disabled={!translatedText}
          >
            {t.playAudio}
          </button>
        </div>
      </div>
    </div>
  );
};

export default TranslationTab;
