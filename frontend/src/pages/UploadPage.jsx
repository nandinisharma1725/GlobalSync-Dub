/**
 * frontend/src/pages/UploadPage.jsx
 * Main portal page — employee selects video + language and submits.
 */
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import FileDrop from "../components/FileDrop";
import { fetchLanguages, createJob } from "../api";

export default function UploadPage() {
  const navigate = useNavigate();

  const [languages, setLanguages] = useState([]);
  const [selectedLang, setSelectedLang] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  useEffect(() => {
    fetchLanguages().then((langs) => {
      setLanguages(langs);
      // Pre-select Hindi as default (most common MNC India use case)
      const hi = langs.find((l) => l.code === "hi");
      if (hi) setSelectedLang("hi");
    });
  }, []);

  async function handleSubmit() {
    if (!videoFile || !selectedLang) return;
    setUploading(true);
    setUploadError("");

    try {
      const job = await createJob(videoFile, selectedLang);
      navigate(`/jobs/${job.job_id}`, {
        state: {
          languageName: languages.find(l => l.code === selectedLang)?.name,
          filename: videoFile.name,
        },
      });
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Upload failed.";
      setUploadError(msg);
      setUploading(false);
    }
  }

  const langName = languages.find(l => l.code === selectedLang)?.name;
  const canSubmit = videoFile && selectedLang && !uploading;

  return (
    <div className="page">
      <div className="container">

        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 8,
              background: "var(--accent)", display: "flex",
              alignItems: "center", justifyContent: "center",
            }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="23 7 16 12 23 17 23 7"/>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
            </div>
            <h1>Meeting Dubbing Portal</h1>
          </div>
          <p className="muted">
            Upload a board meeting recording. We'll translate and dub it into your
            preferred language, preserving each speaker's voice.
          </p>
        </div>

        <div className="card stack">

          {/* File picker */}
          <div>
            <label>Meeting video</label>
            <FileDrop onFile={setVideoFile} disabled={uploading} />
            {videoFile && (
              <div style={{
                display: "flex", alignItems: "center", gap: 10,
                marginTop: 12, padding: "10px 14px",
                background: "#f0fdf4", border: "1px solid #bbf7d0",
                borderRadius: 8,
              }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                  stroke="#16a34a" strokeWidth="2" strokeLinecap="round">
                  <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                <span style={{ fontSize: 14, color: "#166534" }}>
                  {videoFile.name} ({(videoFile.size / 1e6).toFixed(1)} MB)
                </span>
                {!uploading && (
                  <button
                    onClick={() => setVideoFile(null)}
                    style={{ marginLeft: "auto", background: "none", border: "none",
                      cursor: "pointer", color: "#16a34a", fontSize: 13 }}>
                    Remove
                  </button>
                )}
              </div>
            )}
          </div>

          <hr className="divider" style={{ margin: "4px 0" }} />

          {/* Language selector */}
          <div>
            <label>Preferred language</label>
            <select
              value={selectedLang}
              onChange={(e) => setSelectedLang(e.target.value)}
              disabled={uploading}
            >
              <option value="">Select a language…</option>
              {languages.map((l) => (
                <option key={l.code} value={l.code}>{l.name}</option>
              ))}
            </select>
          </div>

          {/* Submit */}
          <div>
            <button
              className="btn btn-primary"
              style={{ width: "100%", justifyContent: "center", padding: "13px 0" }}
              onClick={handleSubmit}
              disabled={!canSubmit}
            >
              {uploading ? (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke="#fff" strokeWidth="2.5" strokeLinecap="round"
                    style={{ animation: "spin 1s linear infinite" }}>
                    <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
                    <path d="M21 12a9 9 0 11-6.219-8.56"/>
                  </svg>
                  Uploading…
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke="#fff" strokeWidth="2" strokeLinecap="round">
                    <line x1="22" y1="2" x2="11" y2="13"/>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                  </svg>
                  {canSubmit ? `Dub into ${langName}` : "Start dubbing"}
                </>
              )}
            </button>

            {uploadError && (
              <p style={{ color: "var(--danger)", fontSize: 13, marginTop: 10, textAlign: "center" }}>
                {uploadError}
              </p>
            )}
          </div>
        </div>

        {/* Info strip */}
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(3, 1fr)",
          gap: 12, marginTop: 20,
        }}>
          {[
            { icon: "🎙️", title: "Voice preserved", body: "Each speaker's voice is cloned and maintained across languages" },
            { icon: "⏱️", title: "Synced audio", body: "Dynamic time-stretching keeps dubbed speech in sync with the video" },
            { icon: "🔒", title: "Private & secure", body: "Videos are processed on your infrastructure and never shared" },
          ].map((item) => (
            <div key={item.title} style={{
              background: "var(--surface)", border: "1px solid var(--border)",
              borderRadius: 10, padding: "14px 16px",
            }}>
              <div style={{ fontSize: 20, marginBottom: 6 }}>{item.icon}</div>
              <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 3 }}>{item.title}</p>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>{item.body}</p>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}