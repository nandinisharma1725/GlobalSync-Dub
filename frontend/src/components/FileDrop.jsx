/**
 * frontend/src/components/FileDrop.jsx
 * Drag-and-drop + click-to-browse video file picker.
 */
import { useState, useRef } from "react";

const ACCEPTED = [".mp4", ".mov", ".webm", ".mkv"];
const MAX_MB = 500;

export default function FileDrop({ onFile, disabled }) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef();

  function validate(file) {
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!ACCEPTED.includes(ext)) {
      setError(`Unsupported format. Please upload ${ACCEPTED.join(", ")}.`);
      return false;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`File too large. Maximum is ${MAX_MB} MB.`);
      return false;
    }
    setError("");
    return true;
  }

  function handleFile(file) {
    if (!file) return;
    if (validate(file)) onFile(file);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }

  const borderColor = dragging ? "var(--accent)" : error ? "var(--danger)" : "var(--border)";
  const bgColor = dragging ? "#eff6ff" : "var(--bg)";

  return (
    <div>
      <div
        onClick={() => !disabled && inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        style={{
          border: `2px dashed ${borderColor}`,
          borderRadius: "10px",
          background: bgColor,
          padding: "36px 24px",
          textAlign: "center",
          cursor: disabled ? "not-allowed" : "pointer",
          transition: "all 0.15s",
          opacity: disabled ? 0.5 : 1,
        }}
      >
        {/* Icon */}
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
          stroke="var(--muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ margin: "0 auto 12px", display: "block" }}>
          <path d="M15 10l-4 4m0 0l-4-4m4 4V3"/>
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
        </svg>

        <p style={{ fontWeight: 500, marginBottom: 4 }}>
          Drop your meeting video here
        </p>
        <p className="muted">or click to browse — MP4, MOV, WebM, MKV up to 500 MB</p>
      </div>

      {error && (
        <p style={{ color: "var(--danger)", fontSize: 13, marginTop: 8 }}>{error}</p>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
    </div>
  );
}