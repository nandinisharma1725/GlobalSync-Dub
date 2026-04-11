/**
 * frontend/src/pages/JobStatusPage.jsx
 * Polls the API for job progress and shows the download button when done.
 */
import { useState, useEffect, useRef } from "react";
import { useParams, useLocation, Link } from "react-router-dom";
import ProgressTracker from "../components/ProgressTracker";
import { getJob } from "../api";

const POLL_INTERVAL_MS = 3000; // poll every 3 seconds

export default function JobStatusPage() {
  const { jobId } = useParams();
  const location = useLocation();
  const { languageName, filename } = location.state || {};

  const [job, setJob] = useState(null);
  const [notFound, setNotFound] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    // Initial fetch
    fetchStatus();

    // Start polling
    intervalRef.current = setInterval(fetchStatus, POLL_INTERVAL_MS);
    return () => clearInterval(intervalRef.current);
  }, [jobId]);

  async function fetchStatus() {
    try {
      const data = await getJob(jobId);
      setJob(data);

      // Stop polling when terminal state reached
      if (data.status === "done" || data.status === "failed") {
        clearInterval(intervalRef.current);
      }
    } catch (err) {
      if (err.response?.status === 404) {
        setNotFound(true);
        clearInterval(intervalRef.current);
      }
    }
  }

  if (notFound) {
    return (
      <div className="page">
        <div className="container">
          <div className="card" style={{ textAlign: "center", padding: "48px 32px" }}>
            <p style={{ fontSize: 32, marginBottom: 12 }}>🔍</p>
            <h2 style={{ marginBottom: 8 }}>Job not found</h2>
            <p className="muted" style={{ marginBottom: 24 }}>
              Job ID <code>{jobId}</code> doesn't exist or has expired.
            </p>
            <Link to="/" className="btn btn-primary">Back to Upload</Link>
          </div>
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="page">
        <div className="container">
          <div className="card" style={{ textAlign: "center", padding: "48px 32px" }}>
            <p className="muted">Loading job status…</p>
          </div>
        </div>
      </div>
    );
  }

  const isDone   = job.status === "done";
  const isFailed = job.status === "failed";

  return (
    <div className="page">
      <div className="container">

        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <Link to="/" style={{ fontSize: 13, color: "var(--muted)", textDecoration: "none",
            display: "inline-flex", alignItems: "center", gap: 4, marginBottom: 16 }}>
            ← New dubbing job
          </Link>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
            <div>
              <h1 style={{ marginBottom: 4 }}>
                {isDone ? "Dubbing complete" : isFailed ? "Dubbing failed" : "Dubbing in progress…"}
              </h1>
              <p className="muted">
                {filename && <><strong>{filename}</strong> → </>}
                {languageName || job.target_language}
              </p>
            </div>
            <StatusBadge status={job.status} />
          </div>
        </div>

        {/* Progress card */}
        <div className="card" style={{ marginBottom: 16 }}>
          <ProgressTracker status={job.status} percent={job.percent || 0} />
        </div>

        {/* Done — download card */}
        {isDone && job.download_url && (
          <div className="card" style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            gap: 16, flexWrap: "wrap",
            background: "#f0fdf4", border: "1px solid #bbf7d0",
          }}>
            <div>
              <p style={{ fontWeight: 600, color: "#166534", marginBottom: 3 }}>
                Your dubbed video is ready
              </p>
              <p style={{ fontSize: 13, color: "#15803d" }}>
                {job.filename || `dubbed_${job.target_language}.mp4`}
              </p>
            </div>
            <a
              href={job.download_url}
              download
              className="btn btn-primary"
              style={{ background: "#16a34a", textDecoration: "none" }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                stroke="#fff" strokeWidth="2" strokeLinecap="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
              Download MP4
            </a>
          </div>
        )}

        {/* Failed — error card */}
        {isFailed && (
          <div className="card" style={{
            background: "#fff1f2", border: "1px solid #fecdd3",
          }}>
            <p style={{ fontWeight: 600, color: "#b91c1c", marginBottom: 6 }}>
              Something went wrong
            </p>
            {job.error && (
              <pre style={{
                fontSize: 12, color: "#991b1b",
                background: "#ffe4e6", padding: "10px 14px",
                borderRadius: 6, overflowX: "auto", whiteSpace: "pre-wrap",
              }}>
                {job.error}
              </pre>
            )}
            <p style={{ fontSize: 13, color: "#b91c1c", marginTop: 12 }}>
              The worker will retry automatically. If this keeps failing, contact your IT admin.
            </p>
          </div>
        )}

        {/* Job details */}
        <div style={{ marginTop: 16, padding: "14px 18px",
          background: "var(--surface)", border: "1px solid var(--border)",
          borderRadius: 10, fontSize: 13, color: "var(--muted)",
          display: "flex", gap: 24, flexWrap: "wrap" }}>
          <span>Job ID: <code style={{ color: "var(--text)" }}>{jobId}</code></span>
          {!isDone && !isFailed && (
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{
                width: 7, height: 7, borderRadius: "50%",
                background: "var(--accent)",
                animation: "pulse 1.5s ease-in-out infinite",
                display: "inline-block",
              }}/>
              <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }`}</style>
              Polling every 3s…
            </span>
          )}
        </div>

      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const map = {
    queued:       ["badge-yellow", "Queued"],
    starting:     ["badge-yellow", "Starting"],
    extracting:   ["badge-blue",   "Extracting"],
    transcribing: ["badge-blue",   "Transcribing"],
    translating:  ["badge-blue",   "Translating"],
    synthesizing: ["badge-blue",   "Synthesizing"],
    syncing:      ["badge-blue",   "Syncing"],
    done:         ["badge-green",  "Done"],
    failed:       ["badge-red",    "Failed"],
  };
  const [cls, label] = map[status] || ["badge-yellow", status];
  return <span className={`badge ${cls}`}>{label}</span>;
}