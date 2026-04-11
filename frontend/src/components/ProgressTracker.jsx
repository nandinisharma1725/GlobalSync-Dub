/**
 * frontend/src/components/ProgressTracker.jsx
 * Visual step-by-step progress display for a dubbing job.
 */

const STEPS = [
  { key: "extracting",   label: "Extracting audio",      desc: "FFmpeg strips audio · pyannote detects speakers" },
  { key: "transcribing", label: "Transcribing speech",   desc: "Whisper converts speech to text with timestamps" },
  { key: "translating",  label: "Translating",           desc: "GPT-4o translates with length-aware prompting" },
  { key: "synthesizing", label: "Cloning voices",        desc: "ElevenLabs clones each speaker and synthesizes dubbed audio" },
  { key: "syncing",      label: "Syncing & exporting",   desc: "DTW time-stretch + FFmpeg final video mux" },
  { key: "done",         label: "Done",                  desc: "Your dubbed video is ready to download" },
];

function stepState(stepKey, currentStatus) {
  const stepIndex = STEPS.findIndex(s => s.key === stepKey);
  const currentIndex = STEPS.findIndex(s => s.key === currentStatus);
  if (currentStatus === "failed") return "error";
  if (stepIndex < currentIndex)  return "done";
  if (stepIndex === currentIndex) return "active";
  return "pending";
}

export default function ProgressTracker({ status, percent }) {
  return (
    <div>
      {/* Progress bar */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 500, color: "var(--muted)" }}>
            {STATUS_LABELS[status] || "Processing…"}
          </span>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--accent)" }}>
            {percent}%
          </span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${percent}%` }} />
        </div>
      </div>

      {/* Step list */}
      <div className="stack-sm">
        {STEPS.map((step, i) => {
          const state = stepState(step.key, status);
          return (
            <StepRow key={step.key} step={step} state={state} index={i} />
          );
        })}
      </div>
    </div>
  );
}

function StepRow({ step, state, index }) {
  const colors = {
    done:    { icon: "#16a34a", bg: "#dcfce7", text: "var(--text)" },
    active:  { icon: "var(--accent)", bg: "#dbeafe", text: "var(--text)" },
    pending: { icon: "var(--border)", bg: "transparent", text: "var(--muted)" },
    error:   { icon: "var(--danger)", bg: "#fee2e2", text: "var(--text)" },
  };
  const c = colors[state] || colors.pending;

  return (
    <div style={{
      display: "flex",
      alignItems: "flex-start",
      gap: 12,
      padding: "10px 14px",
      borderRadius: 8,
      background: state === "active" || state === "done" ? c.bg : "transparent",
      opacity: state === "pending" ? 0.5 : 1,
      transition: "all 0.3s",
    }}>
      {/* Icon */}
      <div style={{
        width: 26, height: 26,
        borderRadius: "50%",
        background: c.icon,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        marginTop: 2,
      }}>
        {state === "done" && <CheckIcon />}
        {state === "active" && <SpinnerIcon />}
        {state === "pending" && (
          <span style={{ fontSize: 11, fontWeight: 600, color: "#fff" }}>
            {index + 1}
          </span>
        )}
        {state === "error" && <XIcon />}
      </div>

      {/* Text */}
      <div>
        <p style={{ fontWeight: 500, color: c.text, fontSize: 14 }}>{step.label}</p>
        {(state === "active" || state === "done") && (
          <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>{step.desc}</p>
        )}
      </div>
    </div>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
      stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
      stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
      stroke="#fff" strokeWidth="2.5" strokeLinecap="round"
      style={{ animation: "spin 1s linear infinite" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <path d="M21 12a9 9 0 11-6.219-8.56" />
    </svg>
  );
}

const STATUS_LABELS = {
  queued:       "Queued — waiting for a worker…",
  starting:     "Starting up…",
  extracting:   "Extracting audio from video…",
  transcribing: "Transcribing speech with Whisper…",
  translating:  "Translating with GPT-4o…",
  synthesizing: "Cloning voices & synthesizing dubbed audio…",
  syncing:      "Syncing audio with video timeline…",
  done:         "Dubbing complete!",
  failed:       "Job failed — see error below",
};