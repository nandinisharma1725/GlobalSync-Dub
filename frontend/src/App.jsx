import { Routes, Route } from 'react-router-dom'
import UploadPage from './pages/UploadPage'
import JobStatusPage from './pages/JobStatusPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/jobs/:jobId" element={<JobStatusPage />} />
    </Routes>
  )
}
