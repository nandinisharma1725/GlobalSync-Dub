import axios from 'axios'

const API_BASE_URL = '/api'

const client = axios.create({
  baseURL: API_BASE_URL,
})

export async function fetchLanguages() {
  const response = await client.get('/languages')
  return response.data
}

export async function createJob(videoFile, languageCode) {
  const formData = new FormData()
  formData.append('video', videoFile)
  formData.append('target_language', languageCode)
  
  const response = await client.post('/jobs', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export async function getJob(jobId) {
  const response = await client.get(`/jobs/${jobId}`)
  return response.data
}

export async function downloadJobResult(jobId) {
  const response = await client.get(`/jobs/${jobId}/download`, {
    responseType: 'blob',
  })
  return response.data
}

export default client
