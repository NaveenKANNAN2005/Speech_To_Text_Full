import { useState, useRef } from 'react'
import { Button, Box, CircularProgress } from '@mui/material'
import MicIcon from '@mui/icons-material/Mic'
import StopIcon from '@mui/icons-material/Stop'
import axios from 'axios'

const AudioRecorder = ({
  isRecording,
  setIsRecording,
  inputLanguage,
  outputLanguage,
  onTranscriptionResult
}) => {
  const [isProcessing, setIsProcessing] = useState(false)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error('Error accessing microphone:', error)
      alert('Unable to access microphone. Please ensure you have granted permission.')
    }
  }

  const stopRecording = async () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      setIsProcessing(true)

      await new Promise((resolve) => {
        if (mediaRecorderRef.current) {
          mediaRecorderRef.current.onstop = () => resolve()
        }
      })

      const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' })
      const formData = new FormData()
      formData.append('file', audioBlob, 'recording.wav')
      formData.append('input_lang', inputLanguage)
      formData.append('output_lang', outputLanguage)

      try {
        const response = await axios.post('http://localhost:8000/transcribe', formData)
        if (response.data && response.data.segments) {
          const transcription = response.data.segments
            .map((segment) => segment.text)
            .join(' ')
          onTranscriptionResult(transcription)
        }
      } catch (error) {
        console.error('Transcription error:', error)
        alert('Error processing audio. Please try again.')
      } finally {
        setIsProcessing(false)
      }

      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop())
    }
  }

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
      <Button
        variant="contained"
        color={isRecording ? "secondary" : "primary"}
        onClick={isRecording ? stopRecording : startRecording}
        startIcon={isRecording ? <StopIcon /> : <MicIcon />}
        disabled={isProcessing}
      >
        {isRecording ? 'Stop Recording' : 'Start Recording'}
      </Button>
      {isProcessing && <CircularProgress size={24} />}
    </Box>
  )
}

export default AudioRecorder