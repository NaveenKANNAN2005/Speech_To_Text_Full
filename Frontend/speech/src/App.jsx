import { useState } from 'react'
import { Box, Container, ThemeProvider, createTheme } from '@mui/material'
import axios from 'axios'

import AudioRecorder from './components/AudioRecorder';
import FileUploader from './components/FileUploader';
import TranscriptionOutput from './components/TranscriptionOutput'
import LanguageSelector from './components/LanguageSelector'
import './App.css'
import React from 'react'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('React Error Boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong. Please check the console for errors.</h1>;
    }
    return this.props.children;
  }
}

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
})

function App() {
  console.log('App component rendering');
  const [transcription, setTranscription] = useState('')
  const [inputLanguage, setInputLanguage] = useState('auto')
  const [outputLanguage, setOutputLanguage] = useState('en')
  const [isRecording, setIsRecording] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const handleFileUpload = async (file) => {
    try {
      setIsLoading(true)
      const formData = new FormData()
      formData.append('audio', file)

      const response = await axios.post('http://localhost:5000/transcribe', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      if (response.data.success) {
        setTranscription(response.data.transcription)
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Error processing audio file')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <ErrorBoundary>
      <ThemeProvider theme={theme}>
        <Container maxWidth="md">
          <Box sx={{ my: 4 }}>
            <h1>Speech to Text Converter</h1>
            
            <Box sx={{ mb: 3 }}>
              <LanguageSelector 
                inputLanguage={inputLanguage}
                outputLanguage={outputLanguage}
                onInputLanguageChange={setInputLanguage}
                onOutputLanguageChange={setOutputLanguage}
              />
            </Box>

            <Box sx={{ mb: 3 }}>
              <AudioRecorder 
                isRecording={isRecording}
                setIsRecording={setIsRecording}
                inputLanguage={inputLanguage}
                outputLanguage={outputLanguage}
                onTranscriptionResult={setTranscription}
              />
            </Box>

            <Box sx={{ mb: 3 }}>
              <FileUploader 
                inputLanguage={inputLanguage}
                outputLanguage={outputLanguage}
                onTranscriptionResult={setTranscription}
              />
            </Box>

            <TranscriptionOutput 
              text={transcription}
              onClear={() => setTranscription('')}
              isLoading={isLoading}
            />
          </Box>
        </Container>
      </ThemeProvider>
    </ErrorBoundary>
  )
}

export default App
