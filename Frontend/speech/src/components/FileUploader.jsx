import { useState, useRef } from 'react'
import { Button, Box, CircularProgress, Typography } from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import axios from 'axios'

const FileUploader = ({
  inputLanguage,
  outputLanguage,
  onTranscriptionResult
}) => {
  const [isProcessing, setIsProcessing] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileSelect = (event) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('input_lang', inputLanguage)
    formData.append('output_lang', outputLanguage)

    try {
      const response = await axios.post('http://localhost:8000/transcribe', formData)
      onTranscriptionResult(response.data.text)
    } catch (error) {
      console.error('Error uploading file:', error)
      // Add error handling UI feedback here
    } finally {
      setIsProcessing(false)
      setSelectedFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <input
        type="file"
        ref={fileInputRef}
        accept="audio/*"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Button
          variant="outlined"
          onClick={() => fileInputRef.current?.click()}
          startIcon={<CloudUploadIcon />}
          disabled={isProcessing}
        >
          Select Audio File
        </Button>
        {selectedFile && (
          <Typography variant="body2" color="textSecondary">
            {selectedFile.name}
          </Typography>
        )}
      </Box>
      {selectedFile && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Button
            variant="contained"
            onClick={handleUpload}
            disabled={isProcessing}
          >
            Upload and Transcribe
          </Button>
          {isProcessing && <CircularProgress size={24} />}
        </Box>
      )}
    </Box>
  )
}

export default FileUploader