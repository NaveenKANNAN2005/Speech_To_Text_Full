import { Paper, Typography } from '@mui/material'

const TranscriptionOutput = ({ text }) => {
  return (
    <Paper 
      elevation={2} 
      sx={{ 
        p: 2,
        minHeight: 100,
        maxHeight: 300,
        overflowY: 'auto',
        backgroundColor: '#f5f5f5'
      }}
    >
      <Typography variant="body1">
        {text || 'Transcription will appear here...'}
      </Typography>
    </Paper>
  )
}

export default TranscriptionOutput