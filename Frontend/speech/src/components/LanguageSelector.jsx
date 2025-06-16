import React from 'react'
import { FormControl, InputLabel, Select, MenuItem, Grid } from '@mui/material'

const LanguageSelector = ({ 
  inputLanguage, 
  outputLanguage, 
  onInputLanguageChange, 
  onOutputLanguageChange 
}) => {
  const inputLanguages = {
    "auto": "Auto-detect",
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian"
  }

  const outputLanguages = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi"
  }

  return (
    <Grid container spacing={2}>
      <Grid item xs={6}>
        <FormControl fullWidth>
          <InputLabel>Input Language</InputLabel>
          <Select
            value={inputLanguage}
            label="Input Language"
            onChange={(e) => onInputLanguageChange(e.target.value)}
          >
            {Object.entries(inputLanguages).map(([code, name]) => (
              <MenuItem key={code} value={code}>{name}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={6}>
        <FormControl fullWidth>
          <InputLabel>Output Language</InputLabel>
          <Select
            value={outputLanguage}
            label="Output Language"
            onChange={(e) => onOutputLanguageChange(e.target.value)}
          >
            {Object.entries(outputLanguages).map(([code, name]) => (
              <MenuItem key={code} value={code}>{name}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
    </Grid>
  )
}

export default LanguageSelector