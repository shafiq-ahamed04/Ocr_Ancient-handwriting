import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ocr': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ocr/palmleaf': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/classify': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/export': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
