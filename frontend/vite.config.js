import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  base: '/',
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/users': {
        target: 'http://localhost:8000',
        bypass: (req) => {
          if (req.headers.accept && req.headers.accept.includes('text/html')) {
            return '/index.html'
          }
        }
      },
      '/user': {
        target: 'http://localhost:8000',
        bypass: (req) => {
          if (req.headers.accept && req.headers.accept.includes('text/html')) {
            return '/index.html'
          }
        }
      },
      '/jobs': 'http://localhost:8000',
      '/upload_replay': 'http://localhost:8000',
      '/delete_replay': 'http://localhost:8000',
      '/delete_map': 'http://localhost:8000',
      '/api': 'http://localhost:8000'
    }
  }
})
