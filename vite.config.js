import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Merge both configurations into ONE single export
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
})