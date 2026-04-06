/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: 'var(--surface)',
          dim: 'var(--surface-dim)',
          bright: 'var(--surface-bright)',
          variant: 'var(--surface-variant)',
          container: {
            DEFAULT: 'var(--surface-container)',
            low: 'var(--surface-container-low)',
            lowest: 'var(--surface-container-lowest)',
            high: 'var(--surface-container-high)',
            highest: 'var(--surface-container-highest)',
          },
        },
        'on-surface': {
          DEFAULT: 'var(--on-surface)',
          variant: 'var(--on-surface-variant)',
        },
        primary: {
          DEFAULT: 'var(--primary)',
          container: 'var(--primary-container)',
        },
        'on-primary': {
          DEFAULT: 'var(--on-primary)',
          container: 'var(--on-primary-container)',
        },
        secondary: {
          DEFAULT: 'var(--secondary)',
          container: 'var(--secondary-container)',
        },
        'on-secondary': {
          DEFAULT: 'var(--on-secondary)',
        },
        outline: {
          DEFAULT: 'var(--outline)',
          variant: 'var(--outline-variant)',
        },
        error: {
          DEFAULT: 'var(--error)',
          container: 'var(--error-container)',
        },
        'on-error': {
          DEFAULT: 'var(--on-error)',
        },
        tertiary: {
          DEFAULT: 'var(--tertiary)',
          fixed: 'var(--tertiary-fixed)',
        },
      },
      fontFamily: {
        headline: ['Space Grotesk', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
        label: ['Outfit', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        tamil: ['Noto Sans Tamil', 'sans-serif'],
      },
      borderRadius: {
        DEFAULT: '0px',
        none: '0px',
        sm: '2px',
        md: '4px',
        lg: '0px',
        xl: '0px',
        full: '9999px',
      },
      animation: {
        'pulse-ring': 'pulse-ring 2s infinite',
      },
      keyframes: {
        'pulse-ring': {
          '0%': { boxShadow: '0 0 0 0 rgba(44, 57, 201, 0.7)' },
          '70%': { boxShadow: '0 0 0 15px rgba(44, 57, 201, 0)' },
          '100%': { boxShadow: '0 0 0 0 rgba(44, 57, 201, 0)' },
        },
      },
    },
  },
  plugins: [],
}
