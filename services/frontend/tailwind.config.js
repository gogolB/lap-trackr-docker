/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          50: "#eef2ff",
          100: "#dde4ff",
          200: "#c3cfff",
          300: "#99abff",
          400: "#6b7eff",
          500: "#4854ff",
          600: "#3030f5",
          700: "#2621d8",
          800: "#1e1cae",
          900: "#1b1d89",
          950: "#0f1035",
        },
        slate: {
          850: "#1a2332",
          925: "#111827",
          950: "#0b1120",
        },
      },
    },
  },
  plugins: [],
};
