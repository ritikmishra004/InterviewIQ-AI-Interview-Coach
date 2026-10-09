
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#09090f",
        surface: "#12121c",
        panel: "#191927",
        accent: "#8b5cf6",
      },
    },
  },
  plugins: [],
};
