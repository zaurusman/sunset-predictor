import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

// Next 16 removed `next lint`; ESLint runs directly against this flat config.
const config = [
  ...nextVitals,
  ...nextTs,
  {
    rules: {
      // New in the React-Compiler-era hooks plugin. It flags nine existing
      // mount-flag / load-from-storage effects; lint never ran before this
      // config, so they predate it. Kept visible as a warning until those
      // components are refactored, rather than failing every lint run now.
      "react-hooks/set-state-in-effect": "warn",
    },
  },
  { ignores: [".next/**", "node_modules/**", "next-env.d.ts"] },
];

export default config;
