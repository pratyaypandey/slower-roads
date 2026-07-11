import { defineConfig } from "vite";

// Root is browser/, but main.js imports ../src/*, so allow serving files from
// the sim/ parent. three resolves from node_modules as a normal bare import.
export default defineConfig({
  root: "browser",
  server: {
    fs: { allow: [".."] },
  },
});
