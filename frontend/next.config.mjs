/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow images from common sunset-sharing domains
  images: {
    remotePatterns: [
      { hostname: "i.redd.it" },
      { hostname: "preview.redd.it" },
      { hostname: "i.imgur.com" },
    ],
  },
  experimental: {
    // Disable the default optimizePackageImports list — the SWC autoModularizeImports
    // plugin it activates produces an invalid config on this environment (Node 23 / ARM64).
    optimizePackageImports: [],
  },
};

export default nextConfig;
