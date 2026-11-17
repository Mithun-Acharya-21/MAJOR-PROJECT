/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Disable Turbopack for production builds (use Webpack) to avoid
  // a Turbopack panic on next/font/local CSS module generation (Next 16.3.3).
  // `npm run dev` still uses Turbopack for fast HMR.
  experimental: {
    turbo: {
      // empty — keep Turbopack config present but let build fall back to Webpack
    },
  },
}

export default nextConfig
